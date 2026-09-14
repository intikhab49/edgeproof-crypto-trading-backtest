"""MEXC research desk: live scan, empirical forecast and causal replay. No private API."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as stats
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from market_engine import (VERSION, CONFIG, validate_bars, features, forecast_at,
                           forecast_replay, setups_at, setup_replay)
from trade_math import calculate

BASE = "https://api.mexc.com/api/v1/contract"
ROOT = Path(__file__).resolve().parents[1] / "evidence-v2"


def utc(ts=None):
    return datetime.fromtimestamp(ts if ts is not None else time.time(), timezone.utc).isoformat()


def get(endpoint, params=None):
    url = BASE + "/" + endpoint + ("?"+urlencode(params) if params else "")
    start = time.time()
    for attempt in range(2):
        try:
            with urlopen(Request(url, headers={"User-Agent":"crypto-evidence/2.0"}),timeout=10) as r:
                raw = r.read()
            break
        except HTTPError:
            raise
        except (URLError, TimeoutError):
            if attempt:
                raise
            time.sleep(.5)
    data = json.loads(raw)
    if data.get("success") is not True or data.get("code",0) != 0:
        raise ValueError("MEXC response error: " + str(data.get("code")))
    return dict(url=url, requested_at=utc(start), received_at=utc(), payload=data["data"],
                response_sha256=hashlib.sha256(raw).hexdigest())


def symbol_name(value):
    s = value.upper().replace("/", "_")
    if "_" not in s:
        s = s[:-4]+"_USDT" if s.endswith("USDT") else s+"_USDT"
    if not re.fullmatch(r"[A-Z0-9]+_USDT",s):
        raise ValueError("Only linear USDT contract symbols are supported")
    return s


def normalize(raw, cutoff, seconds=900):
    keys = ("time","open","high","low","close","vol")
    if len({len(raw[k]) for k in keys}) != 1:
        raise ValueError("Mismatched candle arrays")
    bars = []
    for i,t in enumerate(raw["time"]):
        if int(t)+seconds > cutoff:
            continue
        # Use documented OHLC fields; optional real* field semantics are not assumed.
        b = {k:float(raw[k][i]) for k in ("open","high","low","close")}
        b.update(time=int(t),volume=float(raw["vol"][i]))
        bars.append(b)
    validate_bars(bars,seconds)
    if cutoff-(bars[-1]["time"]+seconds) >= seconds+10:
        raise ValueError("Latest closed candle missing/stale")
    return bars


def book_stats(payload):
    bids = sorted([(float(x[0]),float(x[1])) for x in payload["bids"] if float(x[1])>0],reverse=True)
    asks = sorted([(float(x[0]),float(x[1])) for x in payload["asks"] if float(x[1])>0])
    if not bids or not asks or bids[0][0]>=asks[0][0]:
        raise ValueError("Empty/crossed book")
    if any(not math.isfinite(v) or v<=0 for row in bids+asks for v in row):
        raise ValueError("Invalid depth values")
    mid = (bids[0][0]+asks[0][0])/2
    band = min(.005,(mid-bids[-1][0])/mid,(asks[-1][0]-mid)/mid)
    bv = sum(p*q for p,q in bids if p>=mid*(1-band))
    av = sum(p*q for p,q in asks if p<=mid*(1+band))
    return dict(mid=mid,spread_bps=(asks[0][0]-bids[0][0])/mid*10000,
                common_band_pct=band*100, full_half_percent_coverage=band>=.005-1e-12,
                notional_imbalance=(bv-av)/(bv+av) if bv+av else None,
                timestamp=payload.get("timestamp"),version=payload.get("version"))


def collect(symbol):
    raw, errors = {}, {}
    clock = get("ping")
    raw["clock"] = clock
    cutoff = float(clock["payload"])/1000
    endpoints = {"15m":("kline/"+symbol,{"interval":"Min15"}),
                 "1h":("kline/"+symbol,{"interval":"Min60"}),
                 "4h":("kline/"+symbol,{"interval":"Hour4"}),
                 "contract":("detail/country",{"symbol":symbol}),
                 "funding":("funding_rate/"+symbol,None),
                 "trades":("deals/"+symbol,{"limit":100})}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(get,*args):name for name,args in endpoints.items()}
        for job in as_completed(jobs):
            name = jobs[job]
            try:
                raw[name] = job.result()
            except Exception as exc:
                errors[name] = str(exc)
    books = []
    for i in range(3):
        try:
            rec = get("depth/"+symbol,{"limit":100})
            raw["depth_"+str(i)] = rec
            books.append(book_stats(rec["payload"]))
        except Exception as exc:
            errors["depth_"+str(i)] = str(exc)
        if i<2:
            time.sleep(.25)
    for name,endpoint,params in [("ticker","ticker",{"symbol":symbol})]:
        try:
            raw[name] = get(endpoint,params)
        except Exception as exc:
            errors[name] = str(exc)
    bars = {}
    for tf,seconds in [("15m",900),("1h",3600),("4h",14400)]:
        if tf in raw:
            try:
                bars[tf] = normalize(raw[tf]["payload"],cutoff,seconds)
            except Exception as exc:
                errors[tf] = str(exc)
    return dict(symbol=symbol,venue="MEXC",instrument="USDT perpetual",as_of_utc=utc(cutoff),
                as_of_epoch=cutoff,collected_at_utc=utc(),raw=raw,errors=errors,bars=bars,books=books)


def analyze(snapshot, fee_bps=None, slippage_bps=2, funding_bps=0, replay=True):
    raw, bars = snapshot["raw"], snapshot["bars"]
    errors = dict(snapshot["errors"])
    result = dict(version=VERSION,symbol=snapshot["symbol"],as_of_utc=snapshot["as_of_utc"],
                  errors=errors, status="DATA LIMITED", forecasts=[], candidates=[],
                  experimental=True, leverage_feasibility="unknown; requires account liquidation reference")
    if "15m" not in bars or len(bars["15m"]) < 180:
        errors["history"] = "Need at least 180 valid closed 15m candles"
        return result
    b = bars["15m"]
    fs = features(b)
    if fs[-1] is None:
        errors["volatility"] = "Insufficient nonzero volatility for analysis"
        return result
    result["context"] = {tf:features(rows)[-1] for tf,rows in bars.items()}
    result["data_window"] = dict(start_utc=utc(b[0]["time"]),end_utc=utc(b[-1]["time"]+900),bars=len(b))
    for h in (1,4,16):
        f = forecast_at(b,fs,len(b)-1,h)
        if f:
            f["origin_price"] = b[-1]["close"]
            f["origin_time_utc"] = utc(b[-1]["time"]+900)
            f["target_time_utc"] = utc(b[-1]["time"]+900+h*900)
            if replay:
                f["replay"] = forecast_replay(b,fs,h)
            result["forecasts"].append(f)
    ticker = raw.get("ticker",{}).get("payload",{})
    contract = raw.get("contract",{}).get("payload",{})
    if isinstance(contract,list):
        contract = next((x for x in contract if x.get("symbol")==snapshot["symbol"]),{})
    if contract.get("symbol") != snapshot["symbol"] or contract.get("state") != 0:
        errors["contract"] = "Missing, wrong, or inactive contract"
    if fee_bps is None:
        public_fee = contract.get("takerFeeRate")
        fee_bps = float(public_fee)*10000 if public_fee is not None else None
        fee_source = "public base rate, not verified account tier"
    else:
        fee_source = "user-supplied per-side assumption"
    if fee_bps is None:
        errors["fee"] = "Missing fee assumption"
    result["cost_model"] = dict(fee_bps_per_side=fee_bps,fee_source=fee_source,
                                slippage_bps_per_side=slippage_bps,funding_bps_per_trade=funding_bps,
                                note="slippage assumption includes spread in historical replay; stress test costs; account fees may differ")
    result["ticker"] = ticker
    result["contract"] = {k:contract.get(k) for k in ("contractSize","priceUnit","volUnit","minVol","stopOnlyFair","feeRateMode")}
    result["funding"] = raw.get("funding",{}).get("payload")
    if result["funding"] and float(result["funding"].get("nextSettleTime",0))/1000 < snapshot["as_of_epoch"]+3600:
        result["funding_warning"] = "Funding may occur within 60m; stress test --funding-bps"
    local_now = time.time()
    clock = raw["clock"]
    clock_received = datetime.fromisoformat(clock["received_at"]).timestamp()
    exchange_now = snapshot["as_of_epoch"]+max(0,local_now-clock_received)
    for name,payload in [("ticker",ticker)]+[("depth",x) for x in snapshot["books"][-1:]]:
        timestamp = payload.get("timestamp")
        if timestamp is None or not -2 <= exchange_now-float(timestamp)/1000 <= 10:
            errors[name+"_freshness"] = "Missing or stale source timestamp (10-second budget)"
    if not snapshot["books"]:
        errors["depth"] = "No valid depth"
    else:
        books = snapshot["books"]
        valid = [x["notional_imbalance"] for x in books if x["notional_imbalance"] is not None]
        result["microstructure"] = dict(depth_samples=len(books),unique_versions=len({x["version"] for x in books}),
                                        common_band_pct_min=min(x["common_band_pct"] for x in books),
                                        imbalance_mean=stats.mean(valid) if valid else None,
                                        imbalance_min=min(valid) if valid else None,
                                        imbalance_max=max(valid) if valid else None,
                                        note="short REST observation only; varying band coverage; not proof of wall persistence or future flow")
    trades = raw.get("trades",{}).get("payload",[])
    if trades:
        vol = sum(float(t["v"]) for t in trades)
        result["recent_tape"] = dict(count=len(trades),oldest_timestamp=min(t["t"] for t in trades),
                                    latest_timestamp=max(t["t"] for t in trades),
                                    buy_contract_ratio=sum(float(t["v"]) for t in trades if t["T"]==1)/vol if vol else None,
                                    note="last up-to-100 trades, not a full 15m CVD series")
    plans = setups_at(b,fs,len(b)-1)
    result["status"] = "DATA LIMITED" if errors else "WATCH" if plans else "NO SETUP"
    for p in plans:
        p = dict(p)
        side = p["side"]
        quote = ticker.get("ask1" if side=="long" else "bid1")
        p.update(entry_rule="market at refreshed executable quote after this closed-bar signal; cancel if barrier already crossed",
                 entry_expiry_utc=utc(b[-1]["time"]+1800),
                 exit_rule="100% at first stop/target or 60 minutes after fill",
                 stop_trigger_reference="last; requires separate fair-price model if contract/account mandates fair",
                 status="WATCH - experimental, recheck quotes/expiry before entry")
        if contract.get("stopOnlyFair"):
            p["status"] = "DATA LIMITED - fair-price-only stop requires fair-price history"
        if quote is not None and fee_bps is not None:
            try:
                p["current_quote"] = float(quote)
                p["risk"] = calculate(side,float(quote),p["stop"],p["target"],1,fee_bps,fee_bps,slippage_bps,float(quote)*funding_bps/10000)
                if p["risk"]["target_net_R"] <= 0:
                    p["status"] = "REJECT - costs consume target"
            except ValueError:
                p["status"] = "REJECT - current quote crossed barrier"
        if errors:
            p["status"] = "DATA LIMITED - see source errors"
        result["candidates"].append(p)
    if replay and fee_bps is not None:
        result["setup_replay"] = setup_replay(b,fs,fee_bps,slippage_bps,funding_bps)
    return result


def save_unique(folder,label,value):
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/(label+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:10]+".json")
    data = json.dumps(value,indent=2,allow_nan=False)
    with path.open("x",encoding="utf-8") as f:
        f.write(data+"\n")
    return path,hashlib.sha256((data+"\n").encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symbols",nargs="*",default=["BTC_USDT","ETH_USDT","SOL_USDT"])
    p.add_argument("--replay-file",type=Path,help="Reanalyze a saved snapshot offline; it remains historical")
    p.add_argument("--output",type=Path,default=ROOT)
    p.add_argument("--fee-bps",type=float)
    p.add_argument("--slippage-bps",type=float,default=2)
    p.add_argument("--funding-bps",type=float,default=0)
    p.add_argument("--quick",action="store_true",help="Skip historical validation; never claim calibrated accuracy")
    args = p.parse_args()
    for name in ("fee_bps","slippage_bps","funding_bps"):
        value = getattr(args,name)
        if value is not None and (not math.isfinite(value) or not 0<=value<10000):
            p.error(name+" must be finite and between 0 and 10000 bps")
    if len(args.symbols)>20:
        p.error("At most 20 symbols per scan")
    reports = []
    for symbol in (["offline"] if args.replay_file else args.symbols):
        try:
            if args.replay_file:
                snapshot = json.loads(args.replay_file.read_text(encoding="utf-8"))
                source_path = args.replay_file
                source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                for tf,seconds in [("15m",900),("1h",3600),("4h",14400)]:
                    if tf in snapshot["bars"]:
                        validate_bars(snapshot["bars"][tf],seconds)
            else:
                snapshot = collect(symbol_name(symbol))
                source_path,source_hash = save_unique(args.output/"snapshots",snapshot["symbol"],snapshot)
            report = analyze(snapshot,args.fee_bps,args.slippage_bps,args.funding_bps,not args.quick)
            if args.replay_file:
                report["status"] = "HISTORICAL REPLAY"
                for candidate in report["candidates"]:
                    candidate["status"] = "HISTORICAL CANDIDATE - not current"
            report.update(snapshot_path=str(source_path),snapshot_sha256=source_hash,config=CONFIG)
            report["implementation_sha256"] = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("crypto_desk.py","market_engine.py")}
            path,_ = save_unique(args.output/"reports",snapshot["symbol"],report)
            reports.append(report)
            print(json.dumps(dict(symbol=report["symbol"],status=report["status"],report=str(path),
                                  errors=report["errors"],regime=(report.get("context",{}).get("15m") or {}).get("regime"),
                                  candidates=[dict(family=x["family"],side=x["side"],status=x["status"],net_target_R=x.get("risk",{}).get("target_net_R")) for x in report["candidates"]],
                                  forecasts=[dict(minutes=f["horizon_minutes"],p_up=round(f["probability_terminal_up"],3),brier_skill=f.get("replay",{}).get("brier_skill")) for f in report["forecasts"]]),allow_nan=False),flush=True)
        except Exception as exc:
            reports.append(dict(symbol=symbol,status="DATA LIMITED",error=str(exc)))
            print(json.dumps(reports[-1]),flush=True)
    ranking = sorted([dict(symbol=r["symbol"],family=c["family"],side=c["side"],net_target_R=c["risk"]["target_net_R"])
                      for r in reports if r["status"]=="WATCH" for c in r.get("candidates",[]) if c["status"].startswith("WATCH") and "risk" in c],
                     key=lambda x:x["net_target_R"],reverse=True)
    scan = dict(version=VERSION,created_at_utc=utc(),ranking=ranking,
                ranking_basis="net target/risk geometry only, not probability or expected profitability; correlated symbols not independent",
                reports=reports)
    path,_ = save_unique(args.output/"reports","scan",scan)
    print(json.dumps(dict(scan_report=str(path),ranking=ranking)),flush=True)
    if not any("context" in r for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
