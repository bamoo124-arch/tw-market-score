"""
資料抓取（增量快取到 data/*.csv）

  data/prices.csv      加權指數 (^TWII)、櫃買指數 (^TWOII) 的 OHLC
  data/twse_insti.csv  上市三大法人買賣超（億元）
  data/tpex_insti.csv  上櫃三大法人買賣超（億元）
  data/fut_oi.csv      外資台指期 (TX) 多空未平倉淨口數

每個資料集都有主要來源和備援來源；某個來源壞掉時不會讓整個流程中斷，
只會在 log 留下警告。
"""
from __future__ import annotations

import io
import os
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

FETCH_START = "2016-01-01"          # 多抓一年當作指標暖機期
FINMIND = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")

S = requests.Session()
S.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def log(*a):
    print("[fetch]", *a, flush=True)


def today_tw() -> dt.date:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date()


def num(x) -> float:
    if x is None:
        return float("nan")
    s = str(x).replace(",", "").replace(" ", "").strip()
    if s in ("", "-", "--", "N/A"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def get_json(url, params=None, method="GET", data=None, tries=3, sleep=3):
    for i in range(tries):
        try:
            if method == "GET":
                r = S.get(url, params=params, timeout=40)
            else:
                r = S.post(url, params=params, data=data, timeout=40)
            if r.status_code == 200 and r.text.strip():
                return r.json()
            log(f"HTTP {r.status_code} {url}")
        except Exception as e:  # noqa: BLE001
            log(f"error {type(e).__name__}: {e} {url}")
        time.sleep(sleep * (i + 1))
    return None


def finmind(dataset, start, end=None, data_id=None) -> pd.DataFrame:
    params = {"dataset": dataset, "start_date": start}
    if end:
        params["end_date"] = end
    if data_id:
        params["data_id"] = data_id
    if FINMIND_TOKEN:
        S.headers["Authorization"] = f"Bearer {FINMIND_TOKEN}"
    j = get_json(FINMIND, params=params)
    if not j or j.get("status") != 200:
        log(f"FinMind {dataset} {start}~{end} failed: {j and j.get('msg')}")
        return pd.DataFrame()
    return pd.DataFrame(j.get("data", []))


def load(name) -> pd.DataFrame:
    p = DATA / f"{name}.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        return df.sort_values("date").drop_duplicates("date", keep="last")
    return pd.DataFrame()


def save(name, df: pd.DataFrame):
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df.to_csv(DATA / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    log(f"saved {name}: {len(df)} rows, last {df['date'].max().date() if len(df) else None}")


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new is None or new.empty:
        return old
    if old.empty:
        return new
    return pd.concat([old, new]).sort_values("date").drop_duplicates("date", keep="last")


# ---------------------------------------------------------------- prices
def fetch_prices():
    old = load("prices")
    start = FETCH_START if old.empty else (old["date"].max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    parts = []
    for fm_id, yf_sym, col in (("TAIEX", "^TWII", "twii"), ("TPEx", "^TWOII", "twoii")):
        part = pd.DataFrame()
        # 主要：FinMind（官方 OHLC，逐年抓避免單次太大）
        chunks, y = [], pd.Timestamp(start)
        end = pd.Timestamp(today_tw())
        while y <= end:
            y2 = min(y + pd.DateOffset(years=3), end)
            c = finmind("TaiwanStockPrice", y.strftime("%Y-%m-%d"), y2.strftime("%Y-%m-%d"), data_id=fm_id)
            if not c.empty:
                chunks.append(c)
            y = y2 + pd.Timedelta(days=1)
        if chunks:
            c = pd.concat(chunks)
            part = pd.DataFrame({"date": pd.to_datetime(c["date"]),
                                 f"{col}_open": c["open"].astype(float), f"{col}_high": c["max"].astype(float),
                                 f"{col}_low": c["min"].astype(float), f"{col}_close": c["close"].astype(float)})
            part = part[part[f"{col}_close"] > 0]
            log(f"{col}: FinMind {len(part)} rows")
        # 備援：yfinance
        if part.empty:
            try:
                import yfinance as yf
                h = yf.Ticker(yf_sym).history(start=start, auto_adjust=False)
                h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
                h = h[h["Volume"] > 0] if col == "twii" else h
                part = h[["Open", "High", "Low", "Close"]].rename(columns=lambda x: f"{col}_{x.lower()}")
                part = part.rename_axis("date").reset_index()
                log(f"{col}: yfinance {len(part)} rows")
            except Exception as e:  # noqa: BLE001
                log(f"{col}: yfinance failed {e}")
        if not part.empty:
            parts.append(part.drop_duplicates("date").set_index("date"))
    new = pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame()
    if "twii_close" not in new:
        new = pd.DataFrame()
    if not new.empty:
        new["date"] = pd.to_datetime(new["date"])
        new = new.dropna(subset=["twii_close"])
    save("prices", merge(old, new))


# ---------------------------------------------------------------- 法人名稱對照
def classify(name: str) -> str | None:
    n = str(name).replace(" ", "").replace("　", "")
    if n.startswith("外資自營商"):          # 注意「外資及陸資(不含外資自營商)」也含此字串
        return "foreign_dealer"
    if "外資" in n or "陸資" in n or "Foreign_Investor" in n:
        return "foreign"
    if "Foreign_Dealer" in n:
        return "foreign_dealer"
    if "投信" in n or "Investment_Trust" in n:
        return "trust"
    if "避險" in n or "Dealer_Hedging" in n:
        return "dealer_hedge"
    if "自行買賣" in n or "Dealer_self" in n:
        return "dealer_self"
    if n in ("自營商", "Dealer", "自營商合計") or n.startswith("自營商(合計") or n == "自營商(合計)":
        return "dealer_total"
    if "合計" in n or n.lower() == "total":
        return "total"
    return None


def rows_to_record(date, rows, scale=1.0) -> dict:
    """rows: [(name, buy, sell, net)] → {'date', 'foreign', 'dealer', ...} 單位億元"""
    rec = {"date": pd.Timestamp(date)}
    for name, buy, sell, net in rows:
        k = classify(name)
        if not k or k in rec:
            continue
        v = net if net == net else buy - sell
        rec[k] = v * scale / 1e8
    if "dealer_total" in rec and "dealer_self" not in rec:
        rec["dealer"] = rec["dealer_total"]
    else:
        rec["dealer"] = rec.get("dealer_self", 0.0) + rec.get("dealer_hedge", 0.0)
    return rec


# ---------------------------------------------------------------- 上市三大法人
def twse_day(d: dt.date) -> dict | None:
    j = get_json("https://www.twse.com.tw/rwd/zh/fund/BFI82U",
                 params={"type": "day", "dayDate": d.strftime("%Y%m%d"), "response": "json"})
    if not j or j.get("stat") != "OK" or not j.get("data"):
        return None
    rows = [(r[0], num(r[1]), num(r[2]), num(r[3])) for r in j["data"]]
    return rows_to_record(d, rows)


def fetch_twse(trading_days: list[dt.date]):
    old = load("twse_insti")
    have = set(old["date"].dt.date) if not old.empty else set()
    missing = [d for d in trading_days if d not in have]
    if not missing:
        log("twse_insti up to date")
        return
    new_recs = []
    # 缺很多天（首次回補）→ 用 FinMind 一次抓完
    if len(missing) > 30:
        fm = finmind("TaiwanStockTotalInstitutionalInvestors", missing[0].isoformat())
        if not fm.empty:
            fm["net"] = fm["buy"].astype(float) - fm["sell"].astype(float)
            for d, g in fm.groupby("date"):
                rows = [(r["name"], r["buy"], r["sell"], r["net"]) for _, r in g.iterrows()]
                new_recs.append(rows_to_record(d, rows))
            got = {pd.Timestamp(r["date"]).date() for r in new_recs}
            missing = [d for d in missing if d not in got]
            log(f"twse FinMind backfill {len(new_recs)} days, still missing {len(missing)}")
    for d in missing[-400:]:          # 剩下的逐日向證交所補
        rec = twse_day(d)
        if rec:
            new_recs.append(rec)
        time.sleep(2.2)               # 證交所限流約每 5 秒 3 次
    if new_recs:
        save("twse_insti", merge(old, pd.DataFrame(new_recs)))


# ---------------------------------------------------------------- 上櫃三大法人
def _tpex_rows_from_json(j) -> list:
    tables = []
    if isinstance(j, dict):
        if j.get("aaData"):
            tables.append(j["aaData"])
        for t in j.get("tables", []) or []:
            if t.get("data"):
                tables.append(t["data"])
        if j.get("data") and isinstance(j["data"], list):
            tables.append(j["data"])
    rows = []
    for t in tables:
        for r in t:
            if not isinstance(r, list) or len(r) < 4:
                continue
            # 名稱欄可能在第 0 欄或第 1 欄（有些版本第 0 欄是分類）
            name = r[0] if not str(r[0]).replace(",", "").replace(".", "").lstrip("-").isdigit() else ""
            nums = [num(x) for x in r[1:]]
            if len(str(r[0]).strip()) == 0 and len(r) > 4:
                name, nums = r[1], [num(x) for x in r[2:]]
            nums = [x for x in nums if x == x]
            if name and len(nums) >= 3:
                rows.append((name, nums[0], nums[1], nums[2]))
    return rows


def tpex_day(d: dt.date) -> dict | None:
    roc = f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"
    attempts = [
        ("GET", "https://www.tpex.org.tw/www/zh-tw/insti/summary",
         {"type": "Daily", "date": d.strftime("%Y/%m/%d"), "response": "json"}, None),
        ("POST", "https://www.tpex.org.tw/www/zh-tw/insti/summary", None,
         {"type": "Daily", "date": d.strftime("%Y/%m/%d"), "response": "json"}),
        ("GET", "https://www.tpex.org.tw/web/stock/3insti/3insti_summary/3itrdsum_result.php",
         {"l": "zh-tw", "t": "D", "p": "0", "d": roc, "o": "json"}, None),
    ]
    for method, url, params, data in attempts:
        j = get_json(url, params=params, method=method, data=data, tries=1)
        rows = _tpex_rows_from_json(j) if j else []
        if rows:
            # 單位判斷：外資買進金額正常約 1e10 元；若小於 1e9 視為千元
            buys = [b for n, b, s, x in rows if classify(n) == "foreign"]
            scale = 1000.0 if buys and buys[0] < 1e9 else 1.0
            return rows_to_record(d, rows, scale)
    return None


def fetch_tpex(trading_days: list[dt.date], budget_min: float = 270):
    """櫃買中心會擋太密集的請求：放慢速度，失敗時指數退避（最長 10 分鐘），不中途放棄。"""
    old = load("tpex_insti")
    have = set(old["date"].dt.date) if not old.empty else set()
    missing = [d for d in trading_days if d not in have]
    log(f"tpex missing {len(missing)} days")
    new_recs, fails, backoff = [], 0, 30
    t0 = time.time()
    i = 0
    while i < len(missing):
        if time.time() - t0 > budget_min * 60:
            log(f"tpex: time budget reached, {len(missing) - i} days left for next run")
            break
        d = missing[i]
        rec = tpex_day(d)
        if rec and "foreign" in rec:
            new_recs.append(rec)
            fails, backoff = 0, 30
            i += 1
        else:
            fails += 1
            if fails >= 3:                      # 同一天連續失敗 3 次 → 視為被擋，退避
                log(f"tpex: blocked at {d}, sleep {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 600)
                if backoff >= 600 and fails >= 12:
                    log(f"tpex: give up on {d}, skip")
                    i += 1
                    fails = 0
                continue
        if len(new_recs) >= 50:                 # 定期存檔，逾時也不會白做
            old = merge(old, pd.DataFrame(new_recs))
            save("tpex_insti", old)
            new_recs = []
            log(f"tpex progress {i}/{len(missing)} ({time.time() - t0:.0f}s)")
        time.sleep(2.5)
    if new_recs:
        save("tpex_insti", merge(old, pd.DataFrame(new_recs)))


# ---------------------------------------------------------------- 外資台指期未平倉
def taifex_range(a: dt.date, b: dt.date) -> pd.DataFrame:
    r = None
    for i in range(3):
        try:
            r = S.post("https://www.taifex.com.tw/cht/3/futContractsDateDown", data={
                "queryStartDate": a.strftime("%Y/%m/%d"), "queryEndDate": b.strftime("%Y/%m/%d"),
                "commodityId": "TXF"}, timeout=60)
            if r.status_code == 200 and len(r.content) > 200:
                break
        except Exception as e:  # noqa: BLE001
            log("taifex error", e)
        time.sleep(3 * (i + 1))
    if r is None or r.status_code != 200:
        return pd.DataFrame()
    txt = r.content.decode("big5", errors="ignore")
    try:
        df = pd.read_csv(io.StringIO(txt))
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    who = next((c for c in df.columns if "身份" in c or "身分" in c), None)
    net = next((c for c in df.columns if "未平倉" in c and "淨額" in c and "口數" in c), None)
    if not who or not net or "日期" not in df.columns:
        return pd.DataFrame()
    df = df[df[who].astype(str).str.contains("外資")]
    out = pd.DataFrame({"date": pd.to_datetime(df["日期"]), "fut_net": df[net].map(num)})
    return out


def fetch_futures():
    old = load("fut_oi")
    start = pd.Timestamp(FETCH_START) if old.empty else old["date"].max() - pd.Timedelta(days=7)
    end = pd.Timestamp(today_tw())
    new = []
    # 主要：FinMind（逐年）
    y = start
    while y <= end:
        y2 = min(y + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        fm = finmind("TaiwanFuturesInstitutionalInvestors", y.strftime("%Y-%m-%d"),
                     y2.strftime("%Y-%m-%d"), data_id="TX")
        if not fm.empty:
            col = "institutional_investors" if "institutional_investors" in fm.columns else fm.columns[2]
            f = fm[fm[col].astype(str).str.contains("外資|Foreign")]
            f = pd.DataFrame({"date": pd.to_datetime(f["date"]),
                              "fut_net": f["long_open_interest_balance_volume"].astype(float)
                              - f["short_open_interest_balance_volume"].astype(float)})
            new.append(f)
        y = y2 + pd.Timedelta(days=1)
        time.sleep(1)
    got = pd.concat(new) if new else pd.DataFrame(columns=["date", "fut_net"])
    last = got["date"].max() if len(got) else start
    # 備援 / 補最新：期交所（逐月）
    if pd.isna(last) or last < end - pd.Timedelta(days=1):
        a = (last if not pd.isna(last) else start).date()
        while a <= end.date():
            b = min(a + dt.timedelta(days=30), end.date())
            got = pd.concat([got, taifex_range(a, b)])
            a = b + dt.timedelta(days=1)
            time.sleep(2)
    # FinMind 只有 2018-06 之後 → 更早的從期交所補
    firsts = [x for x in (got["date"].min() if len(got) else None, old["date"].min() if len(old) else None) if x is not None and not pd.isna(x)]
    first = min(firsts) if firsts else None
    if first is None or first > pd.Timestamp(FETCH_START) + pd.Timedelta(days=30):
        a = pd.Timestamp(FETCH_START).date()
        stop = (first - pd.Timedelta(days=1)).date() if first is not None else end.date()
        while a <= stop:
            b = min(a + dt.timedelta(days=30), stop)
            got = pd.concat([got, taifex_range(a, b)])
            a = b + dt.timedelta(days=1)
            time.sleep(2)
        log(f"futures TAIFEX backfill -> first {got['date'].min()}")
    if len(got):
        got = got.dropna().groupby("date", as_index=False).last()
    save("fut_oi", merge(old, got))


# ---------------------------------------------------------------- main
def main():
    fetch_prices()
    prices = load("prices")
    days = [d.date() for d in prices["date"] if d >= pd.Timestamp(FETCH_START)]
    fetch_twse(days)
    fetch_tpex(days)
    fetch_futures()


if __name__ == "__main__":
    only = sys.argv[1:]
    if only:
        prices = load("prices")
        days = [d.date() for d in prices["date"]] if not prices.empty else []
        for name in only:
            {"prices": fetch_prices, "twse": lambda: fetch_twse(days),
             "tpex": lambda: fetch_tpex(days), "futures": fetch_futures}[name]()
    else:
        main()
