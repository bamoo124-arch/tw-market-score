"""快速檢測各資料來源能否連線，結果寫到 logs/probe.txt"""
import json, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fetch as F
import datetime as dt
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
d = dt.date(2026, 10, 14)
for name, fn in [
  ("twse_day", lambda: F.twse_day(d)),
  ("tpex_day", lambda: F.tpex_day(d)),
  ("tpex_day_2018", lambda: F.tpex_day(dt.date(2018, 3, 14))),
  ("finmind_insti", lambda: F.finmind("TaiwanStockTotalInstitutionalInvestors", "2026-10-01").tail(8).to_dict("records")),
  ("finmind_fut", lambda: F.finmind("TaiwanFuturesInstitutionalInvestors", "2026-10-10", "2026-10-14", data_id="TX").tail(6).to_dict("records")),
  ("taifex", lambda: F.taifex_range(dt.date(2026, 10, 1), d).tail(5).to_dict("records")),
]:
    try:
        p(f"== {name}:", json.dumps(fn(), ensure_ascii=False, default=str)[:1500])
    except Exception:
        p(f"== {name}: EXC", traceback.format_exc()[-800:])
# 櫃買中心原始回應
for url, kw in [
  ("https://www.tpex.org.tw/www/zh-tw/insti/summary", dict(params={"type": "Daily", "date": "2026/10/14", "response": "json"})),
  ("https://www.tpex.org.tw/web/stock/3insti/3insti_summary/3itrdsum_result.php", dict(params={"l": "zh-tw", "t": "D", "p": "0", "d": "115/10/14", "o": "json"})),
]:
    try:
        r = F.S.get(url, timeout=30, **kw); p("== RAW", r.status_code, url, r.text[:1500])
    except Exception as e:
        p("== RAW EXC", url, e)
try:
    import yfinance as yf
    p("== yf", yf.Ticker("^TWII").history(period="5d").tail(2).to_string(), yf.Ticker("^TWOII").history(period="5d").tail(2).to_string())
except Exception as e:
    p("== yf EXC", e)
Path("logs").mkdir(exist_ok=True); Path("logs/probe.txt").write_text("\n".join(out), encoding="utf-8")
