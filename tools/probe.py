"""檢測上櫃與期交所在各年份的回應，結果寫到 logs/probe.txt"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fetch as F
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
for d in ["2016/12/30", "2017/01/03", "2017/06/01", "2017/12/01", "2018/03/14", "2026/10/14"]:
    y, m, dd = d.split("/"); roc = f"{int(y)-1911}/{m}/{dd}"
    for url, kw in [
        ("https://www.tpex.org.tw/www/zh-tw/insti/summary", dict(params={"type": "Daily", "date": d, "response": "json"})),
        ("https://www.tpex.org.tw/web/stock/3insti/3insti_summary/3itrdsum_result.php", dict(params={"l": "zh-tw", "t": "D", "p": "0", "d": roc, "o": "json"})),
    ]:
        try:
            r = F.S.get(url, timeout=30, **kw)
            p("==", d, url.split("/")[-1], r.status_code, r.text[:700].replace("\n", " "))
        except Exception as e:
            p("==", d, url.split("/")[-1], "EXC", repr(e)[:200])
        time.sleep(3)
import datetime as dt
for a, b in [("2017/01/03", "2017/01/31"), ("2026/10/01", "2026/10/14")]:
    try:
        r = F.S.post("https://www.taifex.com.tw/cht/3/futContractsDateDown", data={"queryStartDate": a, "queryEndDate": b, "commodityId": "TXF"}, timeout=60)
        p("== taifex", a, r.status_code, len(r.content), r.content[:400].decode("big5", "ignore").replace("\n", " | "))
    except Exception as e:
        p("== taifex EXC", a, repr(e)[:200])
    time.sleep(3)
Path("logs").mkdir(exist_ok=True); Path("logs/probe.txt").write_text("\n".join(out), encoding="utf-8")
