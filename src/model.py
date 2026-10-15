"""
大盤多空分數（0–100）

分數用 T 日收盤後的資料計算，作為 T+1 日的操作參考。
回測時一律假設在 T+1 開盤進場，避免偷看未來。

組成因子（每一個都先壓縮到 -1 ~ +1，正值代表偏多）：
  fx_twse  上市外資現貨：5 日與 20 日累積買賣超的 Z 分數
  dl_twse  上市自營商（自行買賣＋避險）：5 日累積的 Z 分數
  fx_otc   上櫃外資現貨
  dl_otc   上櫃自營商
  fut      外資台指期淨未平倉：水位 Z 分數＋5 日變化
  trend    加權指數趨勢：收盤和 20／60 日均線的關係、60 日均線斜率
  level    位階（反向）：距離 60 日均線越遠越過熱 → 扣分
  capit    恐慌賣壓（反向）：外資 20 日大賣＋指數遠低於季線 → 加分（常見的相對低點）
  distrib  高檔出貨：指數遠高於年線但外資持續賣 → 扣分
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

COMPONENTS = ["trend", "fut", "spot", "bearconf", "capit"]
COMP_LABEL = {
    "trend": "加權指數趨勢", "fut": "外資台指期增減倉", "spot": "外資現貨（上市＋上櫃）",
    "bearconf": "空方共振（趨勢＋期貨＋現貨同步偏空）", "capit": "恐慌賣壓（跌深＋外資大賣 → 加分）",
}
# 權重只依 2017–2022 決定（見 README 的研究過程）
W = {"trend": 2.0, "fut": 2.0, "spot": 1.0, "bearconf": 1.0, "capit": 1.0}
DEFAULT_W = W

TRAIN_END = "2022-12-31"   # 2017–2022 訓練，2023 之後是樣本外
SHOW_START = "2017-01-01"

BANDS = [(80, "強力偏多", 1.0), (60, "偏多", 0.5), (40, "中性", 0.0), (20, "偏空", -0.5), (-1, "強力放空", -1.0)]


def regime(score: float) -> tuple[str, float]:
    for th, name, pos in BANDS:
        if score >= th:
            return name, pos
    return BANDS[-1][1], BANDS[-1][2]


def pos_array(scores: np.ndarray) -> np.ndarray:
    return np.select([scores >= 80, scores >= 60, scores >= 40, scores >= 20], [1.0, 0.5, 0.0, -0.5], -1.0)


# ------------------------------------------------------------------ data
def load_panel() -> pd.DataFrame:
    px = pd.read_csv(DATA / "prices.csv", parse_dates=["date"]).set_index("date")
    df = px.copy()
    for name, pre in (("twse_insti", "twse"), ("tpex_insti", "otc")):
        p = DATA / f"{name}.csv"
        if p.exists():
            t = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            df[f"{pre}_foreign"] = t.get("foreign")
            df[f"{pre}_dealer"] = t.get("dealer")
    p = DATA / "fut_oi.csv"
    if p.exists():
        f = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        df["fut_net"] = f["fut_net"].reindex(df.index).ffill(limit=3)
    for c in ("twse_foreign", "twse_dealer", "otc_foreign", "otc_dealer", "fut_net"):
        if c not in df:
            df[c] = np.nan
    return df.sort_index()


# ------------------------------------------------------------------ features
def z(x: pd.Series, w=250, minp=120) -> pd.Series:
    m = x.rolling(w, min_periods=minp).mean()
    s = x.rolling(w, min_periods=minp).std()
    return (x - m) / s.replace(0, np.nan)


def sq(x):
    return np.tanh(x / 2.0)


def flow(x: pd.Series, long=True) -> pd.Series:
    a = sq(z(x.rolling(5, min_periods=3).sum()))
    if not long:
        return a
    b = sq(z(x.rolling(20, min_periods=15).sum()))
    return 0.5 * a + 0.5 * b


def features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["twii_close"]
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    out = pd.DataFrame(index=df.index)
    out["trend"] = (np.sign(c - ma20) + np.sign(c - ma60) + np.sign(ma60.diff(10)) + np.sign(ma20 - ma60)) / 4.0
    out["fut"] = 0.5 * sq(z(df["fut_net"].diff(5))) + 0.5 * sq(z(df["fut_net"].diff(20)))
    fx = df["twse_foreign"] + df["otc_foreign"].fillna(0)
    fx = fx.where(df["twse_foreign"].notna())
    out["spot"] = 0.5 * sq(z(fx.rolling(5, min_periods=3).sum())) + 0.5 * sq(z(fx.rolling(20, min_periods=10).sum()))
    out["bearconf"] = -((out["trend"] < 0) & (out["fut"] < 0) & (out["spot"] < 0)).astype(float) * out["trend"].abs().clip(upper=1)
    bias60 = c / ma60 - 1
    dd250 = c / c.rolling(250).max() - 1
    zf20 = z(fx.rolling(20, min_periods=10).sum())
    out["capit"] = (np.clip(-zf20 - 0.5, 0, 1.5) / 1.5) * np.clip((-dd250 - 0.08) / 0.10, 0, 1)
    out["guard"] = np.clip((-bias60 - 0.04) / 0.06, 0, 1)      # 跌破季線 4% 起，偏空訊號逐步打折，10% 時歸零
    # 只顯示、不計分：自營商（研究顯示預測力不穩定）
    out["dl_twse"] = sq(z(df["twse_dealer"].rolling(5, min_periods=3).sum()))
    out["dl_otc"] = sq(z(df["otc_dealer"].rolling(5, min_periods=3).sum()))
    out["bias60"], out["bias240"], out["dd250"], out["zf20"] = bias60, c / c.rolling(240).mean() - 1, dd250, zf20
    return out


def score_from(F: pd.DataFrame, w: dict) -> pd.Series:
    lin = ["trend", "fut", "spot", "bearconf"]
    raw = sum(w[k] * F[k].fillna(0.0) for k in lin) / sum(w[k] for k in lin)
    raw = raw.where(raw >= 0, raw * (1 - F["guard"].fillna(0.0)))
    raw = raw + w["capit"] * F["capit"].fillna(0.0)
    s = 50 + 50 * np.tanh(2.0 * raw)
    return s.ewm(span=3, adjust=False).mean()


# ------------------------------------------------------------------ returns
def fwd_returns(df: pd.DataFrame) -> pd.DataFrame:
    """T 日收盤後拿到分數 → T+1 開盤進場。r1 是 T+1 開盤到 T+2 開盤的報酬。"""
    o = df["twii_open"].where(df["twii_open"] > 0, df["twii_close"])
    out = pd.DataFrame(index=df.index)
    out["r1"] = o.shift(-2) / o.shift(-1) - 1
    for h in (5, 20, 60):
        out[f"f{h}"] = o.shift(-1 - h) / o.shift(-1) - 1
    return out


def strat_stats(pos: np.ndarray, r1: np.ndarray, cost=0.0005) -> dict:
    ok = ~np.isnan(r1)
    pos, r1 = pos[ok], r1[ok]
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    ret = pos * r1 - turn * cost
    if len(ret) < 20:
        return {"cagr": None, "sharpe": None, "mdd": None, "ret": ret}
    eq = np.cumprod(1 + ret)
    yrs = len(ret) / 252
    peak = np.maximum.accumulate(eq)
    return {"cagr": float(eq[-1] ** (1 / yrs) - 1), "sharpe": float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(252)),
            "mdd": float((eq / peak - 1).min()), "ret": ret}


# ------------------------------------------------------------------ validation
def zigzag(close: pd.Series, th=0.10) -> list[dict]:
    pts, trend = [], 0
    hi_i = lo_i = close.index[0]
    hi = lo = close.iloc[0]
    for d, p in close.items():
        if p > hi:
            hi, hi_i = p, d
        if p < lo:
            lo, lo_i = p, d
        if trend >= 0 and p <= hi * (1 - th):
            pts.append({"date": hi_i, "type": "peak", "price": float(hi)})
            trend, lo, lo_i = -1, p, d
        elif trend <= 0 and p >= lo * (1 + th):
            pts.append({"date": lo_i, "type": "trough", "price": float(lo)})
            trend, hi, hi_i = 1, p, d
    return pts


def pivot_report(pivots, score: pd.Series, close: pd.Series) -> list[dict]:
    out = []
    idx = score.index
    for pv in pivots:
        d = pv["date"]
        if d not in idx:
            continue
        i = idx.get_loc(d)
        before = score.iloc[max(0, i - 20):i + 1]
        after = score.iloc[i:i + 61]
        if pv["type"] == "trough":
            warn = bool((before <= 40).any())            # 低點前是否已經處於偏空（代表抓到賣壓）
            hit = after[after >= 60]
        else:
            warn = bool((before >= 60).any())
            hit = after[after <= 40]
        lag = int(idx.get_loc(hit.index[0]) - i) if len(hit) else None
        cost = None
        if lag is not None:
            p2 = close.iloc[i + lag]
            cost = float(p2 / pv["price"] - 1)
        out.append({"date": d.strftime("%Y-%m-%d"), "type": pv["type"], "price": pv["price"],
                    "score": round(float(score.iloc[i]), 1), "regime": regime(float(score.iloc[i]))[0], "min20": round(float(before.min()), 1),
                    "max20": round(float(before.max()), 1), "flag_before": warn,
                    "confirm_lag": lag, "move_during_lag": cost})
    return out


def leg_report(pivots, close: pd.Series, pos: pd.Series, r1: pd.Series) -> list[dict]:
    """每一段 ±10% 波段：指數漲跌、照分數操作的報酬、平均部位。"""
    out = []
    pts = [p for p in pivots if p["date"] in close.index]
    for a, b in zip(pts, pts[1:] + [{"date": close.index[-1], "type": "now", "price": float(close.iloc[-1])}]):
        m = (close.index >= a["date"]) & (close.index < b["date"])
        if m.sum() < 3:
            continue
        rr = r1[m].fillna(0).to_numpy()
        pp = pos[m].to_numpy()
        strat = float(np.prod(1 + pp * rr) - 1)
        mkt = float(b["price"] / a["price"] - 1)
        out.append({"start": a["date"].strftime("%Y-%m-%d"), "end": b["date"].strftime("%Y-%m-%d"),
                    "dir": "up" if a["type"] == "trough" else "down", "days": int(m.sum()),
                    "mkt": mkt, "strat": strat, "avg_pos": float(pp.mean()),
                    "open": b["type"] == "now"})
    return out


def regime_table(score: pd.Series, R: pd.DataFrame, mask: pd.Series) -> list[dict]:
    rows = []
    s = score[mask]
    total = s.notna().sum()
    labels = s.map(lambda v: regime(v)[0])
    for _, name, _ in BANDS:
        m = labels == name
        if m.sum() == 0:
            rows.append({"regime": name, "days": 0})
            continue
        r = R.loc[m[m].index]
        row = {"regime": name, "days": int(m.sum()), "share": float(m.sum() / total)}
        for h in (5, 20, 60):
            x = r[f"f{h}"].dropna()
            row[f"avg{h}"] = float(x.mean()) if len(x) else None
            row[f"win{h}"] = float((x > 0).mean()) if len(x) else None
        rows.append(row)
    return rows


def run(write=True) -> dict:
    df = load_panel()
    F = features(df)
    R = fwd_returns(df)
    show = F.index >= pd.Timestamp(SHOW_START)
    train = pd.Series(show & (F.index <= pd.Timestamp(TRAIN_END)), index=F.index)
    test = pd.Series(F.index > pd.Timestamp(TRAIN_END), index=F.index)

    score = score_from(F, W)
    pos = pos_array(score.to_numpy())
    results = {"weights": W}
    for per, m in (("train", train), ("test", test), ("all", pd.Series(show, index=F.index))):
        mm = m.to_numpy()
        st = strat_stats(pos[mm], R["r1"].to_numpy()[mm])
        bh = strat_stats(np.ones(int(mm.sum())), R["r1"].to_numpy()[mm], cost=0)
        sv, fv = score[m], R.loc[m, "f20"]
        ok = sv.notna() & fv.notna()
        results[per] = {k: st[k] for k in ("cagr", "sharpe", "mdd")}
        results[per]["bh"] = {k: bh[k] for k in ("cagr", "sharpe", "mdd")}
        results[per]["ic20"] = float(sv[ok].rank().corr(fv[ok].rank())) if ok.sum() > 30 else None
        results[per]["regimes"] = regime_table(score, R, m)
        results[per]["time_in"] = {n: float((pos[mm] == p).mean()) for _, n, p in BANDS}

    close = df["twii_close"][show]
    zz = zigzag(close, 0.10)
    pivots = pivot_report(zz, score, df["twii_close"])
    legs = leg_report(zz, close, pd.Series(pos, index=F.index)[show], R["r1"][show])

    # 權益曲線（全期間）
    r1 = R["r1"].to_numpy()
    m = show & ~np.isnan(r1)
    turn = np.abs(np.diff(np.concatenate([[0.0], pos[m]])))
    eq_s = np.cumprod(1 + pos[m] * r1[m] - turn * 0.0005)
    eq_b = np.cumprod(1 + r1[m])

    # 個別因子單獨的預測力（20 日 IC），讓使用者知道哪個因子真的有用
    comp_ic = {}
    for k in COMPONENTS + ["dl_twse", "dl_otc"]:
        for per, mm in (("train", train), ("test", test)):
            a, b = F.loc[mm, k], R.loc[mm, "f20"]
            ok = a.notna() & b.notna() & (a != 0)
            comp_ic.setdefault(k, {})[per] = float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 60 else None

    last = df.index[-1]
    latest = {
        "date": last.strftime("%Y-%m-%d"),
        "score": round(float(score.iloc[-1]), 1),
        "prev": round(float(score.iloc[-2]), 1),
        "regime": regime(score.iloc[-1])[0],
        "twii": float(df["twii_close"].iloc[-1]),
        "twoii": float(df["twoii_close"].iloc[-1]) if "twoii_close" in df else None,
        "components": {k: {"label": COMP_LABEL[k], "value": None if pd.isna(F[k].iloc[-1]) else round(float(F[k].iloc[-1]), 3),
                           "weight": W[k]} for k in COMPONENTS},
        "guard": round(float(F["guard"].iloc[-1]), 3), "dd250": float(F["dd250"].iloc[-1]),
        "raw": {k: (None if pd.isna(df[k].iloc[-1]) else float(df[k].iloc[-1]))
                for k in ("twse_foreign", "twse_dealer", "otc_foreign", "otc_dealer", "fut_net")},
        "bias60": float(F["bias60"].iloc[-1]), "bias240": float(F["bias240"].iloc[-1]),
    }
    fresh = {}
    for k in ("twse_foreign", "otc_foreign", "fut_net", "twii_close"):
        s = df[k].dropna()
        fresh[k] = s.index[-1].strftime("%Y-%m-%d") if len(s) else None

    sub = df[show]
    series = {
        "date": [d.strftime("%Y-%m-%d") for d in sub.index],
        "twii": [round(float(x), 2) for x in sub["twii_close"]],
        "twoii": [None if pd.isna(x) else round(float(x), 2) for x in sub["twoii_close"]],
        "score": [None if pd.isna(x) else round(float(x), 1) for x in score[show]],
        "fx": [None if pd.isna(x) else round(float(x), 1) for x in (sub["twse_foreign"].fillna(0) + sub["otc_foreign"].fillna(0)).where(sub["twse_foreign"].notna())],
        "fut": [None if pd.isna(x) else int(x) for x in sub["fut_net"]],
    }
    eq_dates = [d.strftime("%Y-%m-%d") for d in F.index[m]]
    out = {
        "generated_at": pd.Timestamp.now("UTC").tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
        "latest": latest, "freshness": fresh, "weights": W, "results": results,
        "component_ic": comp_ic, "component_labels": {**COMP_LABEL, "dl_twse": "上市自營商（僅參考）", "dl_otc": "上櫃自營商（僅參考）"},
        "pivots": pivots, "legs": legs, "series": series,
        "equity": {"date": eq_dates, "strat": [round(float(x), 4) for x in eq_s], "bh": [round(float(x), 4) for x in eq_b]},
        "bands": [{"th": th, "name": n, "pos": p} for th, n, p in BANDS],
        "train_end": TRAIN_END,
    }
    if write:
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs" / "data.json").write_text(json.dumps(clean(out), ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"[model] latest {latest['date']} score {latest['score']} {latest['regime']}")
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == "__main__":
    run()
