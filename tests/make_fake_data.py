"""產生合成資料，只用來測試程式能不能正常執行（數值沒有意義）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
dates = pd.bdate_range("2016-01-04", "2026-10-15")
n = len(dates)
drift = np.repeat(rng.normal(0.0004, 0.0015, n // 60 + 1), 60)[:n]
flow = rng.normal(0, 1, n)
ret = drift + 0.004 * pd.Series(flow).rolling(5, min_periods=1).mean().shift(1).fillna(0).to_numpy() + rng.normal(0, 0.009, n)
close = 8300 * np.cumprod(1 + ret)
opn = close / (1 + rng.normal(0, 0.003, n))
pd.DataFrame({"date": dates, "twii_open": opn, "twii_high": close * 1.005, "twii_low": close * 0.995,
              "twii_close": close, "twoii_open": close / 80, "twoii_high": close / 79, "twoii_low": close / 81,
              "twoii_close": close / 80}).to_csv(out / "prices.csv", index=False)
pd.DataFrame({"date": dates, "foreign": flow * 80, "dealer": rng.normal(0, 30, n)}).to_csv(out / "twse_insti.csv", index=False)
pd.DataFrame({"date": dates, "foreign": flow * 10 + rng.normal(0, 5, n), "dealer": rng.normal(0, 5, n)}).to_csv(out / "tpex_insti.csv", index=False)
pd.DataFrame({"date": dates, "fut_net": np.cumsum(flow * 800) - 20000}).to_csv(out / "fut_oi.csv", index=False)
print("fake data", n, "days ->", out)
