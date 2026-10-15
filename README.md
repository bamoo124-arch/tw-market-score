# 大盤多空分數

每天盤後自動抓資料，算出下一個交易日的大盤多空分數（0–100），並驗證這個分數在 2017 年以來各個轉折點的準確度。

| 分數 | 判斷 | 回測部位 |
|---|---|---|
| ≥ 80 | 強力偏多 | 100% 做多 |
| 60–80 | 偏多 | 50% 做多 |
| 40–60 | 中性 | 空手 |
| 20–40 | 偏空 | 50% 放空 |
| < 20 | 強力放空 | 100% 放空 |

## 資料來源

| 資料 | 主要來源 | 備援 |
|---|---|---|
| 上市三大法人買賣超 | FinMind（首次回補）、證交所 BFI82U（每日） | 互為備援 |
| 上櫃三大法人買賣超 | 櫃買中心（新版、舊版網址都會試） | – |
| 外資台指期淨未平倉 | FinMind | 期交所 futContractsDateDown |
| 加權、櫃買指數 | Yahoo Finance（yfinance） | FinMind |

## 自動更新

`.github/workflows/daily.yml` 在台北時間週一到週五 17:40 執行，20:10 再補跑一次（以防有資料晚出）。
流程是：`src/fetch.py` 增量抓資料到 `data/` → `src/model.py` 算分數和回測 → 寫出 `docs/data.json` → commit 回 repo。
GitHub Pages 從 `docs/` 發布網頁。

第一次執行要回補 2016 年以來的上櫃逐日資料，大約需要 1 小時。之後每天只抓新的一天，幾分鐘就跑完。

## 選用：FinMind token

不設定也能跑。如果 FinMind 回傳次數限制的錯誤，可以到 finmindtrade.com 免費註冊，
把 token 存成 repo 的 Secret，名稱設為 `FINMIND_TOKEN`。

## 本機測試

```bash
pip install -r requirements.txt
python tests/make_fake_data.py data   # 只用合成資料測試流程
python src/model.py
python -m http.server -d docs          # 打開 http://localhost:8000
```

這是統計工具，不是投資建議。
