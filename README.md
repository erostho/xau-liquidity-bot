# XAU Liquidity-Pool Telegram Bot (PRO) ⭐️

This bot replies in Telegram with **BUY / SELL / WAIT**, a ⭐ rating, and **Entry / SL / TP**.
TPs are computed using **Liquidity Pool targets** (equal highs/lows, swing pools) + structure.

> Note: XAU is OTC (no centralized orderbook). "BUY limit lớn / SELL limit lớn" is inferred from price action.

## Features
- Webhook-based Telegram bot (ideal for **Render Free**)
- Multi-layer PRO logic: context, session, volatility (ATR), trap risk, MTF conflict
- ⭐ scoring (0–5)
- **Liquidity Pool TP** (Method #5): targets likely liquidity (equal highs/lows, recent swing pools)
- Safety: bot does **not** place orders

---

## 1) Run locally
1) Install
```bash
pip install -r requirements.txt
```

2) Configure
```bash
cp .env.example .env
# fill TELEGRAM_BOT_TOKEN
# choose DATA_PROVIDER=mock OR twelvedata (and TWELVEDATA_API_KEY)
```

3) Start
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 2) Deploy on Render FREE (Webhook)
### A) Push to GitHub
Create a repo, push this folder.

### B) Create Render Web Service
- New + -> Web Service
- Select your GitHub repo
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Add env vars in Render:
  - `TELEGRAM_BOT_TOKEN`
  - `DATA_PROVIDER` = `twelvedata` (or `mock`)
  - `TWELVEDATA_API_KEY` (if twelvedata)
  - optional `TELEGRAM_SECRET_TOKEN`

### C) Set Telegram webhook
After deploy you have a URL like:
`https://your-service.onrender.com`

Set webhook (run once):
```bash
curl -X POST "https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook" \
  -d "url=https://your-service.onrender.com/telegram/webhook"
```

If you set `TELEGRAM_SECRET_TOKEN`, include it:
```bash
curl -X POST "https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook" \
  -d "url=https://your-service.onrender.com/telegram/webhook" \
  -d "secret_token=<YOUR_SECRET>"
```

---

## 3) Use in Telegram
Examples:
- `NÊN BUY HAY SELL?`
- `ANALYZE XAUUSD 15m risk=15`
- `XAUUSD 30m risk=20`

The bot replies with:
- Recommendation + stars
- Liquidity pools (TP targets)
- Entry/SL/TP plan

---

## Data provider note
- `mock` works without any API key.
- `twelvedata` needs `TWELVEDATA_API_KEY`.

