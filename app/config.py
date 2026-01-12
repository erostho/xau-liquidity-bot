import os
from dotenv import load_dotenv

load_dotenv()

def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

TELEGRAM_BOT_TOKEN = env("TELEGRAM_BOT_TOKEN")
TELEGRAM_SECRET_TOKEN = env("TELEGRAM_SECRET_TOKEN")

DATA_PROVIDER = (env("DATA_PROVIDER","mock") or "mock").lower()
TWELVEDATA_API_KEY = env("TWELVEDATA_API_KEY")

DEFAULT_SYMBOL = env("DEFAULT_SYMBOL","XAUUSD")
DEFAULT_TIMEFRAME = env("DEFAULT_TIMEFRAME","15m")
DEFAULT_RISK_USD = float(env("DEFAULT_RISK_USD","15"))

MIN_RR = float(env("MIN_RR","1.3"))

TIMEZONE = "Asia/Ho_Chi_Minh"
