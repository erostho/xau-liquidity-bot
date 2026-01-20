import asyncio
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cron")

# TODO: import đúng hàm bạn đang chạy trong /cron/run
# Ví dụ: from app.pro_analysis import run_scan
# hoặc: from app.main import run_xau_liquidity_once

async def run_job():
    # ✅ gọi logic bot của bạn ở đây
    # await run_scan()
    logger.info("[CRON] run_job: TODO call your bot logic here")

def main():
    logger.info("[CRON] start")
    asyncio.run(run_job())
    logger.info("[CRON] done")

if __name__ == "__main__":
    main()
