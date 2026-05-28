import time
import schedule
from datetime import datetime
from pathlib import Path
from main import run_pipeline

Path("output").mkdir(exist_ok=True)
LOG_FILE = "output/log.txt"


def _log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def scheduled_run():
    try:
        video_path = run_pipeline()
        _log(f"SUCCESS: {video_path}")
    except Exception as e:
        _log(f"ERROR: {e}")


if __name__ == "__main__":
    _log("Scheduler started — will run daily at 08:00")
    schedule.every().day.at("08:00").do(scheduled_run)
    while True:
        schedule.run_pending()
        time.sleep(60)
