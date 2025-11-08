from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from .market_data import MarketDataLoader

def schedule_market_data_refresh():
    print("Refreshing market data...")
    market_data_loader = MarketDataLoader()
    market_data_loader.load_call_option_chain("aapl")
    market_data_loader.load_put_option_chain("aapl")
    print("Market data refreshed.")

def start_auto_refresher():
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        schedule_market_data_refresh,
        trigger=IntervalTrigger(seconds=30)
    )

    scheduler.start()