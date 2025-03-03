import time
import requests
import pandas as pd
from data_handling import fetch_stock_news, store_stock_data, store_stock_news, fetch_stock_data

from bs4 import BeautifulSoup

def get_nasdaq100_tickers() -> pd.DataFrame:
    try:
        tickers = []
        url = 'https://www.slickcharts.com/nasdaq100'
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0'  # Default user-agent fails.
        response = requests.get(url, headers={'User-Agent': user_agent})
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})

        if not table:
            print("No table found")
            return []

        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            if len(columns) >= 3:
                ticker = columns[2].text.strip()
                tickers.append(ticker)

        return tickers
    except Exception as e:
        print(f"Error fetching Nasdaq100 tickers: {e}")
        return []
    
def batch_store_stock_data(tickers):
    for ticker in tickers:
        store_stock_data(ticker)
        time.sleep(0.5)

def batch_store_stock_news(tickers):
    for ticker in tickers:
        news_data = fetch_stock_news(ticker)
        store_stock_news(ticker, news_data)
        time.sleep(0.5)
    
def run_batch_processing():
    tickers = get_nasdaq100_tickers()
    batch_store_stock_data(tickers)
    batch_store_stock_news(tickers)
    print("Batch processing completed.")

run_batch_processing()