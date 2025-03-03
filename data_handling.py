"""This module will contain functions that fetch and store data related to stock data and news."""
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import create_engine
from sqlalchemy import text
import yfinance as yf
import requests
from sentence_transformers import SentenceTransformer
import requests
import os
from ta.momentum import RSIIndicator
from dotenv import load_dotenv

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

embedding_model = SentenceTransformer('all-mpnet-base-v2')

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

def fetch_stock_data(symbol):
    """Fetch stock data for a given symbol."""
    try:
        stock_info = yf.Ticker(symbol).history(period="1y")
        stock_fundamentals = yf.Ticker(symbol).info
        sector = stock_fundamentals.get('sector', None)

        # Extracting PE Ratio and Market Cap
        pe_ratio = stock_fundamentals.get('trailingPE', None)
        market_cap = stock_fundamentals.get('marketCap', None)
        # calculate rsi using 14-day period
        rsi = RSIIndicator(stock_info["Close"], window=14).rsi().iloc[-1] if not stock_info.empty else None
        # Calculate MA50, MA200
        ma50 = stock_info['Close'].rolling(window=50).mean().iloc[-1] if not stock_info.empty else None
        ma200 = stock_info['Close'].rolling(window=200).mean().iloc[-1] if not stock_info.empty else None

        stock_data = {
            'symbol': symbol,
            'sector': sector,
            'price': stock_info['Close'].iloc[-1] if not stock_info.empty else None,
            'rsi': rsi,
            'pe_ratio': pe_ratio,
            'market_cap': market_cap,
            'ma50': ma50,
            'ma200': ma200
        }

        return stock_data

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
    
def fetch_stock_news(ticker, num_articles=6, company_name=None):
    """Get recent news articles related to a stock symbol."""
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize={num_articles}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    news_data = []
    for article in articles:
        news_data.append({
            "title": article["title"],
            "url": article["url"],
            "content": article["description"] or article["content"],
            "source": article["source"]["name"],
            "published_at": article["publishedAt"]
        })

    return news_data

def store_stock_data(symbol):
    """Store stock data in the database."""
    stock_data = fetch_stock_data(symbol)
    # example of stock_data: {'symbol': 'AAPL', 'sector': 'Technology', 'price': np.float64(241.83999633789062), 'rsi': np.float64(53.476956230638855), 'pe_ratio': 38.255943, 'market_cap': 3626259972096, 'ma50': np.float64(240.03136322021484), 'ma200': np.float64(225.4269783782959)} 
    if stock_data is not None:
        query = text("""
            INSERT INTO stocks(symbol, sector, price, rsi, pe_ratio, market_cap, ma50, ma200)
            VALUES(:symbol, :sector, :price, :rsi, :pe_ratio, :market_cap, :ma50, :ma200)
            ON CONFLICT (symbol) DO UPDATE
            SET price = EXCLUDED.price, rsi = EXCLUDED.rsi, pe_ratio = EXCLUDED.pe_ratio, 
            market_cap = EXCLUDED.market_cap, ma50 = EXCLUDED.ma50, ma200 = EXCLUDED.ma200;
        """)

        try:
            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(query, {
                        "symbol": stock_data['symbol'],
                        "sector": stock_data['sector'],
                        "price": float(stock_data['price']),
                        "rsi": float(stock_data['rsi']),
                        "pe_ratio": stock_data['pe_ratio'],
                        "market_cap": stock_data['market_cap'],
                        "ma50": float(stock_data['ma50']),
                        "ma200": float(stock_data['ma200'])
                    })
                print("Stock data stored successfully.")
        except Exception as e:
            print(f"Error inserting stock data: {e}")
    else:
        print(f"No data found for {symbol}")

def store_stock_news(symbol, news_data):
    """Store stock news in the database."""
    query = text("""
        INSERT INTO stock_news(symbol, headline, summary, url, sentiment, source, published_at, embedding)
        VALUES(:symbol, :headline, :summary, :url, :sentiment, :source, :published_at, :embedding)
        ON CONFLICT (symbol, headline) DO NOTHING;
    """)

    try:
        with engine.connect() as conn:
            with conn.begin():
                for article in news_data:
                    article['embedding'] = str(generate_embeddings([article])[0])
                    article['sentiment'] = analyze_sentiment(article['content'])

                    conn.execute(query, {
                        "symbol": symbol,
                        "headline": article['title'],
                        "summary": article['content'],
                        "url": article['url'],
                        "sentiment": article['sentiment'],
                        "source": article['source'],
                        "published_at": article['published_at'],
                        "embedding": article['embedding']
                    })
                print("News data stored successfully.")
    except Exception as e:
        print(f"Error inserting news data: {e}")

# create an embedding function that takes in what is returned from fetch_stock_news and embeds the heading and content
def generate_embeddings(news_data):
    """Generate embeddings for news articles."""
    embeddings = []
    for article in news_data:
        text = f"Title: {article['title']}. Content: {article['content']}"
        embedding = embedding_model.encode(text).tolist()  # Convert NumPy array to list
        embeddings.append(embedding)
    return embeddings

def analyze_sentiment(text):
    """Analyze the sentiment of the text using VADER."""
    sentiment = sentiment_analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return "Positive"
    elif sentiment['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# chunking skipped since length of content is less than 512

# test the function for fetching stock data
#stock_data = fetch_stock_data("AAPL")
#print(stock_data)

# test the function for fetching stock news
#news_data = fetch_stock_news("AAPL")
# print(news_data)

# test the function for storing stock data
# store_stock_data("AAPL")

# test the function for storing stock news
# store_stock_news("AAPL", news_data)

# test the function for generating embeddings