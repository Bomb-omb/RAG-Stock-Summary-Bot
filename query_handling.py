"""handle the query from the user"""

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import numpy as np
import os

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

embedding_model = SentenceTransformer('all-mpnet-base-v2')
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

def search_news(query, top_n=5):
    query_embedding = embedding_model.encode(query).tolist()

    search_query = text("""
        SELECT symbol, headline, summary, url, sentiment, source, embedding 
        FROM stock_news 
        ORDER BY embedding <-> :query_embedding
        LIMIT :top_n;
    """)

    try:
        with engine.connect() as conn:
            with conn.begin():
                result = conn.execute(search_query, {"query_embedding": str(query_embedding), "top_n": top_n})
                news_res = result.fetchall()

            formatted_news = []
            for news in news_res:
                formatted_news.append({
                    "symbol": news[0],
                    "headline": news[1],
                    "content": news[2],
                    "url": news[3],
                    "sentiment": news[4]
                })

            return formatted_news
    except Exception as e:
        print(f"Error retrieving news: {e}")
        return []
    
# stock screening based on fundamental and technical analysis
def screen_stocks(query, top_n=5):
    """Dynamically generate SQL query based on user query."""
    filters = get_screening_filters(query)

    if len(filters) > 1:
        filters = " AND ".join(filters)
    elif len(filters) == 1:
        filters = filters[0]

    if not filters:
        filters.append(match_query_to_rule(query))
        filters = filters[0]

    query = f"SELECT symbol, sector, price, rsi, pe_ratio, market_cap, ma50, ma200 FROM stocks WHERE {filters} ORDER BY market_cap DESC LIMIT :top_n;" 

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), {"top_n": top_n})
            stocks = result.fetchall()

            formatted_stocks = []
            for stock in stocks:
                formatted_stocks.append({
                    "symbol": stock[0],
                    "sector": stock[1],
                    "price": stock[2],
                    "rsi": stock[3],
                    "pe_ratio": stock[4],
                    "market_cap": stock[5],
                    "ma50": stock[6],
                    "ma200": stock[7]
                })

            return formatted_stocks
    except Exception as e:
        print(f"Error retrieving stocks: {e}")
        return []

SCREENING_RULES = {
    "undervalued": "pe_ratio < (SELECT AVG(pe_ratio) FROM stocks)",
    "oversold": "rsi < 40",
    "momentum": "price > ma50 AND price > ma200"
}

def get_screening_filters(query):
    filters = []
    for keyword, condition in SCREENING_RULES.items():
        if keyword in query.lower():
            filters.append(condition)
    return filters

def match_query_to_rule(query_text):
    """Use embedding similarity to find the best matching rule if no direct match is found."""
    query_embedding = embedding_model.encode(query_text)
    rule_embeddings = {rule: embedding_model.encode(rule) for rule in SCREENING_RULES.keys()}
    
    # Compute similarity scores
    similarities = {
        rule: np.dot(query_embedding, rule_embedding)
        for rule, rule_embedding in rule_embeddings.items()
    }
    
    # Select the best matching rule
    best_match = max(similarities, key=similarities.get)
    return SCREENING_RULES[best_match]

def handle_user_query(query):
    """Convert user query to LLM context."""
    screened_stocks = screen_stocks(query, top_n=5)
    screened_tickers = [stock['symbol'] for stock in screened_stocks]

    relevant_news = []
    for ticker in screened_tickers:
        news = search_news(ticker, top_n=5)
        relevant_news.extend(news)
    
    context = format_context_for_llm(query, screened_stocks, relevant_news)
    return context
    
def format_context_for_llm(query, stocks, news):
    """Format the context for the LLM."""
    context = f"Query: {query}\n\n"
    
    context += "Stocks:\n"
    for stock in stocks:
        context += f"{stock['symbol']} ({stock['sector']}): Price: {stock['price']}, RSI: {stock['rsi']}, PE Ratio: {stock['pe_ratio']}, Market Cap: {stock['market_cap']}, 50-day MA: {stock['ma50']}, 200-day MA: {stock['ma200']}\n"
    
    context += "\nNews:\n"
    for article in news:
        context += f"{article['headline']} ({article['sentiment']}): {article['content']}\n"
    
    return context.strip()