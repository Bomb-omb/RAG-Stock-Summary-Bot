import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Suppress warning when using transformers
import yfinance as yf
from newsapi import NewsApiClient
import matplotlib.pyplot as plt
import time
import spacy
from chonkie import TokenChunker
from tokenizers import Tokenizer
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, BartConfig
import tensorflow as tf
import logging

tf.get_logger().setLevel('ERROR')
logging.getLogger("transformers").setLevel(logging.ERROR)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to get stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    return {
        "price": data['Close'].iloc[-1],
        "pe_ratio": stock.info.get('trailingPE'),
        "high_52week": stock.info.get('fiftyTwoWeekHigh'),
        "low_52week": stock.info.get('fiftyTwoWeekLow')
    }

# Function to get news
def get_stock_news(query):
    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
    articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy')
    relevant_articles = [
        article for article in articles['articles']
        if query.lower() in article.get('title', '').lower() or query.lower() in article.get('description', '').lower()
    ]
    return relevant_articles[:5]  # Return top 5 relevant articles

# Function to summarize news using T5
def generate_summary(news_articles):
    start_time = time.time()
    # Combine all articles into a single prompt
    prompt = "You are a financial news summarizer. Focus on Apple Inc. (AAPL) and exclude irrelevant topics. Summarize the following:\n"
    for article in news_articles:
        title = article.get('title', 'No Title')
        description = article.get('description', 'No Description')
        prompt += f"Title: {title}\nContent: {description}\n\n"

    # Use the summarizer pipeline
    summary = summarizer(prompt, max_length=150, min_length=30, length_penalty=2.0, num_beams=4)[0]['summary_text']

    end_time = time.time()
    print(f"Summary generated in {end_time - start_time:.2f} seconds")
    
    return summary

def refine_summary(summary):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(summary)
    refined_summary = " ".join([sent.text for sent in doc.sents if "Apple" in sent.text])
    return refined_summary

def analyze_sentiment(summary):
    sentiment = sentiment_analyzer(summary)
    return sentiment

# Function to visualize stock data
def plot_stock_data(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="1mo")
    plt.figure(figsize=(10, 6))
    plt.plot(history.index, history['Close'], label='Close Price')
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Main function to bring everything together
def stock_summary_bot(ticker):
    print(f"Fetching data for {ticker}...\n")

    # Get stock data
    stock_data = get_stock_data(ticker)
    print(f"Current Price: ${stock_data['price']:.2f}")
    print(f"52-Week High: ${stock_data['high_52week']:.2f}")
    print(f"52-Week Low: ${stock_data['low_52week']:.2f}")
    print(f"P/E Ratio: {stock_data['pe_ratio']}\n")

    # Get news and generate summary
    news = get_stock_news(ticker)
    summary = refine_summary(generate_summary(news))
    sentiment = analyze_sentiment(summary)
    print("News Summary:")
    print(summary)
    print(f"Sentiment: {sentiment[0]['label']} ({sentiment[0]['score']:.2f})\n")

    # Plot stock data
    plot_stock_data(ticker)

# Run the bot for a given ticker symbol
if __name__ == "__main__":
    stock_summary_bot("AAPL")