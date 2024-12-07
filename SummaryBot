import yfinance as yf
from newsapi import NewsApiClient
import openai
import matplotlib.pyplot as plt

# Add your OpenAI API key here
openai.api_key = 'OPENAI_API_KEY'

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
    newsapi = NewsApiClient(api_key='NEWS_API_KEY')
    articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy')
    return articles['articles'][:5]

# Function to summarize news
def generate_summary(news_articles):
    prompt = "Summarize the following stock news:\n"
    for article in news_articles:
        prompt += f"Title: {article['title']}\nContent: {article['description']}\n\n"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

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
    summary = generate_summary(news)
    print("News Summary:")
    print(summary)

    # Plot stock data
    plot_stock_data(ticker)

# Run the bot for a given ticker symbol
if __name__ == "__main__":
    stock_summary_bot("AAPL")
