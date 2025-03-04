# RAG-Stock-Screening-Bot

# Overview
This project is a RAG-based LLM stock screener focused on the NASDAQ 100 that retrieves stock data, pulls related news, and generates insightful summaries or recommendations for stock trends based on fundamentals and news sentiment using a LLM model.

# Features
- **Stock Data Retrieval:** Fetches stock data on the NASDAQ 100 stocks (Price, RSI, P/E Ratio, Market Cap, MA50 and MA200) and stores it in PostgreSQL
- **Stock News Retrieval and Embedding:** Uses NewsAPI to retrieve latest news articles based on ticker and embeds the text using a sentence transformer, and stores them as a vector using PostgreSQL vector database
- **Query-Based Stock Screening:** Dynamically screens stocks based on user queries, applying technical and fundamental filters (e.g., "undervalued stocks with strong growth potential")
- **Prompt Formatting:** Constructs a prompt containing selected stock data and relevant news for AI-powered analysis.
- **LLM response generation (In progress):** Uses LLM model to generate a response based on formatted prompt with provided data and generates a natural language response with investment insights.

# Technologies Used
- Python
- PostgreSQL
 
# Next steps
- Improve LLM prompt efficiency to avoid token limitations
- Optimize screening process for more relevant results
- Simple UI for user interactions
