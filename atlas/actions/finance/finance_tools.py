#############################################################################
# equity_tools.py
#
# Here are the APIs that we will use:
#
# - Alpha Vantage API: Used for stock price data (current and historical). Sign up at https://www.alphavantage.co/
# - Finnhub API: Used for company research, equity research, industry/sector research, and company leadership. Sign up at https://finnhub.io/
# - World Bank API: Used for region and country research. Sign up at https://data.worldbank.org/
# - Google Trends API: Used for consumer trends. This is an unofficial API, and you'll need to install the pytrends library. No API key is required.
# - NewsAPI: Used for sentiment analysis and key business news. Sign up at https://newsapi.org/
#
#############################################################################
import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("equity_tools")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import requests
from typing import List, Dict, Any
from datetime import datetime
from urllib.parse import quote
import yfinance as yf
from textblob import TextBlob

class FinanceResearchTool:
    """
    A comprehensive tool for equity research, providing various financial data and analysis.

    This class offers methods to retrieve stock prices, company information, market trends,
    and news related to specific stocks or industries. It utilizes multiple APIs to gather
    diverse financial data.

    Note:
        Ensure all required API keys are set in the environment variables before using this class.
    """

    # Class constant variables for API keys
    ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
    FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
    NEWSAPI_API_KEY = os.environ.get("NEWSAPI_API_KEY")

    def __init__(self):
        self._check_api_keys()

    def _check_api_keys(self):
        api_keys = {
            "ALPHA_VANTAGE_API_KEY": self.ALPHA_VANTAGE_API_KEY,
            "FINNHUB_API_KEY": self.FINNHUB_API_KEY,
            "NEWSAPI_API_KEY": self.NEWSAPI_API_KEY,
        }

        for key_name, key_value in api_keys.items():
            if not key_value:
                _logger.warning(f"{key_name} is missing or invalid. Some functionality may be limited.")

    def search_for_stock_symbol(self, company_name: str) -> dict:
        """
        Search for a stock symbol given the name of a company.

        Args:
            company_name (str): The name of the company.

        Returns:
            str: The stock symbol of the company.

        Raises:
            ValueError: If the API returns an error or if no symbol is found.
        """
        url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={quote(company_name)}&apikey={self.ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        if "bestMatches" not in data or not data["bestMatches"]:
            raise ValueError(f"No stock symbol found for company: {company_name}")

        _logger.info(f"best match: {data['bestMatches'][0]}")

        # Return the first matching symbol
        return {
            "symbol": data["bestMatches"][0]["1. symbol"],
            "name": data["bestMatches"][0]["2. name"],
            "type": data["bestMatches"][0]["3. type"],
            "region": data["bestMatches"][0]["4. region"],
            "currency": data["bestMatches"][0]["8. currency"]
        }

    def get_current_stock_price(self, symbol: str) -> float:
        """
        Retrieve the current stock price for a given symbol.

        Args:
            symbol (str): The stock symbol (e.g., 'AAPL' for Apple Inc.).

        Returns:
            float: The current stock price.

        Raises:
            ValueError: If the API returns an error or if the stock symbol is invalid.
        """
        # Use Alpha Vantage API for current stock price
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        return float(data["Global Quote"]["05. price"])

    def get_historical_stock_prices(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Retrieve historical stock prices for a given symbol within a specified date range.

        Args:
            symbol (str): The stock symbol (e.g., 'AAPL' for Apple Inc.).
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing historical price data.
            Each dictionary has 'date' (str) and 'price' (float) keys.

        Raises:
            ValueError: If the API returns an error, if the stock symbol is invalid,
                        or if no data is found for the specified date range.
        """
        # Use Alpha Vantage API for historical stock prices
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={self.ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        
        if "Time Series (Daily)" not in data:
            raise ValueError(f"Unable to fetch historical data for {symbol}. Response: {data}")
        
        time_series = data["Time Series (Daily)"]
        historical_prices = []
        
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        for date_str, values in time_series.items():
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if start_date <= date <= end_date:
                historical_prices.append({
                    "date": date_str,
                    "price": float(values["4. close"])
                })
        
        if not historical_prices:
            raise ValueError(f"No historical prices found for {symbol} between {start_date.date()} and {end_date.date()}")
        
        return sorted(historical_prices, key=lambda x: x["date"])

    def get_company_research(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve detailed company research information for a given stock symbol.

        Args:
            symbol (str): The stock symbol (e.g., 'AAPL' for Apple Inc.).

        Returns:
            Dict[str, Any]: A dictionary containing various company details such as
            name, industry, market cap, etc. The exact keys may vary based on the
            data available from the Finnhub API.

        Raises:
            ValueError: If the API returns an error or if the stock symbol is invalid.
        """
        # Use Finnhub API for company research
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={self.FINNHUB_API_KEY}"
        response = requests.get(url)
        return response.json()

    def get_equity_research(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve equity research data, including price targets and recommendations.

        Args:
            symbol (str): The stock symbol (e.g., 'AAPL' for Apple Inc.).

        Returns:
            Dict[str, Any]: A dictionary containing equity research data such as
            buy/sell recommendations, price targets, etc. The exact keys may vary
            based on the data available from the Finnhub API.

        Raises:
            ValueError: If the API returns an error or if the stock symbol is invalid.
        """
        # Use Finnhub API for equity research (e.g., price target, recommendations)
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={self.FINNHUB_API_KEY}"
        response = requests.get(url)
        return response.json()

    def get_industry_sector_research(self, industry: str) -> Dict[str, Any]:
        """
        Retrieve industry and sector research data.

        Args:
            industry (str): The industry or sector name (e.g., 'TECH' for technology).

        Returns:
            Dict[str, Any]: A dictionary containing industry research data such as
            peer companies, sector trends, etc. The exact keys may vary based on
            the data available from the Finnhub API.

        Raises:
            ValueError: If the API returns an error or if the industry name is invalid.
        """
        # Use Finnhub API for industry and sector research
        url = f"https://finnhub.io/api/v1/stock/peers?symbol={industry}&token={self.FINNHUB_API_KEY}"
        response = requests.get(url)
        return response.json()

    def get_country_research(self, country_code: str) -> Dict[str, Any]:
        """
        Retrieve economic and financial data for a specific country.

        Args:
            country_code (str): The two-letter ISO country code (e.g., 'US' for United States).

        Returns:
            Dict[str, Any]: A dictionary containing country-specific economic and
            financial data. The exact keys may vary based on the data available
            from the World Bank API.

        Raises:
            ValueError: If the API returns an error or if the country code is invalid.
        """
        # Use World Bank API for region and country research
        url = f"https://api.worldbank.org/v2/country/{country_code}?format=json"
        response = requests.get(url)
        return response.json()

    def get_consumer_trends(self, keyword: str) -> Dict[str, Any]:
        """
        Retrieve consumer trend data for a specific keyword using Google Trends.

        Args:
            keyword (str): The keyword to analyze for consumer trends.

        Returns:
            Dict[str, Any]: A dictionary containing trend data over time for the
            specified keyword. The exact structure depends on the pytrends library output.

        Raises:
            ValueError: If there's an error fetching the trend data or if the keyword is invalid.
        """
        # Use Google Trends API (unofficial, requires separate installation)
        # You'll need to install and import the pytrends library
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe='today 12-m')
        return pytrends.interest_over_time()

    def get_sentiment_analysis(self, query: str, entity_type: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis on recent news articles related to a specific query.

        Args:
            query (str): The search query for news articles (e.g., company name, product).
            entity_type (str): The type of entity being analyzed (e.g., 'company', 'product').

        Returns:
            Dict[str, Any]: A dictionary containing sentiment analysis results.
            Keys include 'overall_sentiment', 'positive_articles', 'neutral_articles',
            'negative_articles', and 'article_count'.

        Raises:
            ValueError: If there's an error fetching news articles or performing sentiment analysis.
        """
        try:
            # Use NewsAPI to fetch recent articles
            url = f"https://newsapi.org/v2/everything?q={quote(query)}&apiKey={self.NEWSAPI_API_KEY}&language=en&sortBy=publishedAt&pageSize=100"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            if not articles:
                raise ValueError(f"No news articles found for the query: {query}")

            # Perform sentiment analysis on each article
            sentiments = []
            for article in articles:
                text = f"{article['title']} {article['description']}"
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)

            # Calculate overall sentiment and counts
            overall_sentiment = sum(sentiments) / len(sentiments)
            positive_count = sum(1 for s in sentiments if s > 0.05)
            neutral_count = sum(1 for s in sentiments if -0.05 <= s <= 0.05)
            negative_count = sum(1 for s in sentiments if s < -0.05)

            return {
                "overall_sentiment": overall_sentiment,
                "positive_articles": positive_count,
                "neutral_articles": neutral_count,
                "negative_articles": negative_count,
                "article_count": len(articles)
            }

        except requests.RequestException as e:
            raise ValueError(f"Error fetching news articles: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error performing sentiment analysis: {str(e)}")

    def get_company_leadership(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieve information about a company's key details including CEO, full company name, industry, 
        sector, full time employees, and website.

        Args:
            ticker (str): The stock ticker symbol of the company (e.g., 'AAPL' for Apple Inc.).

        Returns:
            Dict[str, Any]: A dictionary containing company leadership information.
            Keys include 'CEO', 'Company', 'Industry', 'Sector', 'Full Time Employees', and 'Website'.
            Values are strings, with 'N/A' for unavailable data.

        Raises:
            Exception: If there's an error fetching the company information.
        """
        try:
            # Fetch company information using yfinance
            company = yf.Ticker(ticker)
            
            # Get company information
            info = company.info
            
            # Extract relevant leadership information
            leadership = {
                "CEO": info.get("ceo", "N/A"),
                "Company": info.get("longName", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Full Time Employees": info.get("fullTimeEmployees", "N/A"),
                "Website": info.get("website", "N/A")
            }
            
            return leadership
        except Exception as e:
            print(f"Error fetching company leadership: {e}")
            return {}

    def get_latest_business_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Retrieve recent key business news articles related to a specific stock symbol.

        Args:
            symbol (str): The stock symbol (e.g., 'AAPL' for Apple Inc.).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing details of a news article.
            The exact keys in each dictionary depend on the NewsAPI response structure.

        Raises:
            ValueError: If there's an error fetching news articles or if the symbol is invalid.
        """
        # Use NewsAPI for key business news
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.NEWSAPI_API_KEY}"
        response = requests.get(url)
        return response.json()["articles"]



def main():
    tool = FinanceResearchTool()
    TICKER = "CRM"

    # Test search_for_stock_symbol
    print("\nTesting search_for_stock_symbol:")
    try:
        result_dict = tool.search_for_stock_symbol("Salesforce")
        print(f"Stock symbol for Salesforce: {result_dict}")
    except Exception as e:
        print(f"Error in search_for_stock_symbol: {e}")


    # Test get_current_stock_price
    print("Testing get_current_stock_price:")
    try:
        price = tool.get_current_stock_price(TICKER)
        print(f"Current price of AAPL: ${price:.2f}")
    except Exception as e:
        print(f"Error in get_current_stock_price: {e}")

    # Test get_historical_stock_prices
    print("\nTesting get_historical_stock_prices:")
    try:
        historical_prices = tool.get_historical_stock_prices(TICKER, "2023-01-01", "2023-03-31")
        print(f"Historical prices for AAPL: {len(historical_prices)} entries")
        if historical_prices:
            print(f"First entry: {historical_prices[0]}")
            print(f"Last entry: {historical_prices[-1]}")
    except Exception as e:
        print(f"Error in get_historical_stock_prices: {e}")

    # Test get_company_research
    print("\nTesting get_company_research:")
    try:
        company_research = tool.get_company_research(TICKER)
        print(f"Company research for AAPL: {company_research}")
    except Exception as e:
        print(f"Error in get_company_research: {e}")

    # Test get_equity_research
    print("\nTesting get_equity_research:")
    try:
        equity_research = tool.get_equity_research(TICKER)
        print(f"Equity research for AAPL: {equity_research}")
    except Exception as e:
        print(f"Error in get_equity_research: {e}")

    # Test get_industry_sector_research
    print("\nTesting get_industry_sector_research:")
    try:
        industry_research = tool.get_industry_sector_research("TECH")
        print(f"Industry research for TECH: {industry_research}")
    except Exception as e:
        print(f"Error in get_industry_sector_research: {e}")

    # Test get_region_country_research
    print("\nTesting get_region_country_research:")
    try:
        country_research = tool.get_country_research("US")
        print(f"Country research for US: {country_research}")
    except Exception as e:
        print(f"Error in get_region_country_research: {e}")

    # Test get_consumer_trends
    print("\nTesting get_consumer_trends:")
    try:
        consumer_trends = tool.get_consumer_trends("iPhone")
        print(f"Consumer trends for iPhone: {consumer_trends}")
    except Exception as e:
        print(f"Error in get_consumer_trends: {e}")

    # Test get_sentiment_analysis
    print("\nTesting get_sentiment_analysis:")
    try:
        sentiment = tool.get_sentiment_analysis("Nike", "company")
        print(f"Sentiment analysis for Nike: {sentiment}")
    except Exception as e:
        print(f"Error in get_sentiment_analysis: {e}")

    # Test get_company_leadership
    print("\nTesting get_company_leadership:")
    try:
        leadership = tool.get_company_leadership(TICKER)
        print(f"Company leadership for {TICKER}:")
        for key, value in leadership.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error in get_company_leadership: {e}")

    # Test get_key_business_news
    print("\nTesting get_key_business_news:")
    try:
        news = tool.get_latest_business_news(TICKER)
        print(f"Key business news for {TICKER}: {news[:2]}...")
    except Exception as e:
        print(f"Error in get_key_business_news: {e}")


if __name__ == "__main__":
    main()
