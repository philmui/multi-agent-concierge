import pytest
from unittest.mock import patch, Mock
from atlas.actions.finance.finance_tools import FinanceResearchTool

@pytest.fixture
def finance_tool():
    return FinanceResearchTool()

def test_get_current_stock_price(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "Global Quote": {
                "05. price": "150.00"
            }
        }
        mock_get.return_value = mock_response

        price = finance_tool.get_current_stock_price("AAPL")
        assert price == 150.00

def test_get_historical_stock_prices(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-01-01": {"4. close": "150.00"},
                "2023-01-02": {"4. close": "155.00"},
            }
        }
        mock_get.return_value = mock_response

        prices = finance_tool.get_historical_stock_prices("AAPL", "2023-01-01", "2023-01-02")
        assert len(prices) == 2
        assert prices[0]["price"] == 150.00
        assert prices[1]["price"] == 155.00

def test_get_company_research(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "Apple Inc.",
            "industry": "Technology"
        }
        mock_get.return_value = mock_response

        research = finance_tool.get_company_research("AAPL")
        assert research["name"] == "Apple Inc."
        assert research["industry"] == "Technology"

def test_get_equity_research(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = [
            {"symbol": "AAPL", "recommendation": "buy"}
        ]
        mock_get.return_value = mock_response

        research = finance_tool.get_equity_research("AAPL")
        assert research[0]["symbol"] == "AAPL"
        assert research[0]["recommendation"] == "buy"

def test_get_industry_sector_research(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = ["AAPL", "MSFT"]
        mock_get.return_value = mock_response

        research = finance_tool.get_industry_sector_research("TECH")
        assert "AAPL" in research
        assert "MSFT" in research

def test_get_region_country_research(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = [{"name": "United States"}]
        mock_get.return_value = mock_response

        research = finance_tool.get_region_country_research("US")
        assert research[0]["name"] == "United States"

def test_get_consumer_trends(finance_tool):
    with patch('pytrends.request.TrendReq') as mock_trendreq:
        mock_pytrends = Mock()
        mock_pytrends.interest_over_time.return_value = {"iPhone": [100, 90, 80]}
        mock_trendreq.return_value = mock_pytrends

        trends = finance_tool.get_consumer_trends("iPhone")
        assert trends["iPhone"] == [100, 90, 80]

def test_get_sentiment_analysis(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "articles": [
                {"title": "Apple is great", "description": "Apple stock is rising."},
                {"title": "Apple is bad", "description": "Apple stock is falling."}
            ]
        }
        mock_get.return_value = mock_response

        sentiment = finance_tool.get_sentiment_analysis("Apple", "company")
        assert "overall_sentiment" in sentiment
        assert "positive_articles" in sentiment
        assert "neutral_articles" in sentiment
        assert "negative_articles" in sentiment
        assert "article_count" in sentiment

def test_get_company_leadership(finance_tool):
    with patch('yfinance.Ticker') as mock_ticker:
        mock_company = Mock()
        mock_company.info = {
            "ceo": "Tim Cook",
            "longName": "Apple Inc.",
            "industry": "Technology",
            "sector": "Consumer Electronics",
            "fullTimeEmployees": 147000,
            "website": "https://www.apple.com"
        }
        mock_ticker.return_value = mock_company

        leadership = finance_tool.get_company_leadership("AAPL")
        assert leadership["CEO"] == "Tim Cook"
        assert leadership["Company"] == "Apple Inc."
        assert leadership["Industry"] == "Technology"
        assert leadership["Sector"] == "Consumer Electronics"
        assert leadership["Full Time Employees"] == 147000
        assert leadership["Website"] == "https://www.apple.com"

def test_get_key_business_news(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "articles": [
                {"title": "Apple releases new product", "description": "Apple has released a new product."},
                {"title": "Apple stock rises", "description": "Apple stock is rising."}
            ]
        }
        mock_get.return_value = mock_response

        news = finance_tool.get_key_business_news("AAPL")
        assert len(news) == 2
        assert news[0]["title"] == "Apple releases new product"
        assert news[1]["title"] == "Apple stock rises"

def test_search_for_stock_symbol(finance_tool):
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "bestMatches": [
                {
                    "1. symbol": "AAPL",
                    "2. name": "Apple Inc.",
                    "3. type": "Equity",
                    "4. region": "United States",
                    "5. marketOpen": "09:30",
                    "6. marketClose": "16:00",
                    "7. timezone": "UTC-05",
                    "8. currency": "USD",
                    "9. matchScore": "1.0000"
                }
            ]
        }
        mock_get.return_value = mock_response

        result_dict = finance_tool.search_for_stock_symbol("Apple")
        print(f"result: {result_dict}")

        assert result_dict["symbol"] == "AAPL"
        assert result_dict["name"] == "Apple Inc."
        assert result_dict["type"] == "Equity"
        assert result_dict["region"] == "United States"
        assert result_dict["currency"] == "USD"