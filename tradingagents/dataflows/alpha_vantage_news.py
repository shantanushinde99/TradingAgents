from .alpha_vantage_common import _make_api_request, format_datetime_for_api
from datetime import datetime, timedelta

def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
        "sort": "LATEST",
        "limit": "50",
    }
    
    return _make_api_request("NEWS_SENTIMENT", params)

def get_global_news(curr_date, look_back_days=7, limit=50) -> dict[str, str] | str:
    """Returns global market news and sentiment data from Alpha Vantage.

    Retrieves general market news without specific ticker filtering.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles to return (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    # Calculate start date
    if isinstance(curr_date, str):
        end_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    else:
        end_date_dt = curr_date
    
    start_date_dt = end_date_dt - timedelta(days=look_back_days)
    
    params = {
        "time_from": format_datetime_for_api(start_date_dt.strftime("%Y-%m-%d")),
        "time_to": format_datetime_for_api(curr_date if isinstance(curr_date, str) else curr_date.strftime("%Y-%m-%d")),
        "topics": "financial_markets,economy_macro,technology,earnings",  # Broad market topics
        "sort": "LATEST",
        "limit": str(limit),
    }
    
    return _make_api_request("NEWS_SENTIMENT", params)

def get_insider_sentiment(ticker: str, curr_date: str = None) -> dict[str, str] | str:
    """Returns insider sentiment data derived from news sentiment for a company.

    Uses Alpha Vantage NEWS_SENTIMENT to gauge insider and institutional sentiment.

    Args:
        ticker: Ticker symbol. Example: "AAPL".
        curr_date: Current date (optional, used for historical analysis).

    Returns:
        Dictionary containing insider sentiment data or JSON string.
    """
    # Use recent 30 days for sentiment analysis
    if curr_date:
        end_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    else:
        end_date_dt = datetime.now()
    
    start_date_dt = end_date_dt - timedelta(days=30)
    
    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date_dt.strftime("%Y-%m-%d")),
        "time_to": format_datetime_for_api(end_date_dt.strftime("%Y-%m-%d")),
        "topics": "earnings,mergers_and_acquisitions,ipo",
        "sort": "RELEVANCE",
        "limit": "50",
    }
    
    return _make_api_request("NEWS_SENTIMENT", params)

def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)