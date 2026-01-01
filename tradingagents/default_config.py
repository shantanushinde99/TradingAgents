import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings - Default to OpenRouter instead of OpenAI
    "llm_provider": "openrouter",
    "deep_think_llm": "deepseek/deepseek-chat-v3-0324:free",
    "quick_think_llm": "meta-llama/llama-3.3-8b-instruct:free",
    "backend_url": "https://openrouter.ai/api/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    # Using Alpha Vantage as primary with Yahoo Finance as fallback
    "data_vendors": {
        "core_stock_apis": "alpha_vantage, yfinance",       # Primary: Alpha Vantage, Fallback: Yahoo Finance
        "technical_indicators": "alpha_vantage, yfinance",  # Primary: Alpha Vantage, Fallback: Yahoo Finance
        "fundamental_data": "alpha_vantage, yfinance",      # Primary: Alpha Vantage, Fallback: Yahoo Finance
        "news_data": "alpha_vantage, yfinance",             # Primary: Alpha Vantage, Fallback: Yahoo Finance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Specific tool overrides - all use Alpha Vantage with Yahoo Finance fallback
        "get_stock_data": "alpha_vantage, yfinance",
        "get_indicators": "alpha_vantage, yfinance",
        "get_fundamentals": "alpha_vantage, yfinance",
        "get_balance_sheet": "alpha_vantage, yfinance",
        "get_cashflow": "alpha_vantage, yfinance",
        "get_income_statement": "alpha_vantage, yfinance",
        "get_news": "alpha_vantage, yfinance",
        "get_insider_transactions": "alpha_vantage, yfinance",
    },
}
