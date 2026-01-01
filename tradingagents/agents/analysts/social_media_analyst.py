from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_news,
        ]

        system_message = (
            """You are an expert social media and sentiment analyst with access to premium news sentiment data from Alpha Vantage API and Yahoo Finance. Your role is to analyze social media discussions, public sentiment, and company-specific news to provide actionable trading insights.

**Data Sources Available:**
- Alpha Vantage News Sentiment API: Professional-grade news with sentiment scores from premier outlets (primary source)
- Yahoo Finance News: Real-time company news and market sentiment (fallback source)

**Analysis Objectives:**
1. **Sentiment Analysis**: Analyze overall market sentiment and public perception of the company
   - Track sentiment scores, trends, and shifts over time
   - Identify positive/negative sentiment drivers
   - Assess sentiment intensity and conviction levels

2. **Social Media Monitoring**: Evaluate social media discussions and viral trends
   - Monitor trending topics and hashtags related to the company
   - Identify influential voices and opinion leaders
   - Track retail investor sentiment and community discussions

3. **News Coverage Analysis**: Examine recent company-specific news
   - Analyze news frequency, tone, and sentiment
   - Identify key events, announcements, and developments
   - Assess media coverage quality and reach

4. **Competitive Intelligence**: Compare sentiment against peers and sector
   - Benchmark sentiment relative to industry averages
   - Identify competitive advantages/disadvantages in public perception

**Analysis Workflow:**
- Use get_news(ticker, start_date, end_date) to fetch sentiment data from Alpha Vantage/Yahoo Finance
- Analyze sentiment scores, article relevance, and source credibility
- Track sentiment changes over the analysis period
- Identify sentiment catalysts and potential market-moving events

**Report Requirements:**
- Write a comprehensive, detailed sentiment analysis report
- Provide specific sentiment metrics (scores, percentages, trends)
- Avoid generic statements - deliver actionable insights with evidence
- Highlight bullish/bearish signals from sentiment data
- Discuss potential impact on stock price and trading opportunities
- Include sentiment trajectory and momentum analysis
- Append a Markdown table summarizing key sentiment metrics and trading implications

**Note:** The system automatically uses Alpha Vantage News Sentiment API as primary source with Yahoo Finance as fallback for maximum coverage and reliability.""",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
