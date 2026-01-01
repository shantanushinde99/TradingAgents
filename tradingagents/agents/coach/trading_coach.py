"""
Trading Coach Agent - Professional Trading Mentor with Firm Personality
Provides real-time strategy validation, risk assessment, and educational guidance
Acts as strict but caring mentor to protect traders from costly mistakes
"""

from typing import Dict, List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
import json

from ..utils.risk_validator import RiskValidator
from ..utils.strategy_analyzer import StrategyAnalyzer


class TradingCoachAgent:
    """
    Professional Trading Coach with firm, protective personality.
    Validates strategies, enforces risk management, and educates traders.
    """
    
    SYSTEM_PROMPT = """You are a Professional Trading Coach with 20+ years of experience.

YOUR PERSONALITY:
- FIRM and PROTECTIVE: You will NOT approve dangerous trades
- DIRECT: Say exactly what needs to be said, no sugar-coating
- EDUCATIONAL: Always explain WHY something is right or wrong
- PROFESSIONAL: Use trading terminology correctly
- CARING: Your strictness comes from wanting to protect the trader's capital

YOUR ROLE:
1. Validate trading strategies against professional standards
2. Enforce risk management rules strictly
3. Catch dangerous patterns and warn immediately
4. Approve good strategies enthusiastically
5. Educate traders on proper execution

CRITICAL RULES YOU ENFORCE:
- MANDATORY stop loss on every trade (non-negotiable)
- Maximum 2-3% risk per trade
- Minimum 1:1.5 risk-reward ratio (recommend 1:2)
- Position sizing must be appropriate
- No emotional trading, revenge trading, or averaging down

YOUR COMMUNICATION STYLE:
✅ Good Strategy: "EXCELLENT! Your plan checks all the boxes. Here's why this works..."
⚠️ Risky Strategy: "I see what you're trying to do, but there's a problem..."
❌ Dangerous Strategy: "STOP! I cannot approve this trade. Here's why..."

When responding:
1. Start with clear verdict: ✅ APPROVED, ⚠️ CAUTION, or ❌ REJECTED
2. Explain reasoning with specific numbers
3. Reference the market analysis when available
4. Provide actionable fixes for problems
5. End with execution guidance

CONTEXT AVAILABLE:
- Stock analysis from 4 expert agents (market, technical, sentiment, fundamentals)
- Current market data and technical indicators
- User's proposed trade plan
- Risk validation results
- Strategy quality assessment

Remember: Your job is to PROTECT traders from destroying their accounts while teaching them to trade professionally."""

    def __init__(
        self,
        llm_provider: str = "google",
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None
    ):
        """
        Initialize Trading Coach Agent.
        
        Args:
            llm_provider: "google", "groq", or "openrouter"
            model_name: Model to use
            api_key: API key (or will use environment variable)
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        
        # Initialize validators
        self.risk_validator = RiskValidator()
        self.strategy_analyzer = StrategyAnalyzer()
        
        # Initialize LLM
        self.llm = self._initialize_llm(llm_provider, model_name, api_key)
        
        # Conversation history
        self.conversation_history = []
        
        # Current trading context
        self.current_context = {}
    
    def _initialize_llm(self, provider: str, model: str, api_key: Optional[str]):
        """Initialize the language model."""
        if provider == "google":
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=2048
            )
        elif provider == "groq":
            api_key = api_key or os.getenv("GROQ_API_KEY")
            return ChatGroq(
                model=model,
                groq_api_key=api_key,
                temperature=0.7,
                max_tokens=2048
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def set_context(
        self,
        ticker: str,
        current_price: float,
        market_analysis: Optional[Dict] = None,
        technical_analysis: Optional[Dict] = None,
        sentiment_analysis: Optional[Dict] = None,
        fundamental_analysis: Optional[Dict] = None,
        generated_signal: Optional[Dict] = None
    ):
        """
        Set trading context from agent analysis.
        
        Args:
            ticker: Stock symbol
            current_price: Current market price
            market_analysis: Analysis from market analyst
            technical_analysis: Technical indicators and patterns
            sentiment_analysis: Sentiment scores and signals
            fundamental_analysis: Fundamental metrics
            generated_signal: Auto-generated trading signal
        """
        self.current_context = {
            "ticker": ticker,
            "current_price": current_price,
            "market_analysis": market_analysis or {},
            "technical_analysis": technical_analysis or {},
            "sentiment_analysis": sentiment_analysis or {},
            "fundamental_analysis": fundamental_analysis or {},
            "generated_signal": generated_signal or {},
            "timestamp": str(json.dumps({}))  # For tracking
        }
    
    def validate_trade_plan(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        position_size: int,
        account_size: float,
        trade_direction: str,
        user_reasoning: str = ""
    ) -> Dict:
        """
        Comprehensive trade plan validation.
        
        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss level
            take_profit: Take profit target
            position_size: Number of shares
            account_size: Total account value
            trade_direction: "LONG" or "SHORT"
            user_reasoning: User's explanation
            
        Returns:
            Complete validation with coach feedback
        """
        # Step 1: Risk Validation
        risk_validation = self.risk_validator.validate_trade_plan(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            account_size=account_size,
            current_price=self.current_context.get("current_price", entry_price),
            trade_direction=trade_direction
        )
        
        # Step 2: Strategy Analysis
        strategy_analysis = self.strategy_analyzer.analyze_trade_plan(
            ticker=self.current_context.get("ticker", "UNKNOWN"),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            trade_direction=trade_direction,
            user_reasoning=user_reasoning,
            market_analysis=self.current_context.get("market_analysis"),
            technical_indicators=self.current_context.get("technical_analysis"),
            sentiment_data=self.current_context.get("sentiment_analysis")
        )
        
        # Step 3: Detect dangerous patterns
        dangerous_patterns = self.risk_validator.detect_dangerous_patterns(user_reasoning)
        
        # Compile validation result
        validation_result = {
            "can_execute": risk_validation["can_execute"] and strategy_analysis["confidence_score"] >= 50,
            "overall_verdict": self._determine_verdict(risk_validation, strategy_analysis),
            "risk_validation": risk_validation,
            "strategy_analysis": strategy_analysis,
            "dangerous_patterns": dangerous_patterns,
            "trade_plan": {
                "ticker": self.current_context.get("ticker"),
                "direction": trade_direction,
                "entry": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "account_size": account_size
            }
        }
        
        return validation_result
    
    def _determine_verdict(self, risk_val: Dict, strategy_analysis: Dict) -> str:
        """Determine overall verdict on trade."""
        if risk_val["severity"] == "CRITICAL":
            return "❌ REJECTED"
        elif strategy_analysis["confidence_score"] < 40:
            return "❌ REJECTED"
        elif risk_val["severity"] == "WARNING" or strategy_analysis["confidence_score"] < 60:
            return "⚠️ CAUTION"
        else:
            return "✅ APPROVED"
    
    def chat(self, user_message: str, include_validation: bool = False) -> Dict:
        """
        Interactive chat with trading coach.
        
        Args:
            user_message: User's question or request
            include_validation: Whether to run full validation
            
        Returns:
            Coach response with analysis
        """
        # Build context summary
        context_summary = self._build_context_summary()
        
        # Build conversation messages
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            SystemMessage(content=f"CURRENT TRADING CONTEXT:\n{context_summary}")
        ]
        
        # Add conversation history
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current message
        messages.append(HumanMessage(content=user_message))
        
        # Get LLM response
        try:
            response = self.llm.invoke(messages)
            coach_response = response.content
        except Exception as e:
            coach_response = f"I apologize, but I encountered an error: {str(e)}"
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "coach",
            "content": coach_response
        })
        
        return {
            "response": coach_response,
            "conversation_turn": len(self.conversation_history) // 2,
            "context_available": bool(self.current_context)
        }
    
    def _build_context_summary(self) -> str:
        """Build summary of current trading context."""
        if not self.current_context:
            return "No trading context loaded. Ask user to run analysis first."
        
        ticker = self.current_context.get("ticker", "UNKNOWN")
        price = self.current_context.get("current_price", 0)
        
        summary_parts = [f"Stock: {ticker} | Current Price: ${price:.2f}"]
        
        # Add market analysis summary
        market = self.current_context.get("market_analysis", {})
        if market:
            signal = market.get("signal", "HOLD")
            confidence = market.get("confidence", 0)
            summary_parts.append(f"Market Signal: {signal} ({confidence}% confidence)")
        
        # Add technical summary
        technical = self.current_context.get("technical_analysis", {})
        if technical:
            trend = technical.get("trend", "NEUTRAL")
            support = technical.get("support", 0)
            resistance = technical.get("resistance", 0)
            summary_parts.append(f"Trend: {trend} | Support: ${support:.2f} | Resistance: ${resistance:.2f}")
        
        # Add generated signal if available
        signal = self.current_context.get("generated_signal", {})
        if signal:
            action = signal.get("action", "HOLD")
            entry = signal.get("entry_price", 0)
            stop = signal.get("stop_loss", 0)
            target = signal.get("take_profit", 0)
            summary_parts.append(f"Agent Recommendation: {action} | Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}")
        
        return "\n".join(summary_parts)
    
    def suggest_optimal_entry(self) -> Dict:
        """Suggest optimal entry point with stop loss and take profit based on analysis."""
        technical = self.current_context.get("technical_analysis", {})
        market = self.current_context.get("market_analysis", {})
        current_price = self.current_context.get("current_price", 0)
        
        if not technical or not market:
            return {
                "error": "Need technical and market analysis first"
            }
        
        signal = market.get("signal", "HOLD")
        support = technical.get("support", current_price * 0.97)
        resistance = technical.get("resistance", current_price * 1.03)
        
        suggestion = {}
        
        if signal == "BUY":
            # For LONG positions:
            # Entry: near support (buying low)
            # Stop Loss: BELOW entry (protect if price drops)
            # Take Profit: ABOVE entry (profit when price rises)
            optimal_entry = support * 1.005  # Slightly above support
            aggressive_entry = current_price
            conservative_entry = support * 0.998  # Just below support for better fill
            
            # Stop loss BELOW entry for LONG
            stop_loss = support * 0.98  # Below support level
            # Take profit ABOVE entry for LONG
            take_profit = resistance * 0.98  # Near resistance level
            
            # Calculate R:R
            risk = optimal_entry - stop_loss
            reward = take_profit - optimal_entry
            rr_ratio = reward / risk if risk > 0 else 0
            
            suggestion = {
                "direction": "LONG",
                "optimal_entry": optimal_entry,
                "aggressive_entry": aggressive_entry,
                "conservative_entry": conservative_entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": rr_ratio,
                "reasoning": f"LONG setup: Entry near support (${support:.2f}). Stop loss at ${stop_loss:.2f} (BELOW entry). Target at ${take_profit:.2f} (ABOVE entry). R:R = 1:{rr_ratio:.2f}"
            }
        
        elif signal == "SELL":
            # For SHORT positions:
            # Entry: near resistance (selling high)
            # Stop Loss: ABOVE entry (protect if price rises)
            # Take Profit: BELOW entry (profit when price drops)
            optimal_entry = resistance * 0.995  # Slightly below resistance
            aggressive_entry = current_price
            conservative_entry = resistance * 1.002  # Just above resistance for better fill
            
            # Stop loss ABOVE entry for SHORT
            stop_loss = resistance * 1.02  # Above resistance level
            # Take profit BELOW entry for SHORT
            take_profit = support * 1.02  # Near support level
            
            # Calculate R:R
            risk = stop_loss - optimal_entry
            reward = optimal_entry - take_profit
            rr_ratio = reward / risk if risk > 0 else 0
            
            suggestion = {
                "direction": "SHORT",
                "optimal_entry": optimal_entry,
                "aggressive_entry": aggressive_entry,
                "conservative_entry": conservative_entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": rr_ratio,
                "reasoning": f"SHORT setup: Entry near resistance (${resistance:.2f}). Stop loss at ${stop_loss:.2f} (ABOVE entry). Target at ${take_profit:.2f} (BELOW entry). R:R = 1:{rr_ratio:.2f}"
            }
        
        else:
            suggestion = {
                "direction": "HOLD",
                "reasoning": "No clear setup. Wait for better opportunity."
            }
        
        return suggestion
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_size: float,
        risk_percent: float = 2.0
    ) -> Dict:
        """
        Calculate appropriate position size.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss level
            account_size: Total account value
            risk_percent: Percentage to risk (default 2%)
            
        Returns:
            Position sizing recommendation
        """
        if stop_loss is None or entry_price == stop_loss:
            return {
                "error": "Invalid stop loss. Cannot calculate position size."
            }
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Calculate dollar risk
        dollar_risk = account_size * (risk_percent / 100)
        
        # Calculate position size
        position_size = int(dollar_risk / risk_per_share)
        
        # Calculate position value
        position_value = position_size * entry_price
        position_percent = (position_value / account_size) * 100
        
        # Recommendations
        conservative_size = int(position_size * 0.75)
        aggressive_size = int(position_size * 1.25) if risk_percent < 2.5 else position_size
        
        return {
            "recommended_size": position_size,
            "conservative_size": conservative_size,
            "aggressive_size": aggressive_size,
            "risk_per_share": risk_per_share,
            "dollar_risk": dollar_risk,
            "risk_percent": risk_percent,
            "position_value": position_value,
            "position_percent": position_percent,
            "explanation": f"Risking {risk_percent}% (${dollar_risk:.2f}) with ${risk_per_share:.2f} risk per share = {position_size} shares"
        }
    
    def check_risk_reward(
        self,
        entry: float,
        stop: float,
        target: float,
        trade_direction: str = "LONG"
    ) -> Dict:
        """Check risk-reward ratio based on trade direction."""
        is_valid, message, rr_ratio = self.risk_validator.validate_risk_reward(
            entry, stop, target, trade_direction
        )
        
        # Calculate risk and reward based on direction
        if trade_direction == "LONG":
            risk = entry - stop if stop < entry else abs(entry - stop)
            reward = target - entry if target > entry else abs(target - entry)
        elif trade_direction == "SHORT":
            risk = stop - entry if stop > entry else abs(entry - stop)
            reward = entry - target if target < entry else abs(target - entry)
        else:
            risk = abs(entry - stop)
            reward = abs(target - entry)
        
        return {
            "is_valid": is_valid,
            "rr_ratio": rr_ratio,
            "message": message,
            "risk_amount": risk,
            "reward_amount": reward,
            "trade_direction": trade_direction,
            "recommendation": "Excellent" if rr_ratio >= 2 else "Acceptable" if rr_ratio >= 1.5 else "Poor"
        }
    
    def generate_coach_feedback(self, validation_result: Dict) -> str:
        """
        Generate detailed coach feedback from validation.
        
        Args:
            validation_result: Result from validate_trade_plan
            
        Returns:
            Formatted coach feedback message
        """
        verdict = validation_result["overall_verdict"]
        risk_val = validation_result["risk_validation"]
        strategy = validation_result["strategy_analysis"]
        
        # Start with verdict
        feedback = [f"\n{'='*60}"]
        feedback.append(f"TRADING COACH VERDICT: {verdict}")
        feedback.append(f"{'='*60}\n")
        
        # Risk validation summary
        feedback.append(f"RISK ASSESSMENT: {risk_val['severity']}")
        feedback.append(risk_val['summary'])
        feedback.append("")
        
        # Critical issues
        if risk_val['issues']:
            feedback.append("🚨 CRITICAL ISSUES:")
            for issue in risk_val['issues']:
                feedback.append(f"  • {issue['issue']}: {issue['message']}")
                feedback.append(f"    Fix: {issue['fix']}")
            feedback.append("")
        
        # Strategy assessment
        feedback.append(f"STRATEGY QUALITY: {strategy['assessment']}")
        feedback.append(f"Confidence Score: {strategy['confidence_score']}/100")
        feedback.append("")
        
        # Strengths
        if strategy['strengths']:
            feedback.append("✅ STRENGTHS:")
            for strength in strategy['strengths'][:5]:
                feedback.append(f"  • {strength}")
            feedback.append("")
        
        # Weaknesses
        if strategy['weaknesses']:
            feedback.append("⚠️ WEAKNESSES:")
            for weakness in strategy['weaknesses'][:5]:
                feedback.append(f"  • {weakness}")
            feedback.append("")
        
        # Recommendations
        if strategy['recommendations']:
            feedback.append("💡 RECOMMENDATIONS:")
            for rec in strategy['recommendations'][:5]:
                feedback.append(f"  • {rec}")
            feedback.append("")
        
        # Execution advice
        feedback.append("📋 EXECUTION PLAN:")
        for advice in strategy['execution_advice']:
            feedback.append(f"  • {advice}")
        
        feedback.append(f"\n{'='*60}\n")
        
        return "\n".join(feedback)
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history.copy()
