"""
Strategy Analyzer - Evaluates Trade Plans and Provides Recommendations
Assesses trade quality, alignment with market analysis, and confidence scoring
"""

from typing import Dict, List, Optional, Tuple
import json


class StrategyAnalyzer:
    """Analyzes trading strategies and provides professional feedback."""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_trade_plan(
        self,
        ticker: str,
        entry_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        position_size: Optional[int],
        trade_direction: str,  # "LONG", "SHORT", "HOLD"
        user_reasoning: str,
        market_analysis: Optional[Dict] = None,
        technical_indicators: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None
    ) -> Dict:
        """
        Comprehensive trade strategy analysis.
        
        Args:
            ticker: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss level
            take_profit: Take profit target
            position_size: Number of shares
            trade_direction: LONG, SHORT, or HOLD
            user_reasoning: User's explanation of their strategy
            market_analysis: Analysis from market analyst agent
            technical_indicators: Technical analysis data
            sentiment_data: Sentiment analysis results
            
        Returns:
            Analysis with confidence score, strengths, weaknesses, recommendations
        """
        strengths = []
        weaknesses = []
        recommendations = []
        confidence_score = 0  # 0-100
        
        # Analyze completeness
        completeness_score = self._assess_completeness(
            entry_price, stop_loss, take_profit, position_size, user_reasoning
        )
        confidence_score += completeness_score["score"]
        strengths.extend(completeness_score["strengths"])
        weaknesses.extend(completeness_score["weaknesses"])
        
        # Analyze alignment with market data
        if market_analysis:
            alignment_score = self._assess_market_alignment(
                trade_direction, entry_price, market_analysis
            )
            confidence_score += alignment_score["score"]
            strengths.extend(alignment_score["strengths"])
            weaknesses.extend(alignment_score["weaknesses"])
            recommendations.extend(alignment_score["recommendations"])
        
        # Analyze technical setup
        if technical_indicators:
            technical_score = self._assess_technical_setup(
                trade_direction, entry_price, stop_loss, take_profit, technical_indicators
            )
            confidence_score += technical_score["score"]
            strengths.extend(technical_score["strengths"])
            weaknesses.extend(technical_score["weaknesses"])
            recommendations.extend(technical_score["recommendations"])
        
        # Analyze sentiment alignment
        if sentiment_data:
            sentiment_score = self._assess_sentiment_alignment(
                trade_direction, sentiment_data
            )
            confidence_score += sentiment_score["score"]
            strengths.extend(sentiment_score["strengths"])
            weaknesses.extend(sentiment_score["weaknesses"])
        
        # Analyze user reasoning quality
        reasoning_score = self._assess_reasoning_quality(user_reasoning)
        confidence_score += reasoning_score["score"]
        strengths.extend(reasoning_score["strengths"])
        weaknesses.extend(reasoning_score["weaknesses"])
        recommendations.extend(reasoning_score["recommendations"])
        
        # Normalize confidence score to 0-100
        max_possible_score = 100
        confidence_score = min(100, int((confidence_score / max_possible_score) * 100))
        
        # Generate overall assessment
        assessment = self._generate_assessment(confidence_score, strengths, weaknesses)
        
        result = {
            "ticker": ticker,
            "trade_direction": trade_direction,
            "confidence_score": confidence_score,
            "assessment": assessment,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "risk_level": self._determine_risk_level(confidence_score, weaknesses),
            "execution_advice": self._generate_execution_advice(
                confidence_score, trade_direction, entry_price, stop_loss, take_profit
            )
        }
        
        self.analysis_history.append(result)
        return result
    
    def _assess_completeness(
        self,
        entry: Optional[float],
        stop: Optional[float],
        target: Optional[float],
        size: Optional[int],
        reasoning: str
    ) -> Dict:
        """Check if trade plan is complete."""
        score = 0
        strengths = []
        weaknesses = []
        
        if entry and entry > 0:
            score += 5
            strengths.append("✅ Entry price defined")
        else:
            weaknesses.append("❌ Missing entry price")
        
        if stop and stop > 0:
            score += 10
            strengths.append("✅ Stop loss defined")
        else:
            weaknesses.append("❌ No stop loss (CRITICAL)")
        
        if target and target > 0:
            score += 5
            strengths.append("✅ Profit target defined")
        else:
            weaknesses.append("⚠️ No profit target")
        
        if size and size > 0:
            score += 5
            strengths.append("✅ Position size specified")
        else:
            weaknesses.append("⚠️ Position size not specified")
        
        if reasoning and len(reasoning) > 20:
            score += 5
            strengths.append("✅ Trade reasoning provided")
        else:
            weaknesses.append("⚠️ Insufficient reasoning")
        
        return {"score": score, "strengths": strengths, "weaknesses": weaknesses}
    
    def _assess_market_alignment(
        self,
        direction: str,
        entry: Optional[float],
        market_analysis: Dict
    ) -> Dict:
        """Check if trade aligns with market analysis."""
        score = 0
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Extract agent signal if available
        agent_signal = market_analysis.get("signal", "HOLD")
        agent_confidence = market_analysis.get("confidence", 50)
        
        # Check directional alignment
        if direction == "LONG" and agent_signal == "BUY":
            score += 15
            strengths.append(f"✅ Direction aligns with market analysis (BUY signal)")
        elif direction == "SHORT" and agent_signal == "SELL":
            score += 15
            strengths.append(f"✅ Direction aligns with market analysis (SELL signal)")
        elif direction != "HOLD" and agent_signal == "HOLD":
            score += 5
            weaknesses.append(f"⚠️ Market analysis suggests HOLD, but you want to trade")
            recommendations.append("Consider waiting for better setup")
        elif (direction == "LONG" and agent_signal == "SELL") or (direction == "SHORT" and agent_signal == "BUY"):
            weaknesses.append(f"❌ Direction CONFLICTS with market analysis")
            recommendations.append("Trading against the analysis is high-risk. Reconsider.")
        
        # Check confidence alignment
        if agent_confidence >= 70:
            score += 10
            strengths.append(f"✅ High analyst confidence ({agent_confidence}%)")
        elif agent_confidence >= 50:
            score += 5
            weaknesses.append(f"⚠️ Moderate analyst confidence ({agent_confidence}%)")
        else:
            weaknesses.append(f"⚠️ Low analyst confidence ({agent_confidence}%)")
            recommendations.append("Consider reducing position size due to low conviction")
        
        return {
            "score": score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def _assess_technical_setup(
        self,
        direction: str,
        entry: Optional[float],
        stop: Optional[float],
        target: Optional[float],
        technical_indicators: Dict
    ) -> Dict:
        """Evaluate technical setup quality."""
        score = 0
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Check trend alignment
        trend = technical_indicators.get("trend", "NEUTRAL")
        if (direction == "LONG" and trend == "BULLISH") or (direction == "SHORT" and trend == "BEARISH"):
            score += 10
            strengths.append(f"✅ Trading with the trend ({trend})")
        elif trend == "NEUTRAL":
            score += 5
            weaknesses.append("⚠️ No clear trend (choppy market)")
            recommendations.append("Consider tighter stops in range-bound conditions")
        else:
            weaknesses.append(f"⚠️ Trading against the trend ({trend})")
            recommendations.append("Counter-trend trades require strong confirmation")
        
        # Check support/resistance levels
        support = technical_indicators.get("support")
        resistance = technical_indicators.get("resistance")
        
        if direction == "LONG" and support and entry:
            if entry <= support * 1.02:  # Entry near support
                score += 10
                strengths.append(f"✅ Entry near support (${support:.2f})")
            if stop and stop < support:
                score += 5
                strengths.append(f"✅ Stop below support")
        
        if direction == "SHORT" and resistance and entry:
            if entry >= resistance * 0.98:  # Entry near resistance
                score += 10
                strengths.append(f"✅ Entry near resistance (${resistance:.2f})")
            if stop and stop > resistance:
                score += 5
                strengths.append(f"✅ Stop above resistance")
        
        # Check momentum indicators
        rsi = technical_indicators.get("rsi")
        if rsi:
            if direction == "LONG" and 30 <= rsi <= 50:
                score += 5
                strengths.append(f"✅ RSI favorable for longs ({rsi:.1f})")
            elif direction == "SHORT" and 50 <= rsi <= 70:
                score += 5
                strengths.append(f"✅ RSI favorable for shorts ({rsi:.1f})")
            elif rsi > 80:
                weaknesses.append(f"⚠️ Extremely overbought (RSI: {rsi:.1f})")
                if direction == "LONG":
                    recommendations.append("Consider waiting for pullback")
            elif rsi < 20:
                weaknesses.append(f"⚠️ Extremely oversold (RSI: {rsi:.1f})")
                if direction == "SHORT":
                    recommendations.append("Risk of bounce - use tight stops")
        
        return {
            "score": score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def _assess_sentiment_alignment(self, direction: str, sentiment_data: Dict) -> Dict:
        """Check sentiment alignment."""
        score = 0
        strengths = []
        weaknesses = []
        
        sentiment = sentiment_data.get("overall_sentiment", "NEUTRAL")
        sentiment_score = sentiment_data.get("score", 0)
        
        if (direction == "LONG" and sentiment == "BULLISH") or (direction == "SHORT" and sentiment == "BEARISH"):
            score += 10
            strengths.append(f"✅ Sentiment supports your direction ({sentiment})")
        elif sentiment == "NEUTRAL":
            score += 5
        else:
            weaknesses.append(f"⚠️ Sentiment conflicts with direction ({sentiment})")
        
        return {"score": score, "strengths": strengths, "weaknesses": weaknesses}
    
    def _assess_reasoning_quality(self, reasoning: str) -> Dict:
        """Evaluate quality of user's reasoning."""
        score = 0
        strengths = []
        weaknesses = []
        recommendations = []
        
        reasoning_lower = reasoning.lower()
        
        # Check for analytical keywords
        analytical_terms = [
            "support", "resistance", "trend", "breakout", "volume",
            "momentum", "consolidation", "pattern", "analysis"
        ]
        analysis_count = sum(1 for term in analytical_terms if term in reasoning_lower)
        
        if analysis_count >= 3:
            score += 10
            strengths.append("✅ Reasoning includes technical analysis")
        elif analysis_count >= 1:
            score += 5
            strengths.append("⚠️ Some technical analysis mentioned")
        else:
            weaknesses.append("❌ No technical reasoning provided")
            recommendations.append("Base decisions on technical/fundamental analysis, not emotions")
        
        # Check for emotional keywords (red flags)
        emotional_terms = [
            "hope", "feel", "believe", "gut", "hunch", "revenge",
            "must", "can't lose", "guaranteed"
        ]
        emotional_count = sum(1 for term in emotional_terms if term in reasoning_lower)
        
        if emotional_count > 0:
            weaknesses.append(f"⚠️ Emotional language detected ({emotional_count} instances)")
            recommendations.append("Remove emotions from trading decisions")
        else:
            score += 5
            strengths.append("✅ Objective reasoning")
        
        # Check for risk management mentions
        if any(term in reasoning_lower for term in ["stop", "risk", "position size"]):
            score += 5
            strengths.append("✅ Risk management considered")
        
        return {
            "score": score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def _generate_assessment(self, confidence: int, strengths: List, weaknesses: List) -> str:
        """Generate overall trade assessment."""
        if confidence >= 75:
            return f"🟢 STRONG SETUP ({confidence}%): This trade plan is well-constructed with {len(strengths)} positive factors."
        elif confidence >= 60:
            return f"🟡 ACCEPTABLE SETUP ({confidence}%): Trade has merit but could be improved. {len(weaknesses)} concern(s)."
        elif confidence >= 40:
            return f"🟠 WEAK SETUP ({confidence}%): Multiple issues detected. Consider revising plan."
        else:
            return f"🔴 POOR SETUP ({confidence}%): Significant problems found. Not recommended for execution."
    
    def _determine_risk_level(self, confidence: int, weaknesses: List) -> str:
        """Determine overall risk level."""
        critical_weaknesses = [w for w in weaknesses if "CRITICAL" in w or "❌" in w]
        
        if len(critical_weaknesses) > 0:
            return "HIGH"
        elif confidence < 50:
            return "HIGH"
        elif confidence < 70:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_execution_advice(
        self,
        confidence: int,
        direction: str,
        entry: Optional[float],
        stop: Optional[float],
        target: Optional[float]
    ) -> List[str]:
        """Generate specific execution recommendations."""
        advice = []
        
        if confidence >= 70:
            advice.append("✅ Plan is solid. Execute with discipline.")
            advice.append("Monitor position actively after entry")
        elif confidence >= 50:
            advice.append("⚠️ Consider starting with a smaller position")
            advice.append("Add to position if price confirms your thesis")
        else:
            advice.append("🛑 Do NOT execute this trade yet")
            advice.append("Address weaknesses first, then reassess")
        
        if stop:
            advice.append(f"Set stop loss IMMEDIATELY at ${stop:.2f}")
            advice.append("Never move stop loss against your position")
        
        if target:
            advice.append(f"Consider taking partial profits at ${target * 0.75:.2f} (75% to target)")
            advice.append(f"Full exit or trail stop at ${target:.2f}")
        
        return advice
    
    def compare_with_agent_signal(self, user_plan: Dict, agent_signal: Dict) -> Dict:
        """Compare user's plan with agent-generated signal."""
        differences = []
        agreements = []
        
        # Compare direction
        if user_plan.get("direction") == agent_signal.get("action"):
            agreements.append("Direction matches agent recommendation")
        else:
            differences.append({
                "aspect": "Direction",
                "user": user_plan.get("direction"),
                "agent": agent_signal.get("action"),
                "severity": "HIGH"
            })
        
        # Compare entry
        user_entry = user_plan.get("entry_price")
        agent_entry = agent_signal.get("entry_price")
        if user_entry and agent_entry:
            entry_diff_pct = abs(user_entry - agent_entry) / agent_entry * 100
            if entry_diff_pct <= 2:
                agreements.append(f"Entry within 2% of agent recommendation")
            else:
                differences.append({
                    "aspect": "Entry Price",
                    "user": f"${user_entry:.2f}",
                    "agent": f"${agent_entry:.2f}",
                    "difference": f"{entry_diff_pct:.1f}%",
                    "severity": "MEDIUM" if entry_diff_pct < 5 else "HIGH"
                })
        
        # Compare stop loss
        user_stop = user_plan.get("stop_loss")
        agent_stop = agent_signal.get("stop_loss")
        if user_stop and agent_stop:
            if abs(user_stop - agent_stop) / agent_stop * 100 <= 5:
                agreements.append("Stop loss close to agent recommendation")
            else:
                differences.append({
                    "aspect": "Stop Loss",
                    "user": f"${user_stop:.2f}",
                    "agent": f"${agent_stop:.2f}",
                    "severity": "MEDIUM"
                })
        
        return {
            "agreements": agreements,
            "differences": differences,
            "alignment_score": len(agreements) / (len(agreements) + len(differences)) * 100 if agreements or differences else 0
        }
