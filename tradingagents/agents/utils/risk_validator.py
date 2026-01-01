"""
Risk Validator - Enforces Trading Rules and Risk Management
Validates user strategies against professional trading standards
"""

from typing import Dict, List, Optional, Tuple


class RiskValidator:
    """Validates trading strategies against risk management rules."""
    
    # Trading Rules (Industry Standards)
    MAX_RISK_PERCENT = 3.0  # Maximum 3% risk per trade
    RECOMMENDED_RISK_PERCENT = 2.0  # Recommended 2% risk
    MIN_RISK_REWARD_RATIO = 1.5  # Minimum 1:1.5 R:R
    RECOMMENDED_RISK_REWARD_RATIO = 2.0  # Recommended 1:2 R:R
    MAX_POSITION_SIZE_PERCENT = 10.0  # Maximum 10% of account per position
    
    def __init__(self):
        self.validation_history = []
    
    def validate_trade_plan(
        self,
        entry_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        position_size: Optional[int],
        account_size: float,
        current_price: float,
        trade_direction: str = "LONG"
    ) -> Dict:
        """
        Comprehensive trade plan validation.
        
        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss price
            take_profit: Take profit target
            position_size: Number of shares/contracts
            account_size: Total account value
            current_price: Current market price
            trade_direction: "LONG" or "SHORT"
            
        Returns:
            Validation result with status, issues, and recommendations
        """
        issues = []
        warnings = []
        approvals = []
        severity = "SAFE"  # SAFE, WARNING, CRITICAL
        
        # Critical Check #1: Stop Loss
        if stop_loss is None:
            issues.append({
                "type": "CRITICAL",
                "issue": "NO STOP LOSS",
                "message": "Trading without a stop loss is UNACCEPTABLE. This exposes you to unlimited risk.",
                "consequence": "Account blow-up risk, catastrophic losses possible",
                "fix": "Set stop loss below support (for longs) or above resistance (for shorts)"
            })
            severity = "CRITICAL"
        
        # Critical Check #2: Entry Price
        if entry_price is None or entry_price <= 0:
            issues.append({
                "type": "CRITICAL",
                "issue": "INVALID ENTRY PRICE",
                "message": "You must specify a valid entry price",
                "consequence": "Cannot execute trade without entry level",
                "fix": "Define your exact entry price"
            })
            severity = "CRITICAL"
        
        # Calculate metrics if we have the data
        if entry_price and stop_loss and take_profit and position_size:
            # Risk per share
            risk_per_share = abs(entry_price - stop_loss)
            reward_per_share = abs(take_profit - entry_price)
            
            # Risk-Reward Ratio
            if risk_per_share > 0:
                rr_ratio = reward_per_share / risk_per_share
                
                if rr_ratio < self.MIN_RISK_REWARD_RATIO:
                    issues.append({
                        "type": "CRITICAL",
                        "issue": "POOR RISK-REWARD RATIO",
                        "message": f"Your R:R is 1:{rr_ratio:.2f}, which is below minimum 1:{self.MIN_RISK_REWARD_RATIO}",
                        "consequence": "Low probability of long-term profitability",
                        "fix": f"Adjust target to at least ${entry_price + (risk_per_share * self.RECOMMENDED_RISK_REWARD_RATIO):.2f} for 1:2 R:R"
                    })
                    severity = "CRITICAL"
                elif rr_ratio < self.RECOMMENDED_RISK_REWARD_RATIO:
                    warnings.append({
                        "type": "WARNING",
                        "issue": "SUBOPTIMAL RISK-REWARD",
                        "message": f"R:R is 1:{rr_ratio:.2f}. Recommended minimum is 1:2",
                        "suggestion": f"Consider adjusting target to ${entry_price + (risk_per_share * self.RECOMMENDED_RISK_REWARD_RATIO):.2f}"
                    })
                    if severity == "SAFE":
                        severity = "WARNING"
                else:
                    approvals.append({
                        "aspect": "Risk-Reward Ratio",
                        "status": "EXCELLENT",
                        "value": f"1:{rr_ratio:.2f}",
                        "comment": "Meets professional standards"
                    })
            
            # Total risk amount
            total_risk = position_size * risk_per_share
            risk_percent = (total_risk / account_size) * 100
            
            if risk_percent > self.MAX_RISK_PERCENT:
                issues.append({
                    "type": "CRITICAL",
                    "issue": "EXCESSIVE RISK",
                    "message": f"You're risking {risk_percent:.2f}% of your account (Max: {self.MAX_RISK_PERCENT}%)",
                    "consequence": "A few losing trades could devastate your account",
                    "fix": f"Reduce position to {int((account_size * self.RECOMMENDED_RISK_PERCENT / 100) / risk_per_share)} shares"
                })
                severity = "CRITICAL"
            elif risk_percent > self.RECOMMENDED_RISK_PERCENT:
                warnings.append({
                    "type": "WARNING",
                    "issue": "HIGH RISK",
                    "message": f"Risking {risk_percent:.2f}% (Recommended: {self.RECOMMENDED_RISK_PERCENT}%)",
                    "suggestion": f"Consider reducing to {int((account_size * self.RECOMMENDED_RISK_PERCENT / 100) / risk_per_share)} shares"
                })
                if severity == "SAFE":
                    severity = "WARNING"
            else:
                approvals.append({
                    "aspect": "Risk Percentage",
                    "status": "GOOD",
                    "value": f"{risk_percent:.2f}%",
                    "comment": "Within safe limits"
                })
            
            # Position size check
            position_value = position_size * entry_price
            position_percent = (position_value / account_size) * 100
            
            if position_percent > self.MAX_POSITION_SIZE_PERCENT:
                warnings.append({
                    "type": "WARNING",
                    "issue": "LARGE POSITION SIZE",
                    "message": f"Position is {position_percent:.1f}% of account (Max recommended: {self.MAX_POSITION_SIZE_PERCENT}%)",
                    "suggestion": "Consider diversifying to reduce concentration risk"
                })
                if severity == "SAFE":
                    severity = "WARNING"
        
        # Check stop loss placement logic based on trade direction
        if entry_price and stop_loss:
            if trade_direction == "LONG":
                # For LONG positions: Stop loss MUST be BELOW entry price
                if stop_loss >= entry_price:
                    issues.append({
                        "type": "CRITICAL",
                        "issue": "INVALID STOP LOSS PLACEMENT",
                        "message": f"Stop loss (${stop_loss:.2f}) is at or above entry price (${entry_price:.2f}) for a LONG position",
                        "consequence": "You'll be stopped out immediately for a loss",
                        "fix": "Place stop loss BELOW entry price for LONG trades"
                    })
                    severity = "CRITICAL"
                else:
                    approvals.append({
                        "aspect": "Stop Loss Placement",
                        "status": "CORRECT",
                        "value": f"SL ${stop_loss:.2f} < Entry ${entry_price:.2f}",
                        "comment": "Stop loss correctly placed below entry for LONG position"
                    })
            elif trade_direction == "SHORT":
                # For SHORT positions: Stop loss MUST be ABOVE entry price
                if stop_loss <= entry_price:
                    issues.append({
                        "type": "CRITICAL",
                        "issue": "INVALID STOP LOSS PLACEMENT",
                        "message": f"Stop loss (${stop_loss:.2f}) is at or below entry price (${entry_price:.2f}) for a SHORT position",
                        "consequence": "You'll be stopped out immediately for a loss",
                        "fix": "Place stop loss ABOVE entry price for SHORT trades"
                    })
                    severity = "CRITICAL"
                else:
                    approvals.append({
                        "aspect": "Stop Loss Placement",
                        "status": "CORRECT",
                        "value": f"SL ${stop_loss:.2f} > Entry ${entry_price:.2f}",
                        "comment": "Stop loss correctly placed above entry for SHORT position"
                    })
        
        # Check take profit placement logic based on trade direction
        if entry_price and take_profit:
            if trade_direction == "LONG":
                # For LONG positions: Take profit MUST be ABOVE entry price
                if take_profit <= entry_price:
                    issues.append({
                        "type": "CRITICAL",
                        "issue": "INVALID TAKE PROFIT PLACEMENT",
                        "message": f"Take profit (${take_profit:.2f}) is at or below entry price (${entry_price:.2f}) for a LONG position",
                        "consequence": "Target makes no sense - you can't profit by selling lower on a long",
                        "fix": "Place take profit ABOVE entry price for LONG trades"
                    })
                    severity = "CRITICAL"
                else:
                    approvals.append({
                        "aspect": "Take Profit Placement",
                        "status": "CORRECT",
                        "value": f"TP ${take_profit:.2f} > Entry ${entry_price:.2f}",
                        "comment": "Take profit correctly placed above entry for LONG position"
                    })
            elif trade_direction == "SHORT":
                # For SHORT positions: Take profit MUST be BELOW entry price
                if take_profit >= entry_price:
                    issues.append({
                        "type": "CRITICAL",
                        "issue": "INVALID TAKE PROFIT PLACEMENT",
                        "message": f"Take profit (${take_profit:.2f}) is at or above entry price (${entry_price:.2f}) for a SHORT position",
                        "consequence": "Target makes no sense - you can't profit by buying higher on a short",
                        "fix": "Place take profit BELOW entry price for SHORT trades"
                    })
                    severity = "CRITICAL"
                else:
                    approvals.append({
                        "aspect": "Take Profit Placement",
                        "status": "CORRECT",
                        "value": f"TP ${take_profit:.2f} < Entry ${entry_price:.2f}",
                        "comment": "Take profit correctly placed below entry for SHORT position"
                    })
        
        # Compile validation result
        result = {
            "is_valid": len(issues) == 0,
            "severity": severity,
            "can_execute": severity != "CRITICAL",
            "issues": issues,
            "warnings": warnings,
            "approvals": approvals,
            "summary": self._generate_summary(severity, issues, warnings, approvals)
        }
        
        self.validation_history.append(result)
        return result
    
    def _generate_summary(self, severity: str, issues: List, warnings: List, approvals: List) -> str:
        """Generate human-readable summary."""
        if severity == "CRITICAL":
            return f"🚨 TRADE REJECTED: {len(issues)} critical issue(s) found. Cannot proceed."
        elif severity == "WARNING":
            return f"⚠️ PROCEED WITH CAUTION: {len(warnings)} warning(s). Trade is acceptable but not optimal."
        else:
            return f"✅ TRADE APPROVED: All checks passed. {len(approvals)} aspect(s) validated."
    
    def check_stop_loss_mandatory(self, stop_loss: Optional[float]) -> Tuple[bool, str]:
        """Strict check: Stop loss is MANDATORY."""
        if stop_loss is None:
            return False, "🚨 STOP! You MUST have a stop loss. This is non-negotiable."
        return True, "✅ Stop loss present"
    
    def check_position_sizing(
        self,
        position_size: int,
        entry_price: float,
        stop_loss: float,
        account_size: float
    ) -> Tuple[bool, str, Dict]:
        """Check if position sizing is appropriate."""
        risk_per_share = abs(entry_price - stop_loss)
        total_risk = position_size * risk_per_share
        risk_percent = (total_risk / account_size) * 100
        
        recommended_size = int((account_size * self.RECOMMENDED_RISK_PERCENT / 100) / risk_per_share)
        max_size = int((account_size * self.MAX_RISK_PERCENT / 100) / risk_per_share)
        
        details = {
            "current_size": position_size,
            "recommended_size": recommended_size,
            "max_size": max_size,
            "risk_percent": risk_percent,
            "total_risk": total_risk
        }
        
        if risk_percent > self.MAX_RISK_PERCENT:
            return False, f"❌ Position too large! Risking {risk_percent:.2f}% (Max: {self.MAX_RISK_PERCENT}%)", details
        elif risk_percent > self.RECOMMENDED_RISK_PERCENT:
            return True, f"⚠️ Position size is high ({risk_percent:.2f}%). Consider reducing.", details
        else:
            return True, f"✅ Position size is good ({risk_percent:.2f}%)", details
    
    def validate_risk_reward(
        self,
        entry: float,
        stop: float,
        target: float,
        trade_direction: str = "LONG"
    ) -> Tuple[bool, str, float]:
        """Validate risk-reward ratio based on trade direction."""
        
        # First validate placement based on direction
        if trade_direction == "LONG":
            # LONG: Stop must be below entry, target must be above entry
            if stop >= entry:
                return False, f"❌ Invalid for LONG: Stop loss (${stop:.2f}) must be BELOW entry (${entry:.2f})", 0
            if target <= entry:
                return False, f"❌ Invalid for LONG: Target (${target:.2f}) must be ABOVE entry (${entry:.2f})", 0
            risk = entry - stop  # Risk is how much price can drop
            reward = target - entry  # Reward is how much price can rise
        elif trade_direction == "SHORT":
            # SHORT: Stop must be above entry, target must be below entry
            if stop <= entry:
                return False, f"❌ Invalid for SHORT: Stop loss (${stop:.2f}) must be ABOVE entry (${entry:.2f})", 0
            if target >= entry:
                return False, f"❌ Invalid for SHORT: Target (${target:.2f}) must be BELOW entry (${entry:.2f})", 0
            risk = stop - entry  # Risk is how much price can rise
            reward = entry - target  # Reward is how much price can drop
        else:
            # Default to absolute values if direction not specified
            risk = abs(entry - stop)
            reward = abs(target - entry)
        
        if risk == 0:
            return False, "❌ Invalid: Risk is zero", 0
        
        rr_ratio = reward / risk
        
        if rr_ratio < self.MIN_RISK_REWARD_RATIO:
            return False, f"❌ R:R too low: 1:{rr_ratio:.2f} (Min: 1:{self.MIN_RISK_REWARD_RATIO})", rr_ratio
        elif rr_ratio < self.RECOMMENDED_RISK_REWARD_RATIO:
            return True, f"⚠️ R:R acceptable but suboptimal: 1:{rr_ratio:.2f} (Recommended: 1:2)", rr_ratio
        else:
            return True, f"✅ Excellent R:R: 1:{rr_ratio:.2f}", rr_ratio
    
    def detect_dangerous_patterns(self, strategy_description: str) -> List[Dict]:
        """Detect dangerous trading patterns from user description."""
        dangers = []
        description_lower = strategy_description.lower()
        
        # Check for dangerous keywords
        dangerous_patterns = {
            "no stop": {
                "severity": "CRITICAL",
                "message": "Trading without stop loss detected",
                "consequence": "Unlimited loss potential"
            },
            "all in": {
                "severity": "CRITICAL",
                "message": "All-in position detected",
                "consequence": "Zero risk diversification"
            },
            "averaging down": {
                "severity": "CRITICAL",
                "message": "Averaging down in losing trade",
                "consequence": "Compounding losses, doubling risk"
            },
            "double down": {
                "severity": "CRITICAL",
                "message": "Doubling position in losing trade",
                "consequence": "Revenge trading, risk explosion"
            },
            "revenge": {
                "severity": "CRITICAL",
                "message": "Revenge trading detected",
                "consequence": "Emotional decision-making"
            },
            "hope": {
                "severity": "WARNING",
                "message": "Hope-based trading detected",
                "consequence": "No defined exit strategy"
            },
            "hold forever": {
                "severity": "WARNING",
                "message": "No exit plan",
                "consequence": "Unable to cut losses"
            },
            "can't lose": {
                "severity": "WARNING",
                "message": "Overconfidence detected",
                "consequence": "Ignoring risk management"
            }
        }
        
        for pattern, details in dangerous_patterns.items():
            if pattern in description_lower:
                dangers.append(details)
        
        return dangers
