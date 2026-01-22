"""
Confidence Calibration Module
============================

This module implements Bayesian confidence calibration using Beta(2,2) priors
to improve the accuracy of trading confidence estimates based on historical outcomes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
import json

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Represents calibration data for a model."""
    model_name: str
    window_id: str
    total_trades: int
    wins: int
    losses: int
    raw_confidence_sum: float
    calibrated_probability: float
    calibration_quality: float  # Brier score or similar
    last_updated: datetime
    window_start: datetime
    window_end: datetime


@dataclass
class TradeOutcome:
    """Represents a trade outcome for calibration."""
    trade_id: str
    model_name: str
    symbol: str
    raw_confidence: float
    calibrated_confidence: float
    outcome: str  # "WIN", "LOSS", "PENDING"
    trade_date: datetime
    exit_date: Optional[datetime] = None
    pnl: Optional[float] = None
    window_id: str = ""


class ConfidenceCalibrator:
    """Bayesian confidence calibration using Beta(2,2) priors."""
    
    def __init__(self, window_size_days: int = 30, min_trades_for_calibration: int = 10):
        """
        Initialize confidence calibrator.
        
        Args:
            window_size_days: Rolling window size in days for calibration
            min_trades_for_calibration: Minimum trades needed for reliable calibration
        """
        self.window_size_days = window_size_days
        self.min_trades_for_calibration = min_trades_for_calibration
        
        # Beta(2,2) prior parameters
        self.alpha_prior = 2.0
        self.beta_prior = 2.0
        
        # Calibration cache
        self.calibration_cache: Dict[str, CalibrationData] = {}
        
        # Trade outcomes storage
        self.trade_outcomes: List[TradeOutcome] = []
        
        # Load existing calibration data from database
        self._load_calibration_from_database()
        
        logger.info(f"Confidence Calibrator initialized with {window_size_days} day window")
    
    def add_trade_outcome(self, trade_id: str, model_name: str, symbol: str, 
                         raw_confidence: float, outcome: str, trade_date: datetime,
                         exit_date: Optional[datetime] = None, pnl: Optional[float] = None):
        """Add a trade outcome for calibration."""
        # Generate window ID based on trade date
        window_id = self._get_window_id(trade_date)
        
        # Create trade outcome
        trade_outcome = TradeOutcome(
            trade_id=trade_id,
            model_name=model_name,
            symbol=symbol,
            raw_confidence=raw_confidence,
            calibrated_confidence=raw_confidence,  # Will be updated after calibration
            outcome=outcome,
            trade_date=trade_date,
            exit_date=exit_date,
            pnl=pnl,
            window_id=window_id
        )
        
        self.trade_outcomes.append(trade_outcome)
        
        # Update calibration for this model and window
        self._update_calibration(model_name, window_id)
        
        logger.debug(f"Added trade outcome: {trade_id} - {model_name} - {outcome}")
    
    def calibrate_confidence(self, model_name: str, raw_confidence: float, 
                           trade_date: datetime) -> float:
        """
        Calibrate raw confidence using Bayesian Beta(2,2) approach.
        
        Args:
            model_name: Name of the model
            raw_confidence: Raw confidence from model (0-1)
            trade_date: Date of the trade
            
        Returns:
            Calibrated probability (0-1)
        """
        window_id = self._get_window_id(trade_date)
        
        # Get calibration data for this model and window
        calibration_key = f"{model_name}_{window_id}"
        
        if calibration_key not in self.calibration_cache:
            # No calibration data available, return raw confidence
            logger.warning(f"No calibration data for {model_name} in window {window_id}")
            return raw_confidence
        
        calibration_data = self.calibration_cache[calibration_key]
        
        # Check if we have enough data for reliable calibration
        if calibration_data.total_trades < self.min_trades_for_calibration:
            logger.debug(f"Insufficient trades for calibration: {calibration_data.total_trades}")
            return raw_confidence
        
        # Apply Bayesian calibration
        calibrated_prob = self._apply_bayesian_calibration(
            raw_confidence, 
            calibration_data.wins, 
            calibration_data.total_trades
        )
        
        # Ensure calibrated probability is in valid range
        calibrated_prob = max(0.01, min(0.99, calibrated_prob))
        
        logger.debug(f"Calibrated {model_name}: {raw_confidence:.3f} -> {calibrated_prob:.3f}")
        
        return calibrated_prob
    
    def _apply_bayesian_calibration(self, raw_confidence: float, wins: int, total_trades: int) -> float:
        """
        Apply Bayesian Beta(2,2) calibration.
        
        Formula: p_cal = (wins + α_prior) / (total_trades + α_prior + β_prior)
        """
        if total_trades == 0:
            return raw_confidence
        
        # Calculate observed win rate
        observed_win_rate = wins / total_trades
        
        # Apply Bayesian update with Beta(2,2) prior
        calibrated_prob = (wins + self.alpha_prior) / (total_trades + self.alpha_prior + self.beta_prior)
        
        # Blend with raw confidence based on sample size
        # More trades = more weight to observed rate, fewer trades = more weight to raw confidence
        sample_weight = min(1.0, total_trades / (self.min_trades_for_calibration * 2))
        
        blended_prob = (sample_weight * calibrated_prob + 
                       (1 - sample_weight) * raw_confidence)
        
        return blended_prob
    
    def _get_window_id(self, trade_date: datetime) -> str:
        """Generate window ID based on trade date."""
        # Create rolling windows based on trade date
        window_start = trade_date - timedelta(days=self.window_size_days)
        window_id = f"{window_start.strftime('%Y%m%d')}_{trade_date.strftime('%Y%m%d')}"
        return window_id
    
    def _load_calibration_from_database(self) -> None:
        """Load existing calibration data from database into cache."""
        try:
            from src.config.database import execute_query
            from src.config.mode_manager import get_current_mode
            
            mode = get_current_mode()
            
            # Get all calibration data from database
            result = execute_query(
                "SELECT * FROM confidence_calibration ORDER BY trade_date DESC LIMIT 1000", 
                (), mode
            )
            
            if not result:
                logger.debug("No existing calibration data found in database")
                return
            
            # Group by model and window_id
            grouped_data = {}
            for record in result:
                model_name = record['model']
                window_id = record['window_id']
                key = f"{model_name}_{window_id}"
                
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(record)
            
            # Create calibration data for each group
            loaded_count = 0
            for key, records in grouped_data.items():
                # Use minimum of 3 trades for loading (lower than runtime requirement for flexibility)
                if len(records) >= 3:
                    model_name = records[0]['model']
                    window_id = records[0]['window_id']
                    
                    wins = sum(1 for r in records if r['outcome'] == 'WIN')
                    losses = sum(1 for r in records if r['outcome'] == 'LOSS')
                    total_trades = len(records)
                    raw_confidence_sum = sum(r['raw_confidence'] for r in records)
                    
                    # Calculate calibrated probability using Bayesian approach
                    calibrated_prob = (wins + self.alpha_prior) / (total_trades + self.alpha_prior + self.beta_prior)
                    
                    # Calculate calibration quality (Brier score)
                    brier_score = sum((r['raw_confidence'] - (1 if r['outcome'] == 'WIN' else 0)) ** 2 for r in records) / total_trades
                    calibration_quality = 1.0 - brier_score  # Higher is better
                    
                    # Get date range
                    trade_dates = [datetime.fromisoformat(r['trade_date']) for r in records]
                    window_start = min(trade_dates)
                    window_end = max(trade_dates)
                    
                    # Create calibration data
                    calibration_data = CalibrationData(
                        model_name=model_name,
                        window_id=window_id,
                        total_trades=total_trades,
                        wins=wins,
                        losses=losses,
                        raw_confidence_sum=raw_confidence_sum,
                        calibrated_probability=calibrated_prob,
                        calibration_quality=calibration_quality,
                        last_updated=datetime.now(),
                        window_start=window_start,
                        window_end=window_end
                    )
                    
                    # Add to cache
                    self.calibration_cache[key] = calibration_data
                    loaded_count += 1
            
            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} calibration windows from database")
            
        except Exception as e:
            logger.warning(f"Failed to load calibration data from database: {e}")
            # Continue without loading - not a critical error
    
    def _update_calibration(self, model_name: str, window_id: str):
        """Update calibration data for a model and window."""
        # Get trades for this model and window
        model_trades = [
            trade for trade in self.trade_outcomes
            if trade.model_name == model_name and trade.window_id == window_id
        ]
        
        if not model_trades:
            return
        
        # Calculate calibration statistics
        total_trades = len(model_trades)
        wins = sum(1 for trade in model_trades if trade.outcome == "WIN")
        losses = sum(1 for trade in model_trades if trade.outcome == "LOSS")
        raw_confidence_sum = sum(trade.raw_confidence for trade in model_trades)
        
        # Calculate calibrated probability
        calibrated_prob = self._apply_bayesian_calibration(
            raw_confidence_sum / total_trades if total_trades > 0 else 0.5,
            wins,
            total_trades
        )
        
        # Calculate calibration quality (Brier score)
        calibration_quality = self._calculate_brier_score(model_trades, calibrated_prob)
        
        # Create calibration data
        calibration_data = CalibrationData(
            model_name=model_name,
            window_id=window_id,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            raw_confidence_sum=raw_confidence_sum,
            calibrated_probability=calibrated_prob,
            calibration_quality=calibration_quality,
            last_updated=datetime.now(),
            window_start=min(trade.trade_date for trade in model_trades),
            window_end=max(trade.trade_date for trade in model_trades)
        )
        
        # Cache calibration data
        calibration_key = f"{model_name}_{window_id}"
        self.calibration_cache[calibration_key] = calibration_data
        
        # Update calibrated confidence in trade outcomes
        for trade in model_trades:
            trade.calibrated_confidence = calibrated_prob
        
        logger.debug(f"Updated calibration for {model_name}: {wins}/{total_trades} wins, "
                    f"calibrated prob: {calibrated_prob:.3f}")
    
    def _calculate_brier_score(self, trades: List[TradeOutcome], calibrated_prob: float) -> float:
        """Calculate Brier score for calibration quality."""
        if not trades:
            return 1.0
        
        brier_scores = []
        for trade in trades:
            if trade.outcome in ["WIN", "LOSS"]:
                actual_outcome = 1.0 if trade.outcome == "WIN" else 0.0
                brier_score = (calibrated_prob - actual_outcome) ** 2
                brier_scores.append(brier_score)
        
        return np.mean(brier_scores) if brier_scores else 1.0
    
    def get_calibration_summary(self, model_name: str = None) -> Dict[str, Any]:
        """Get calibration summary for a model or all models."""
        if model_name:
            # Get summary for specific model
            model_calibrations = [
                cal for cal in self.calibration_cache.values()
                if cal.model_name == model_name
            ]
        else:
            # Get summary for all models
            model_calibrations = list(self.calibration_cache.values())
        
        if not model_calibrations:
            return {"error": "No calibration data available"}
        
        # Calculate aggregate statistics
        total_trades = sum(cal.total_trades for cal in model_calibrations)
        total_wins = sum(cal.wins for cal in model_calibrations)
        avg_calibrated_prob = np.mean([cal.calibrated_probability for cal in model_calibrations])
        avg_calibration_quality = np.mean([cal.calibration_quality for cal in model_calibrations])
        
        return {
            "model_name": model_name or "ALL",
            "total_trades": total_trades,
            "total_wins": total_wins,
            "win_rate": total_wins / total_trades if total_trades > 0 else 0.0,
            "avg_calibrated_probability": avg_calibrated_prob,
            "avg_calibration_quality": avg_calibration_quality,
            "calibration_windows": len(model_calibrations),
            "last_updated": max(cal.last_updated for cal in model_calibrations).isoformat()
        }
    
    def get_recent_calibration(self, model_name: str, days_back: int = 7) -> Optional[CalibrationData]:
        """Get most recent calibration data for a model."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        model_calibrations = [
            cal for cal in self.calibration_cache.values()
            if cal.model_name == model_name and cal.last_updated >= cutoff_date
        ]
        
        if not model_calibrations:
            return None
        
        # Return most recent calibration
        return max(model_calibrations, key=lambda cal: cal.last_updated)
    
    def clear_old_data(self, days_to_keep: int = 90):
        """Clear old calibration data to manage memory."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Remove old trade outcomes
        self.trade_outcomes = [
            trade for trade in self.trade_outcomes
            if trade.trade_date >= cutoff_date
        ]
        
        # Remove old calibration data
        old_keys = [
            key for key, cal in self.calibration_cache.items()
            if cal.last_updated < cutoff_date
        ]
        
        for key in old_keys:
            del self.calibration_cache[key]
        
        logger.info(f"Cleared calibration data older than {days_to_keep} days")
    
    def export_calibration_data(self) -> Dict[str, Any]:
        """Export calibration data for persistence."""
        return {
            "calibration_cache": {
                key: {
                    "model_name": cal.model_name,
                    "window_id": cal.window_id,
                    "total_trades": cal.total_trades,
                    "wins": cal.wins,
                    "losses": cal.losses,
                    "raw_confidence_sum": cal.raw_confidence_sum,
                    "calibrated_probability": cal.calibrated_probability,
                    "calibration_quality": cal.calibration_quality,
                    "last_updated": cal.last_updated.isoformat(),
                    "window_start": cal.window_start.isoformat(),
                    "window_end": cal.window_end.isoformat()
                }
                for key, cal in self.calibration_cache.items()
            },
            "trade_outcomes": [
                {
                    "trade_id": trade.trade_id,
                    "model_name": trade.model_name,
                    "symbol": trade.symbol,
                    "raw_confidence": trade.raw_confidence,
                    "calibrated_confidence": trade.calibrated_confidence,
                    "outcome": trade.outcome,
                    "trade_date": trade.trade_date.isoformat(),
                    "exit_date": trade.exit_date.isoformat() if trade.exit_date else None,
                    "pnl": trade.pnl,
                    "window_id": trade.window_id
                }
                for trade in self.trade_outcomes
            ]
        }
    
    def import_calibration_data(self, data: Dict[str, Any]):
        """Import calibration data from persistence."""
        # Import calibration cache
        self.calibration_cache = {}
        for key, cal_data in data.get("calibration_cache", {}).items():
            self.calibration_cache[key] = CalibrationData(
                model_name=cal_data["model_name"],
                window_id=cal_data["window_id"],
                total_trades=cal_data["total_trades"],
                wins=cal_data["wins"],
                losses=cal_data["losses"],
                raw_confidence_sum=cal_data["raw_confidence_sum"],
                calibrated_probability=cal_data["calibrated_probability"],
                calibration_quality=cal_data["calibration_quality"],
                last_updated=datetime.fromisoformat(cal_data["last_updated"]),
                window_start=datetime.fromisoformat(cal_data["window_start"]),
                window_end=datetime.fromisoformat(cal_data["window_end"])
            )
        
        # Import trade outcomes
        self.trade_outcomes = []
        for trade_data in data.get("trade_outcomes", []):
            self.trade_outcomes.append(TradeOutcome(
                trade_id=trade_data["trade_id"],
                model_name=trade_data["model_name"],
                symbol=trade_data["symbol"],
                raw_confidence=trade_data["raw_confidence"],
                calibrated_confidence=trade_data["calibrated_confidence"],
                outcome=trade_data["outcome"],
                trade_date=datetime.fromisoformat(trade_data["trade_date"]),
                exit_date=datetime.fromisoformat(trade_data["exit_date"]) if trade_data["exit_date"] else None,
                pnl=trade_data["pnl"],
                window_id=trade_data["window_id"]
            ))
        
        logger.info(f"Imported calibration data: {len(self.calibration_cache)} calibrations, "
                   f"{len(self.trade_outcomes)} trade outcomes")


# Global confidence calibrator instance
_confidence_calibrator: Optional[ConfidenceCalibrator] = None


def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get global confidence calibrator instance."""
    global _confidence_calibrator
    if _confidence_calibrator is None:
        _confidence_calibrator = ConfidenceCalibrator()
    return _confidence_calibrator


# Convenience functions
def calibrate_confidence(model_name: str, raw_confidence: float, trade_date: datetime) -> float:
    """Calibrate raw confidence using Bayesian approach."""
    calibrator = get_confidence_calibrator()
    return calibrator.calibrate_confidence(model_name, raw_confidence, trade_date)


def add_trade_outcome(trade_id: str, model_name: str, symbol: str, 
                     raw_confidence: float, outcome: str, trade_date: datetime,
                     exit_date: Optional[datetime] = None, pnl: Optional[float] = None):
    """Add a trade outcome for calibration."""
    calibrator = get_confidence_calibrator()
    calibrator.add_trade_outcome(trade_id, model_name, symbol, raw_confidence, 
                                outcome, trade_date, exit_date, pnl)


def get_calibration_summary(model_name: str = None) -> Dict[str, Any]:
    """Get calibration summary for a model or all models."""
    calibrator = get_confidence_calibrator()
    return calibrator.get_calibration_summary(model_name)
