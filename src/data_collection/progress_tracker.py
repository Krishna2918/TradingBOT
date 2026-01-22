"""
Progress Tracker - AI-Readable Data Collection Logging

Tracks data collection progress in machine-readable format for AI analysis and self-optimization.
Generates future step recommendations based on completion gaps and data quality.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class CollectionProgress:
    """Data structure for tracking collection progress"""
    symbol: str
    timeframe: str
    fetched_rows: int
    data_size_mb: float
    completion_pct: float
    last_updated: str
    error_count: int
    quality_score: float
    future_steps: List[str]

class ProgressTracker:
    """AI-readable progress tracking for data collection"""
    
    def __init__(self, log_file: str = "logs/data_progress.json", db_file: str = "data/collection_progress.db"):
        self.log_file = Path(log_file)
        self.db_file = Path(db_file)
        
        # Ensure directories exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize JSON log if not exists
        if not self.log_file.exists():
            self._init_json_log()
    
    def _init_database(self):
        """Initialize SQLite database for progress tracking"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    fetched_rows INTEGER DEFAULT 0,
                    data_size_mb REAL DEFAULT 0.0,
                    completion_pct REAL DEFAULT 0.0,
                    last_updated TEXT NOT NULL,
                    error_count INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0.0,
                    future_steps TEXT,
                    UNIQUE(symbol, timeframe)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TEXT NOT NULL,
                    session_end TEXT,
                    total_symbols INTEGER DEFAULT 0,
                    completed_symbols INTEGER DEFAULT 0,
                    total_requests INTEGER DEFAULT 0,
                    failed_requests INTEGER DEFAULT 0,
                    total_data_gb REAL DEFAULT 0.0,
                    session_type TEXT DEFAULT 'historical'
                )
            """)
            
            conn.commit()
    
    def _init_json_log(self):
        """Initialize JSON log file"""
        initial_log = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "AI-readable data collection progress tracking"
            },
            "overall_progress": {
                "total_symbols": 100,
                "completed_symbols": 0,
                "completion_percentage": 0.0,
                "total_data_gb": 0.0,
                "last_updated": datetime.now().isoformat()
            },
            "symbol_progress": {},
            "future_steps": [
                "Initialize symbol verification",
                "Begin historical data collection",
                "Validate data quality"
            ],
            "recommendations": {
                "next_priority": "Verify all 100 TSX/TSXV symbols",
                "estimated_time": "30-60 minutes",
                "resource_requirements": "Stable internet connection"
            }
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(initial_log, f, indent=2)
        
        logger.info(f"📝 Initialized progress log: {self.log_file}")
    
    def start_session(self, session_type: str = "historical") -> int:
        """Start a new collection session"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("""
                INSERT INTO collection_sessions (session_start, session_type)
                VALUES (?, ?)
            """, (datetime.now().isoformat(), session_type))
            
            session_id = cursor.lastrowid
            conn.commit()
        
        logger.info(f"🚀 Started collection session {session_id} ({session_type})")
        return session_id
    
    def end_session(self, session_id: int):
        """End a collection session"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                UPDATE collection_sessions 
                SET session_end = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), session_id))
            conn.commit()
        
        logger.info(f"✅ Ended collection session {session_id}")
    
    def log_progress(self, symbol: str, timeframe: str, fetched_rows: int, 
                    data_size_mb: float = 0.0, error_count: int = 0, 
                    quality_score: float = 1.0):
        """Log progress for a symbol/timeframe combination"""
        
        # Calculate completion percentage (rough estimate)
        expected_rows = self._estimate_expected_rows(timeframe)
        completion_pct = min(100.0, (fetched_rows / expected_rows) * 100) if expected_rows > 0 else 0.0
        
        # Generate future steps
        future_steps = self._generate_future_steps(symbol, timeframe, completion_pct, error_count, quality_score)
        
        # Update database
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO collection_progress 
                (symbol, timeframe, fetched_rows, data_size_mb, completion_pct, 
                 last_updated, error_count, quality_score, future_steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timeframe, fetched_rows, data_size_mb, completion_pct,
                  datetime.now().isoformat(), error_count, quality_score, 
                  json.dumps(future_steps)))
            conn.commit()
        
        # Update JSON log
        self._update_json_log()
        
        logger.info(f"📊 Progress logged: {symbol} {timeframe} - {fetched_rows} rows ({completion_pct:.1f}%)")
    
    def _estimate_expected_rows(self, timeframe: str) -> int:
        """Estimate expected number of rows for a timeframe"""
        estimates = {
            "1min": 7 * 24 * 60,      # 7 days of 1-minute data
            "5min": 60 * 24 * 12,     # 60 days of 5-minute data  
            "15min": 60 * 24 * 4,     # 60 days of 15-minute data
            "30min": 60 * 24 * 2,     # 60 days of 30-minute data
            "1d": 20 * 252,           # 20 years of daily data (252 trading days/year)
            "1wk": 20 * 52,           # 20 years of weekly data
            "1mo": 20 * 12,           # 20 years of monthly data
            "3mo": 20 * 4,            # 20 years of quarterly data
            "1y": 20                  # 20 years of yearly data
        }
        return estimates.get(timeframe, 1000)
    
    def _generate_future_steps(self, symbol: str, timeframe: str, completion_pct: float, 
                              error_count: int, quality_score: float) -> List[str]:
        """Generate AI-readable future step recommendations"""
        steps = []
        
        if completion_pct < 50:
            steps.append(f"Priority: Complete {symbol} {timeframe} data collection ({completion_pct:.1f}% done)")
        elif completion_pct < 95:
            steps.append(f"Fill gaps in {symbol} {timeframe} data ({100-completion_pct:.1f}% remaining)")
        else:
            steps.append(f"Validate {symbol} {timeframe} data quality")
        
        if error_count > 0:
            steps.append(f"Investigate {error_count} errors in {symbol} {timeframe}")
        
        if quality_score < 0.8:
            steps.append(f"Improve data quality for {symbol} {timeframe} (score: {quality_score:.2f})")
        
        if completion_pct >= 95 and quality_score >= 0.8:
            steps.append(f"Ready: Use {symbol} {timeframe} for model training")
        
        return steps
    
    def _update_json_log(self):
        """Update the JSON log with current progress"""
        # Get current progress from database
        with sqlite3.connect(self.db_file) as conn:
            df = pd.read_sql_query("""
                SELECT symbol, timeframe, fetched_rows, data_size_mb, 
                       completion_pct, error_count, quality_score, future_steps
                FROM collection_progress
            """, conn)
        
        if df.empty:
            return
        
        # Calculate overall statistics
        total_symbols = df['symbol'].nunique()
        completed_symbols = df[df['completion_pct'] >= 95]['symbol'].nunique()
        overall_completion = df['completion_pct'].mean()
        total_data_gb = df['data_size_mb'].sum() / 1024
        
        # Build symbol progress dictionary
        symbol_progress = {}
        for _, row in df.iterrows():
            symbol = row['symbol']
            if symbol not in symbol_progress:
                symbol_progress[symbol] = {}
            
            symbol_progress[symbol][row['timeframe']] = {
                "fetched_rows": int(row['fetched_rows']),
                "data_size_mb": float(row['data_size_mb']),
                "completion_pct": float(row['completion_pct']),
                "error_count": int(row['error_count']),
                "quality_score": float(row['quality_score']),
                "future_steps": json.loads(row['future_steps']) if row['future_steps'] else []
            }
        
        # Generate overall future steps
        overall_future_steps = self._generate_overall_future_steps(df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df)
        
        # Update JSON log
        log_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0.0", 
                "description": "AI-readable data collection progress tracking"
            },
            "overall_progress": {
                "total_symbols": 100,  # Target
                "completed_symbols": int(completed_symbols),
                "completion_percentage": float(overall_completion),
                "total_data_gb": float(total_data_gb),
                "last_updated": datetime.now().isoformat()
            },
            "symbol_progress": symbol_progress,
            "future_steps": overall_future_steps,
            "recommendations": recommendations
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _generate_overall_future_steps(self, df: pd.DataFrame) -> List[str]:
        """Generate overall future steps based on progress"""
        steps = []
        
        if df.empty:
            return ["Initialize data collection", "Verify symbols", "Begin historical data fetch"]
        
        # Check completion status
        avg_completion = df['completion_pct'].mean()
        
        if avg_completion < 25:
            steps.append("Continue historical data collection (early stage)")
        elif avg_completion < 75:
            steps.append("Continue historical data collection (mid stage)")
        elif avg_completion < 95:
            steps.append("Complete remaining data collection and fill gaps")
        else:
            steps.append("Begin model training with collected data")
        
        # Check for errors
        total_errors = df['error_count'].sum()
        if total_errors > 0:
            steps.append(f"Investigate and resolve {int(total_errors)} collection errors")
        
        # Check data quality
        avg_quality = df['quality_score'].mean()
        if avg_quality < 0.8:
            steps.append(f"Improve data quality (current score: {avg_quality:.2f})")
        
        return steps
    
    def _generate_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate AI recommendations based on current progress"""
        if df.empty:
            return {
                "next_priority": "Start symbol verification and historical data collection",
                "estimated_time": "3-5 hours for full 20-year dataset",
                "resource_requirements": "Stable internet, ~5GB storage space"
            }
        
        avg_completion = df['completion_pct'].mean()
        
        if avg_completion < 50:
            next_priority = "Continue historical data collection"
            estimated_time = f"{int((100-avg_completion)/20)} hours remaining"
        elif avg_completion < 95:
            next_priority = "Fill data gaps and complete collection"
            estimated_time = "1-2 hours remaining"
        else:
            next_priority = "Begin ML model training"
            estimated_time = "2-4 hours for LSTM/GRU training"
        
        return {
            "next_priority": next_priority,
            "estimated_time": estimated_time,
            "completion_status": f"{avg_completion:.1f}% complete",
            "resource_requirements": "Stable internet connection"
        }
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"error": "No progress data available"}
    
    def get_symbol_progress(self, symbol: str) -> Dict[str, Any]:
        """Get progress for a specific symbol"""
        summary = self.get_progress_summary()
        return summary.get("symbol_progress", {}).get(symbol, {})
    
    def get_incomplete_symbols(self, threshold: float = 95.0) -> List[str]:
        """Get symbols that are not yet complete"""
        with sqlite3.connect(self.db_file) as conn:
            df = pd.read_sql_query("""
                SELECT DISTINCT symbol 
                FROM collection_progress 
                WHERE completion_pct < ?
            """, conn, params=(threshold,))
        
        return df['symbol'].tolist() if not df.empty else []
    
    def export_progress_report(self, filepath: str = "reports/data_collection_progress.json"):
        """Export detailed progress report"""
        summary = self.get_progress_summary()
        
        # Add database statistics
        with sqlite3.connect(self.db_file) as conn:
            stats_df = pd.read_sql_query("""
                SELECT 
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(*) as total_entries,
                    AVG(completion_pct) as avg_completion,
                    SUM(data_size_mb) as total_data_mb,
                    SUM(error_count) as total_errors
                FROM collection_progress
            """, conn)
            
            if not stats_df.empty:
                summary["database_stats"] = stats_df.iloc[0].to_dict()
        
        # Save report
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Progress report exported to {filepath}")

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tracker = ProgressTracker()
    
    # Test logging some progress
    tracker.log_progress("RY.TO", "1d", 5000, 2.5, 0, 0.95)
    tracker.log_progress("TD.TO", "1d", 4800, 2.3, 1, 0.90)
    tracker.log_progress("SHOP.TO", "5min", 1200, 0.8, 0, 0.85)
    
    # Show summary
    summary = tracker.get_progress_summary()
    print("📊 Progress Summary:")
    print(f"Overall completion: {summary['overall_progress']['completion_percentage']:.1f}%")
    print(f"Total data: {summary['overall_progress']['total_data_gb']:.2f} GB")
    print(f"Future steps: {len(summary['future_steps'])}")