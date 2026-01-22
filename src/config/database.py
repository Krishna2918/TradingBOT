"""
Database Configuration - Dual Mode Database Management

This module provides database connection management for both LIVE and DEMO modes,
ensuring complete data isolation between modes.
"""

import sqlite3
import logging
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
# from src.config.mode_manager import get_mode_manager  # Removed to avoid circular import

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections for dual mode operation."""
    
    def __init__(self, mode: Optional[str] = None):
        """Initialize database manager."""
        # self.mode_manager = get_mode_manager()  # Removed to avoid circular import
        self._connections: Dict[str, sqlite3.Connection] = {}
        self.mode = mode  # Store mode parameter for compatibility
        
        # Initialize databases for both modes
        self._initialize_databases()
    
    def _initialize_databases(self) -> None:
        """Initialize database files and schemas for both modes."""
        for mode in ["LIVE", "DEMO"]:
            db_path = f"data/trading_{mode.lower()}.db"
            
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create database and schema
            with self.get_connection_context(mode) as conn:
                self._create_schema(conn, mode)
            
            logger.info(f"Database initialized for {mode} mode: {db_path}")
    
    def _create_schema(self, conn: sqlite3.Connection, mode: str) -> None:
        """Create database schema for the specified mode."""
        cursor = conn.cursor()
        
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_price REAL,
                exit_time TIMESTAMP,
                status TEXT DEFAULT 'OPEN',
                stop_loss REAL,
                take_profit REAL,
                pnl REAL DEFAULT 0.0,
                pnl_percent REAL DEFAULT 0.0,
                mode TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER,
                order_type TEXT NOT NULL,
                side TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                status TEXT DEFAULT 'PENDING',
                filled_quantity INTEGER DEFAULT 0,
                filled_price REAL DEFAULT 0.0,
                submitted_at TIMESTAMP,
                executed_at TIMESTAMP,
                atr REAL,
                atr_multiplier REAL,
                r_multiple REAL,
                bracket_type TEXT,
                volatility_percent REAL,
                mode TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (position_id) REFERENCES positions(id)
            )
        """)
        
        # Trade results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                pnl REAL NOT NULL,
                pnl_percent REAL NOT NULL,
                hold_days INTEGER NOT NULL,
                exit_reason TEXT,
                mode TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (position_id) REFERENCES positions(id)
            )
        """)
        
        # Account balance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_balance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                balance REAL NOT NULL,
                cash REAL NOT NULL,
                invested REAL NOT NULL,
                total_pnl REAL DEFAULT 0.0,
                daily_pnl REAL DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                symbol TEXT,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                mode TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Predictions table (for AI model predictions tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                confidence REAL NOT NULL,
                features TEXT,
                mode TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk metrics table (for risk calculation tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                position_size REAL NOT NULL,
                risk_amount REAL NOT NULL,
                risk_percent REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_reward_ratio REAL NOT NULL,
                max_position_value REAL NOT NULL,
                portfolio_risk REAL NOT NULL,
                raw_confidence REAL DEFAULT 0.0,
                calibrated_confidence REAL DEFAULT 0.0,
                drawdown_scale REAL DEFAULT 1.0,
                daily_drawdown REAL DEFAULT 0.0,
                kelly_fraction REAL DEFAULT 0.0,
                mode TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_level TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                mode TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Phase 0: API validation log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_validation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                validation_type TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                response_time_ms INTEGER,
                status_code INTEGER,
                error_message TEXT,
                response_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mode TEXT DEFAULT 'SYSTEM'
            )
        """)
        
        # Phase 0: Phase execution tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phase_execution_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase_name TEXT NOT NULL,
                execution_id TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_ms INTEGER,
                status TEXT NOT NULL,
                success BOOLEAN,
                error_message TEXT,
                step_labels TEXT,
                mode TEXT DEFAULT 'SYSTEM',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Phase 0: API usage metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                endpoint TEXT,
                request_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                total_response_time_ms INTEGER DEFAULT 0,
                rate_limit_hits INTEGER DEFAULT 0,
                last_request_time TIMESTAMP,
                window_start TIMESTAMP NOT NULL,
                window_end TIMESTAMP NOT NULL,
                mode TEXT DEFAULT 'SYSTEM',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Phase 3: Data provenance tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL,
                source TEXT NOT NULL,
                source_metadata TEXT,
                quality_score REAL,
                quality_level TEXT,
                validation_timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Phase 3: Data quality violations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                column_name TEXT,
                violation_value TEXT,
                expected_range TEXT,
                validation_timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Phase 4: Confidence calibration table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS confidence_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                model TEXT NOT NULL,
                symbol TEXT NOT NULL,
                raw_confidence REAL NOT NULL,
                calibrated_confidence REAL NOT NULL,
                outcome TEXT,
                pnl REAL,
                exit_date TEXT,
                window_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Phase 5: Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                brier_score REAL NOT NULL,
                accuracy REAL NOT NULL,
                n_predictions INTEGER NOT NULL,
                weight REAL NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Phase 6: Bracket parameters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bracket_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                atr REAL NOT NULL,
                atr_multiplier REAL NOT NULL,
                r_multiple REAL NOT NULL,
                bracket_type TEXT NOT NULL,
                volatility_percent REAL NOT NULL,
                risk_amount REAL NOT NULL,
                reward_amount REAL NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Phase 6: Portfolio snapshots table for drawdown tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                total_positions INTEGER NOT NULL,
                open_positions INTEGER NOT NULL,
                total_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Phase 7: Regime state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                regime TEXT NOT NULL,
                trend_direction TEXT NOT NULL,
                volatility_level TEXT NOT NULL,
                trend_strength REAL NOT NULL,
                volatility_ratio REAL NOT NULL,
                atr_percentile REAL NOT NULL,
                regime_confidence REAL NOT NULL,
                transition_probability REAL NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_mode ON positions(mode)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_mode ON orders(mode)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_provenance_symbol ON data_provenance(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_quality_violations_symbol ON data_quality_violations(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence_calibration_model ON confidence_calibration(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence_calibration_symbol ON confidence_calibration(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence_calibration_trade_date ON confidence_calibration(trade_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_window_start ON model_performance(window_start)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bracket_parameters_symbol ON bracket_parameters(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bracket_parameters_mode ON bracket_parameters(mode)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bracket_parameters_created_at ON bracket_parameters(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_mode ON portfolio_snapshots(mode)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_state_symbol ON regime_state(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_state_timestamp ON regime_state(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_state_regime ON regime_state(regime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_state_mode ON regime_state(mode)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_results_symbol ON trade_results(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_results_mode ON trade_results(mode)")
        
        # Phase 0: Indexes for new tables
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_validation_log_api_name ON api_validation_log(api_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_validation_log_timestamp ON api_validation_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_execution_tracking_phase_name ON phase_execution_tracking(phase_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_execution_tracking_execution_id ON phase_execution_tracking(execution_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_execution_tracking_start_time ON phase_execution_tracking(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_metrics_api_name ON api_usage_metrics(api_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_metrics_window_start ON api_usage_metrics(window_start)")
        
        # Run migrations for existing databases
        self._run_migrations(cursor, mode)
        
        conn.commit()
        logger.info(f"Database schema created for {mode} mode")
    
    def _run_migrations(self, cursor: sqlite3.Cursor, mode: str) -> None:
        """Run database migrations for schema updates."""
        try:
            # Check if new columns exist in orders table
            cursor.execute("PRAGMA table_info(orders)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Add new columns if they don't exist
            new_columns = [
                ("atr", "REAL"),
                ("atr_multiplier", "REAL"),
                ("r_multiple", "REAL"),
                ("bracket_type", "TEXT"),
                ("volatility_percent", "REAL")
            ]
            
            # Check if regime_state table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regime_state'")
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE TABLE regime_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        trend_direction TEXT NOT NULL,
                        volatility_level TEXT NOT NULL,
                        trend_strength REAL NOT NULL,
                        volatility_ratio REAL NOT NULL,
                        atr_percentile REAL NOT NULL,
                        regime_confidence REAL NOT NULL,
                        transition_probability REAL NOT NULL,
                        mode TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                logger.info(f"Created regime_state table in {mode} mode")
            
            for column_name, column_type in new_columns:
                if column_name not in columns:
                    cursor.execute(f"ALTER TABLE orders ADD COLUMN {column_name} {column_type}")
                    logger.info(f"Added column {column_name} to orders table in {mode} mode")
            
        except Exception as e:
            logger.error(f"Error running migrations for {mode} mode: {e}")
    
    def get_connection(self, mode: Optional[str] = None):
        """Get database connection for the specified mode."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        db_path = f"data/trading_{mode.lower()}.db"
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        
        return conn
    
    @contextmanager
    def get_connection_context(self, mode: Optional[str] = None):
        """Get database connection context manager for the specified mode."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        db_path = f"data/trading_{mode.lower()}.db"
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        
        try:
            yield conn
        finally:
            conn.close()
    
    def get_current_connection(self):
        """Get database connection for the current mode."""
        return self.get_connection()
    
    def execute_query(self, query: str, params: tuple = (), mode: Optional[str] = None) -> list:
        """Execute a query and return results."""
        with self.get_connection_context(mode) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = (), mode: Optional[str] = None) -> int:
        """Execute an update query and return affected rows."""
        with self.get_connection_context(mode) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def get_table_info(self, table_name: str, mode: Optional[str] = None) -> list:
        """Get table schema information."""
        with self.get_connection_context(mode) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            return cursor.fetchall()
    
    def get_table_count(self, table_name: str, mode: Optional[str] = None) -> int:
        """Get row count for a table."""
        with self.get_connection_context(mode) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
    
    def validate_database_integrity(self, mode: Optional[str] = None) -> bool:
        """Validate database integrity."""
        try:
            with self.get_connection_context(mode) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                
                if result == "ok":
                    logger.info(f"Database integrity check passed for {mode or 'current'} mode")
                    return True
                else:
                    logger.error(f"Database integrity check failed for {mode or 'current'} mode: {result}")
                    return False
        except Exception as e:
            logger.error(f"Database integrity check error: {e}")
            return False
    
    def backup_database(self, mode: str, backup_path: Optional[str] = None) -> bool:
        """Create a backup of the database."""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backups/{mode}_backup_{timestamp}.db"
            
            # Ensure backup directory exists
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Copy database file
            import shutil
            db_path = f"data/trading_{mode.lower()}.db"
            shutil.copy2(db_path, backup_path)
            
            logger.info(f"Database backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def get_database_stats(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get database statistics."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        stats = {
            "mode": mode,
            "database_path": f"data/trading_{mode.lower()}.db",
            "tables": {}
        }
        
        tables = [
            "positions", "orders", "trade_results", "account_balance", "risk_events", "system_logs",
            "api_validation_log", "phase_execution_tracking", "api_usage_metrics"
        ]
        
        for table in tables:
            try:
                count = self.get_table_count(table, mode)
                stats["tables"][table] = count
            except Exception as e:
                logger.warning(f"Failed to get count for table {table}: {e}")
                stats["tables"][table] = "error"
        
        return stats
    
    def log_api_validation(self, api_name: str, validation_type: str, success: bool, 
                          response_time_ms: int = None, status_code: int = None, 
                          error_message: str = None, response_data: str = None, 
                          mode: str = None) -> int:
        """Log API validation result."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        query = """
            INSERT INTO api_validation_log 
            (api_name, validation_type, success, response_time_ms, status_code, 
             error_message, response_data, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_update(
            query, 
            (api_name, validation_type, success, response_time_ms, status_code, 
             error_message, response_data, mode),
            mode
        )
    
    def log_phase_execution(self, phase_name: str, execution_id: str, start_time: datetime,
                           end_time: datetime = None, duration_ms: int = None, 
                           status: str = "RUNNING", success: bool = None, 
                           error_message: str = None, step_labels: str = None, 
                           mode: str = None) -> int:
        """Log phase execution tracking."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        query = """
            INSERT INTO phase_execution_tracking 
            (phase_name, execution_id, start_time, end_time, duration_ms, 
             status, success, error_message, step_labels, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_update(
            query,
            (phase_name, execution_id, start_time, end_time, duration_ms,
             status, success, error_message, step_labels, mode),
            mode
        )
    
    def log_api_usage_metrics(self, api_name: str, endpoint: str, request_count: int,
                             success_count: int, error_count: int, total_response_time_ms: int,
                             rate_limit_hits: int, last_request_time: datetime,
                             window_start: datetime, window_end: datetime, mode: str = None) -> int:
        """Log API usage metrics."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        query = """
            INSERT INTO api_usage_metrics 
            (api_name, endpoint, request_count, success_count, error_count, 
             total_response_time_ms, rate_limit_hits, last_request_time, 
             window_start, window_end, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_update(
            query,
            (api_name, endpoint, request_count, success_count, error_count,
             total_response_time_ms, rate_limit_hits, last_request_time,
             window_start, window_end, mode),
            mode
        )
    
    def get_api_validation_history(self, api_name: str = None, limit: int = 100, mode: str = None) -> list:
        """Get API validation history."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        if api_name:
            query = """
                SELECT * FROM api_validation_log 
                WHERE api_name = ? AND mode = ?
                ORDER BY timestamp DESC LIMIT ?
            """
            return self.execute_query(query, (api_name, mode, limit), mode)
        else:
            query = """
                SELECT * FROM api_validation_log 
                WHERE mode = ?
                ORDER BY timestamp DESC LIMIT ?
            """
            return self.execute_query(query, (mode, limit), mode)
    
    def get_phase_execution_history(self, phase_name: str = None, limit: int = 100, mode: str = None) -> list:
        """Get phase execution history."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        if phase_name:
            query = """
                SELECT * FROM phase_execution_tracking 
                WHERE phase_name = ? AND mode = ?
                ORDER BY start_time DESC LIMIT ?
            """
            return self.execute_query(query, (phase_name, mode, limit), mode)
        else:
            query = """
                SELECT * FROM phase_execution_tracking 
                WHERE mode = ?
                ORDER BY start_time DESC LIMIT ?
            """
            return self.execute_query(query, (mode, limit), mode)
    
    def get_api_usage_metrics(self, api_name: str = None, window_start: datetime = None, mode: str = None) -> list:
        """Get API usage metrics."""
        if mode is None:
            mode = mode or "DEMO"  # Default to DEMO mode
        
        if api_name and window_start:
            query = """
                SELECT * FROM api_usage_metrics 
                WHERE api_name = ? AND window_start >= ? AND mode = ?
                ORDER BY window_start DESC
            """
            return self.execute_query(query, (api_name, window_start, mode), mode)
        elif api_name:
            query = """
                SELECT * FROM api_usage_metrics 
                WHERE api_name = ? AND mode = ?
                ORDER BY window_start DESC
            """
            return self.execute_query(query, (api_name, mode), mode)
        else:
            query = """
                SELECT * FROM api_usage_metrics 
                WHERE mode = ?
                ORDER BY window_start DESC
            """
            return self.execute_query(query, (mode,), mode)
    
    def log_data_provenance(self, symbol: str, data_type: str, source: str, 
                           source_metadata: Dict[str, Any] = None, quality_score: float = None,
                           quality_level: str = None, mode: str = None) -> int:
        """Log data provenance information."""
        mode = mode or "DEMO"
        now = datetime.now().isoformat()
        
        query = """
            INSERT INTO data_provenance 
            (symbol, data_type, source, source_metadata, quality_score, quality_level, 
             validation_timestamp, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        metadata_json = json.dumps(source_metadata) if source_metadata else None
        
        return self.execute_update(
            query, 
            (symbol, data_type, source, metadata_json, quality_score, quality_level, now, now, now),
            mode
        )
    
    def log_data_quality_violation(self, symbol: str, violation_type: str, severity: str,
                                  description: str, column_name: str = None, 
                                  violation_value: Any = None, expected_range: Tuple[float, float] = None,
                                  mode: str = None) -> int:
        """Log data quality violation."""
        mode = mode or "DEMO"
        now = datetime.now().isoformat()
        
        query = """
            INSERT INTO data_quality_violations 
            (symbol, violation_type, severity, description, column_name, violation_value, 
             expected_range, validation_timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        value_str = str(violation_value) if violation_value is not None else None
        range_str = f"{expected_range[0]}-{expected_range[1]}" if expected_range else None
        
        return self.execute_update(
            query,
            (symbol, violation_type, severity, description, column_name, value_str, range_str, now, now),
            mode
        )
    
    def get_data_provenance_history(self, symbol: str = None, data_type: str = None, 
                                   limit: int = 100, mode: str = None) -> List[Dict[str, Any]]:
        """Get data provenance history."""
        mode = mode or "DEMO"
        
        if symbol and data_type:
            query = """
                SELECT * FROM data_provenance 
                WHERE symbol = ? AND data_type = ?
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (symbol, data_type, limit), mode)
        elif symbol:
            query = """
                SELECT * FROM data_provenance 
                WHERE symbol = ?
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (symbol, limit), mode)
        else:
            query = """
                SELECT * FROM data_provenance 
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (limit,), mode)
    
    def get_data_quality_violations(self, symbol: str = None, severity: str = None,
                                   limit: int = 100, mode: str = None) -> List[Dict[str, Any]]:
        """Get data quality violations."""
        mode = mode or "DEMO"
        
        if symbol and severity:
            query = """
                SELECT * FROM data_quality_violations 
                WHERE symbol = ? AND severity = ?
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (symbol, severity, limit), mode)
        elif symbol:
            query = """
                SELECT * FROM data_quality_violations 
                WHERE symbol = ?
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (symbol, limit), mode)
        elif severity:
            query = """
                SELECT * FROM data_quality_violations 
                WHERE severity = ?
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (severity, limit), mode)
        else:
            query = """
                SELECT * FROM data_quality_violations 
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (limit,), mode)
    
    def log_confidence_calibration(self, trade_date: datetime, model: str, symbol: str,
                                  raw_confidence: float, calibrated_confidence: float,
                                  outcome: str = None, pnl: float = None, 
                                  exit_date: datetime = None, window_id: str = "",
                                  mode: str = None) -> int:
        """Log confidence calibration data."""
        mode = mode or "DEMO"
        now = datetime.now().isoformat()
        
        query = """
            INSERT INTO confidence_calibration 
            (trade_date, model, symbol, raw_confidence, calibrated_confidence, 
             outcome, pnl, exit_date, window_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_update(
            query,
            (trade_date.isoformat(), model, symbol, raw_confidence, calibrated_confidence,
             outcome, pnl, exit_date.isoformat() if exit_date else None, window_id, now, now),
            mode
        )
    
    def get_confidence_calibration_history(self, model: str = None, symbol: str = None,
                                          limit: int = 100, mode: str = None) -> List[Dict[str, Any]]:
        """Get confidence calibration history."""
        mode = mode or "DEMO"
        
        if model and symbol:
            query = """
                SELECT * FROM confidence_calibration 
                WHERE model = ? AND symbol = ?
                ORDER BY trade_date DESC
                LIMIT ?
            """
            return self.execute_query(query, (model, symbol, limit), mode)
        elif model:
            query = """
                SELECT * FROM confidence_calibration 
                WHERE model = ?
                ORDER BY trade_date DESC
                LIMIT ?
            """
            return self.execute_query(query, (model, limit), mode)
        elif symbol:
            query = """
                SELECT * FROM confidence_calibration 
                WHERE symbol = ?
                ORDER BY trade_date DESC
                LIMIT ?
            """
            return self.execute_query(query, (symbol, limit), mode)
        else:
            query = """
                SELECT * FROM confidence_calibration 
                ORDER BY trade_date DESC
                LIMIT ?
            """
            return self.execute_query(query, (limit,), mode)
    
    def update_confidence_calibration_outcome(self, trade_id: str, outcome: str, 
                                            pnl: float = None, exit_date: datetime = None,
                                            mode: str = None) -> int:
        """Update confidence calibration with trade outcome."""
        mode = mode or "DEMO"
        now = datetime.now().isoformat()
        
        query = """
            UPDATE confidence_calibration 
            SET outcome = ?, pnl = ?, exit_date = ?, updated_at = ?
            WHERE id = ?
        """
        
        return self.execute_update(
            query,
            (outcome, pnl, exit_date.isoformat() if exit_date else None, now, trade_id),
            mode
        )
    
    def log_model_performance(self, timestamp: datetime, model: str, brier_score: float,
                            accuracy: float, n_predictions: int, weight: float,
                            window_start: datetime, window_end: datetime,
                            mode: str = None) -> int:
        """Log model performance data."""
        mode = mode or "DEMO"
        now = datetime.now().isoformat()
        
        query = """
            INSERT INTO model_performance 
            (timestamp, model, brier_score, accuracy, n_predictions, weight,
             window_start, window_end, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_update(
            query,
            (timestamp.isoformat(), model, brier_score, accuracy, n_predictions, weight,
             window_start.isoformat(), window_end.isoformat(), now, now),
            mode
        )
    
    def get_model_performance_history(self, model: str = None, limit: int = 100, 
                                    mode: str = None) -> List[Dict[str, Any]]:
        """Get model performance history."""
        mode = mode or "DEMO"
        
        if model:
            query = """
                SELECT * FROM model_performance 
                WHERE model = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (model, limit), mode)
        else:
            query = """
                SELECT * FROM model_performance 
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (limit,), mode)
    
    def get_latest_model_performance(self, model: str = None, mode: str = None) -> List[Dict[str, Any]]:
        """Get latest model performance for each model."""
        mode = mode or "DEMO"
        
        if model:
            query = """
                SELECT * FROM model_performance 
                WHERE model = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            return self.execute_query(query, (model,), mode)
        else:
            query = """
                SELECT mp1.* FROM model_performance mp1
                INNER JOIN (
                    SELECT model, MAX(timestamp) as max_timestamp
                    FROM model_performance
                    GROUP BY model
                ) mp2 ON mp1.model = mp2.model AND mp1.timestamp = mp2.max_timestamp
                ORDER BY mp1.model
            """
            return self.execute_query(query, (), mode)
    
    def log_bracket_parameters(self, symbol: str, entry_price: float, stop_loss: float,
                             take_profit: float, atr: float, atr_multiplier: float,
                             r_multiple: float, bracket_type: str, volatility_percent: float,
                             risk_amount: float, reward_amount: float, mode: str = None) -> int:
        """Log bracket parameters."""
        mode = mode or "DEMO"
        now = datetime.now().isoformat()
        
        query = """
            INSERT INTO bracket_parameters 
            (symbol, entry_price, stop_loss, take_profit, atr, atr_multiplier,
             r_multiple, bracket_type, volatility_percent, risk_amount, reward_amount,
             mode, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_update(
            query,
            (symbol, entry_price, stop_loss, take_profit, atr, atr_multiplier,
             r_multiple, bracket_type, volatility_percent, risk_amount, reward_amount,
             mode, now, now),
            mode
        )
    
    def get_bracket_parameters_history(self, symbol: str = None, limit: int = 100, 
                                     mode: str = None) -> List[Dict[str, Any]]:
        """Get bracket parameters history."""
        mode = mode or "DEMO"
        
        if symbol:
            query = """
                SELECT * FROM bracket_parameters 
                WHERE symbol = ? AND mode = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            return self.execute_query(query, (symbol, mode, limit), mode)
        else:
            query = """
                SELECT * FROM bracket_parameters 
                WHERE mode = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            return self.execute_query(query, (mode, limit), mode)
    
    def log_portfolio_snapshot(self, portfolio_value: float, cash_balance: float,
                             total_positions: int, open_positions: int, total_pnl: float,
                             daily_pnl: float, mode: str = None) -> int:
        """Log portfolio snapshot for drawdown tracking."""
        mode = mode or "DEMO"
        now = datetime.now().isoformat()
        
        query = """
            INSERT INTO portfolio_snapshots 
            (timestamp, portfolio_value, cash_balance, total_positions, open_positions,
             total_pnl, daily_pnl, mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_update(
            query,
            (now, portfolio_value, cash_balance, total_positions, open_positions,
             total_pnl, daily_pnl, mode, now),
            mode
        )
    
    def get_portfolio_snapshots(self, start_date: datetime = None, end_date: datetime = None,
                              mode: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get portfolio snapshots for drawdown analysis."""
        mode = mode or "DEMO"
        
        if start_date and end_date:
            query = """
                SELECT * FROM portfolio_snapshots 
                WHERE mode = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (mode, start_date.isoformat(), 
                                            end_date.isoformat(), limit), mode)
        elif start_date:
            query = """
                SELECT * FROM portfolio_snapshots 
                WHERE mode = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (mode, start_date.isoformat(), limit), mode)
        else:
            query = """
                SELECT * FROM portfolio_snapshots 
                WHERE mode = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (mode, limit), mode)

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_connection(mode: Optional[str] = None):
    """Get database connection for the specified mode."""
    return get_database_manager().get_connection(mode)

def execute_query(query: str, params: tuple = (), mode: Optional[str] = None) -> list:
    """Execute a query and return results."""
    return get_database_manager().execute_query(query, params, mode)

def execute_update(query: str, params: tuple = (), mode: Optional[str] = None) -> int:
    """Execute an update query and return affected rows."""
    return get_database_manager().execute_update(query, params, mode)

def validate_database_integrity(mode: Optional[str] = None) -> bool:
    """Validate database integrity."""
    return get_database_manager().validate_database_integrity(mode)

def get_database_stats(mode: Optional[str] = None) -> Dict[str, Any]:
    """Get database statistics."""
    return get_database_manager().get_database_stats(mode)

def log_api_validation(api_name: str, validation_type: str, success: bool, 
                      response_time_ms: int = None, status_code: int = None, 
                      error_message: str = None, response_data: str = None, 
                      mode: str = None) -> int:
    """Log API validation result."""
    return get_database_manager().log_api_validation(
        api_name, validation_type, success, response_time_ms, status_code, 
        error_message, response_data, mode
    )

def log_phase_execution(phase_name: str, execution_id: str, start_time: datetime,
                       end_time: datetime = None, duration_ms: int = None, 
                       status: str = "RUNNING", success: bool = None, 
                       error_message: str = None, step_labels: str = None, 
                       mode: str = None) -> int:
    """Log phase execution tracking."""
    return get_database_manager().log_phase_execution(
        phase_name, execution_id, start_time, end_time, duration_ms,
        status, success, error_message, step_labels, mode
    )

def log_api_usage_metrics(api_name: str, endpoint: str, request_count: int,
                         success_count: int, error_count: int, total_response_time_ms: int,
                         rate_limit_hits: int, last_request_time: datetime,
                         window_start: datetime, window_end: datetime, mode: str = None) -> int:
    """Log API usage metrics."""
    return get_database_manager().log_api_usage_metrics(
        api_name, endpoint, request_count, success_count, error_count,
        total_response_time_ms, rate_limit_hits, last_request_time,
        window_start, window_end, mode
    )

def get_api_validation_history(api_name: str = None, limit: int = 100, mode: str = None) -> list:
    """Get API validation history."""
    return get_database_manager().get_api_validation_history(api_name, limit, mode)

def get_phase_execution_history(phase_name: str = None, limit: int = 100, mode: str = None) -> list:
    """Get phase execution history."""
    return get_database_manager().get_phase_execution_history(phase_name, limit, mode)

def get_api_usage_metrics(api_name: str = None, window_start: datetime = None, mode: str = None) -> list:
    """Get API usage metrics."""
    return get_database_manager().get_api_usage_metrics(api_name, window_start, mode)

def log_data_provenance(symbol: str, data_type: str, source: str, 
                       source_metadata: Dict[str, Any] = None, quality_score: float = None,
                       quality_level: str = None, mode: str = None) -> int:
    """Log data provenance information."""
    return get_database_manager().log_data_provenance(
        symbol, data_type, source, source_metadata, quality_score, quality_level, mode
    )

def log_data_quality_violation(symbol: str, violation_type: str, severity: str,
                              description: str, column_name: str = None, 
                              violation_value: Any = None, expected_range: Tuple[float, float] = None,
                              mode: str = None) -> int:
    """Log data quality violation."""
    return get_database_manager().log_data_quality_violation(
        symbol, violation_type, severity, description, column_name, 
        violation_value, expected_range, mode
    )

def get_data_provenance_history(symbol: str = None, data_type: str = None, 
                               limit: int = 100, mode: str = None) -> List[Dict[str, Any]]:
    """Get data provenance history."""
    return get_database_manager().get_data_provenance_history(symbol, data_type, limit, mode)

def get_data_quality_violations(symbol: str = None, severity: str = None,
                               limit: int = 100, mode: str = None) -> List[Dict[str, Any]]:
    """Get data quality violations."""
    return get_database_manager().get_data_quality_violations(symbol, severity, limit, mode)

def log_confidence_calibration(trade_date: datetime, model: str, symbol: str,
                              raw_confidence: float, calibrated_confidence: float,
                              outcome: str = None, pnl: float = None, 
                              exit_date: datetime = None, window_id: str = "",
                              mode: str = None) -> int:
    """Log confidence calibration data."""
    return get_database_manager().log_confidence_calibration(
        trade_date, model, symbol, raw_confidence, calibrated_confidence,
        outcome, pnl, exit_date, window_id, mode
    )

def get_confidence_calibration_history(model: str = None, symbol: str = None,
                                      limit: int = 100, mode: str = None) -> List[Dict[str, Any]]:
    """Get confidence calibration history."""
    return get_database_manager().get_confidence_calibration_history(model, symbol, limit, mode)

def update_confidence_calibration_outcome(trade_id: str, outcome: str, 
                                        pnl: float = None, exit_date: datetime = None,
                                        mode: str = None) -> int:
    """Update confidence calibration with trade outcome."""
    return get_database_manager().update_confidence_calibration_outcome(
        trade_id, outcome, pnl, exit_date, mode
    )

def log_model_performance(timestamp: datetime, model: str, brier_score: float,
                         accuracy: float, n_predictions: int, weight: float,
                         window_start: datetime, window_end: datetime,
                         mode: str = None) -> int:
    """Log model performance data."""
    return get_database_manager().log_model_performance(
        timestamp, model, brier_score, accuracy, n_predictions, weight,
        window_start, window_end, mode
    )

def get_model_performance_history(model: str = None, limit: int = 100, 
                                mode: str = None) -> List[Dict[str, Any]]:
    """Get model performance history."""
    return get_database_manager().get_model_performance_history(model, limit, mode)

def get_latest_model_performance(model: str = None, mode: str = None) -> List[Dict[str, Any]]:
    """Get latest model performance for each model."""
    return get_database_manager().get_latest_model_performance(model, mode)

def log_bracket_parameters(symbol: str, entry_price: float, stop_loss: float,
                         take_profit: float, atr: float, atr_multiplier: float,
                         r_multiple: float, bracket_type: str, volatility_percent: float,
                         risk_amount: float, reward_amount: float, mode: str = None) -> int:
    """Log bracket parameters."""
    return get_database_manager().log_bracket_parameters(
        symbol, entry_price, stop_loss, take_profit, atr, atr_multiplier,
        r_multiple, bracket_type, volatility_percent, risk_amount, reward_amount, mode
    )

def get_bracket_parameters_history(symbol: str = None, limit: int = 100, 
                                 mode: str = None) -> List[Dict[str, Any]]:
    """Get bracket parameters history."""
    return get_database_manager().get_bracket_parameters_history(symbol, limit, mode)

def log_portfolio_snapshot(portfolio_value: float, cash_balance: float,
                         total_positions: int, open_positions: int, total_pnl: float,
                         daily_pnl: float, mode: str = None) -> int:
    """Log portfolio snapshot for drawdown tracking."""
    return get_database_manager().log_portfolio_snapshot(
        portfolio_value, cash_balance, total_positions, open_positions, total_pnl, daily_pnl, mode
    )

def get_portfolio_snapshots(start_date: datetime = None, end_date: datetime = None,
                          mode: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get portfolio snapshots for drawdown analysis."""
    return get_database_manager().get_portfolio_snapshots(start_date, end_date, mode, limit)
