"""
Progress tracking for data collection with persistent state management.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from .interfaces import ProgressTracker


class JSONProgressTracker(ProgressTracker):
    """Progress tracker using JSON file for persistence."""
    
    def __init__(self, progress_file: str = "progress_state.json"):
        self.progress_file = Path(progress_file)
        self.logger = logging.getLogger(__name__)
        
        # Ensure progress file directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress state
        self.progress_data = self._load_or_initialize_progress()
        
        self.logger.info(f"Initialized progress tracker with file: {self.progress_file}")
    
    def _load_or_initialize_progress(self) -> Dict[str, Any]:
        """Load existing progress or initialize new progress state."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded existing progress: {len(data.get('symbols', {}))} symbols tracked")
                return data
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Failed to load progress file, starting fresh: {e}")
        
        # Initialize new progress state
        initial_state = {
            "session_id": f"session_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "total_symbols": 0,
            "completed_symbols": 0,
            "failed_symbols": 0,
            "symbols": {},  # symbol -> {status, start_time, end_time, error, attempts}
            "statistics": {
                "total_data_points": 0,
                "total_file_size_bytes": 0,
                "average_collection_time": 0.0,
                "success_rate": 0.0
            }
        }
        
        self.logger.info("Initialized new progress state")
        return initial_state
    
    def save_progress(self, progress_data: Dict[str, Any]) -> None:
        """Save current progress state to file."""
        try:
            # Update last update timestamp
            progress_data["last_update"] = datetime.now().isoformat()
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(progress_data, f, indent=2, default=str)
            
            # Atomic rename
            temp_file.replace(self.progress_file)
            
            self.logger.debug(f"Progress saved to {self.progress_file}")
            
        except IOError as e:
            self.logger.error(f"Failed to save progress: {e}")
            raise
    
    def load_progress(self) -> Dict[str, Any]:
        """Load current progress state."""
        return self.progress_data.copy()
    
    def mark_symbol_complete(self, symbol: str, success: bool, error: Optional[str] = None) -> None:
        """Mark a symbol as complete (success or failure)."""
        current_time = datetime.now().isoformat()
        
        if symbol not in self.progress_data["symbols"]:
            self.progress_data["symbols"][symbol] = {
                "status": "not_started",
                "start_time": None,
                "end_time": None,
                "error": None,
                "attempts": 0,
                "data_points": 0,
                "file_size_bytes": 0
            }
        
        symbol_data = self.progress_data["symbols"][symbol]
        symbol_data["end_time"] = current_time
        symbol_data["attempts"] += 1
        
        if success:
            symbol_data["status"] = "completed"
            symbol_data["error"] = None
            self.progress_data["completed_symbols"] += 1
            self.logger.info(f"Marked {symbol} as completed")
        else:
            symbol_data["status"] = "failed"
            symbol_data["error"] = error
            self.progress_data["failed_symbols"] += 1
            self.logger.warning(f"Marked {symbol} as failed: {error}")
        
        # Update statistics
        self._update_statistics()
        
        # Save progress
        self.save_progress(self.progress_data)
    
    def mark_symbol_started(self, symbol: str) -> None:
        """Mark a symbol as started."""
        current_time = datetime.now().isoformat()
        
        if symbol not in self.progress_data["symbols"]:
            self.progress_data["symbols"][symbol] = {
                "status": "not_started",
                "start_time": None,
                "end_time": None,
                "error": None,
                "attempts": 0,
                "data_points": 0,
                "file_size_bytes": 0
            }
        
        symbol_data = self.progress_data["symbols"][symbol]
        symbol_data["status"] = "in_progress"
        symbol_data["start_time"] = current_time
        symbol_data["attempts"] += 1
        
        self.logger.debug(f"Marked {symbol} as started (attempt {symbol_data['attempts']})")
        
        # Save progress
        self.save_progress(self.progress_data)
    
    def update_symbol_data(self, symbol: str, data_points: int, file_size_bytes: int) -> None:
        """Update symbol data statistics."""
        if symbol in self.progress_data["symbols"]:
            symbol_data = self.progress_data["symbols"][symbol]
            symbol_data["data_points"] = data_points
            symbol_data["file_size_bytes"] = file_size_bytes
            
            # Update global statistics
            self._update_statistics()
            
            self.logger.debug(f"Updated {symbol} stats: {data_points} points, {file_size_bytes} bytes")
    
    def get_pending_symbols(self) -> List[str]:
        """Get list of symbols that still need to be processed."""
        pending = []
        
        for symbol, data in self.progress_data["symbols"].items():
            if data["status"] in ["not_started", "failed"]:
                pending.append(symbol)
        
        return pending
    
    def get_completed_symbols(self) -> List[str]:
        """Get list of successfully completed symbols."""
        completed = []
        
        for symbol, data in self.progress_data["symbols"].items():
            if data["status"] == "completed":
                completed.append(symbol)
        
        return completed
    
    def get_failed_symbols(self) -> List[str]:
        """Get list of permanently failed symbols."""
        failed = []
        
        for symbol, data in self.progress_data["symbols"].items():
            if data["status"] == "failed":
                failed.append(symbol)
        
        return failed
    
    def initialize_symbols(self, symbols: List[str]) -> None:
        """Initialize tracking for a list of symbols."""
        self.progress_data["total_symbols"] = len(symbols)
        
        for symbol in symbols:
            if symbol not in self.progress_data["symbols"]:
                self.progress_data["symbols"][symbol] = {
                    "status": "not_started",
                    "start_time": None,
                    "end_time": None,
                    "error": None,
                    "attempts": 0,
                    "data_points": 0,
                    "file_size_bytes": 0
                }
        
        self.logger.info(f"Initialized tracking for {len(symbols)} symbols")
        self.save_progress(self.progress_data)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress."""
        total = self.progress_data["total_symbols"]
        completed = self.progress_data["completed_symbols"]
        failed = self.progress_data["failed_symbols"]
        in_progress = sum(1 for data in self.progress_data["symbols"].values() 
                         if data["status"] == "in_progress")
        pending = total - completed - failed - in_progress
        
        progress_percentage = (completed / total * 100) if total > 0 else 0
        
        return {
            "total_symbols": total,
            "completed_symbols": completed,
            "failed_symbols": failed,
            "in_progress_symbols": in_progress,
            "pending_symbols": pending,
            "progress_percentage": round(progress_percentage, 2),
            "success_rate": self.progress_data["statistics"]["success_rate"],
            "session_id": self.progress_data["session_id"],
            "start_time": self.progress_data["start_time"],
            "last_update": self.progress_data["last_update"],
            "statistics": self.progress_data["statistics"]
        }
    
    def _update_statistics(self) -> None:
        """Update global statistics based on symbol data."""
        stats = self.progress_data["statistics"]
        
        # Calculate totals
        total_data_points = 0
        total_file_size = 0
        total_collection_times = []
        successful_collections = 0
        
        for symbol_data in self.progress_data["symbols"].values():
            if symbol_data["status"] == "completed":
                total_data_points += symbol_data.get("data_points", 0)
                total_file_size += symbol_data.get("file_size_bytes", 0)
                successful_collections += 1
                
                # Calculate collection time if both timestamps exist
                if symbol_data.get("start_time") and symbol_data.get("end_time"):
                    try:
                        start = datetime.fromisoformat(symbol_data["start_time"])
                        end = datetime.fromisoformat(symbol_data["end_time"])
                        collection_time = (end - start).total_seconds()
                        total_collection_times.append(collection_time)
                    except ValueError:
                        pass  # Skip invalid timestamps
        
        # Update statistics
        stats["total_data_points"] = total_data_points
        stats["total_file_size_bytes"] = total_file_size
        
        if total_collection_times:
            stats["average_collection_time"] = sum(total_collection_times) / len(total_collection_times)
        
        total_attempts = self.progress_data["completed_symbols"] + self.progress_data["failed_symbols"]
        if total_attempts > 0:
            stats["success_rate"] = round(self.progress_data["completed_symbols"] / total_attempts * 100, 2)
    
    def reset_progress(self) -> None:
        """Reset all progress (use with caution)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
        self.progress_data = self._load_or_initialize_progress()
        self.logger.warning("Progress has been reset")
    
    def export_progress_report(self, output_file: str) -> None:
        """Export detailed progress report to file."""
        report = {
            "summary": self.get_progress_summary(),
            "detailed_symbols": self.progress_data["symbols"],
            "export_time": datetime.now().isoformat()
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Progress report exported to {output_path}")