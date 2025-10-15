"""
Change Tracker Module

This module implements comprehensive change tracking to log all modifications,
additions, and deletions throughout the system for audit and debugging purposes.
"""

import json
import logging
import os
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of changes that can be tracked."""
    FILE_CREATION = "FILE_CREATION"
    FILE_MODIFICATION = "FILE_MODIFICATION"
    FILE_DELETION = "FILE_DELETION"
    FUNCTION_ADDITION = "FUNCTION_ADDITION"
    FUNCTION_MODIFICATION = "FUNCTION_MODIFICATION"
    FUNCTION_DELETION = "FUNCTION_DELETION"
    CLASS_ADDITION = "CLASS_ADDITION"
    CLASS_MODIFICATION = "CLASS_MODIFICATION"
    CLASS_DELETION = "CLASS_DELETION"
    DEPENDENCY_ADDITION = "DEPENDENCY_ADDITION"
    DEPENDENCY_REMOVAL = "DEPENDENCY_REMOVAL"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    DATABASE_SCHEMA_CHANGE = "DATABASE_SCHEMA_CHANGE"
    API_CHANGE = "API_CHANGE"
    SECURITY_CHANGE = "SECURITY_CHANGE"

class ChangeSeverity(Enum):
    """Severity levels for changes."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class ChangeRecord:
    """Record of a single change."""
    change_id: str
    change_type: ChangeType
    severity: ChangeSeverity
    timestamp: datetime
    file_path: str
    description: str
    reason: str
    impact: str
    author: str
    phase: str
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ChangeSummary:
    """Summary of changes for a specific period or scope."""
    total_changes: int
    changes_by_type: Dict[str, int]
    changes_by_severity: Dict[str, int]
    changes_by_phase: Dict[str, int]
    changes_by_author: Dict[str, int]
    time_range: tuple
    most_modified_files: List[tuple]  # (file_path, change_count)

class ChangeTracker:
    """Comprehensive change tracking system."""
    
    def __init__(self, db_path: str = "data/change_log.db"):
        self.db_path = db_path
        self._ensure_database()
        logger.info(f"Change Tracker initialized with database: {db_path}")
    
    def _ensure_database(self):
        """Ensure the change tracking database exists and is properly initialized."""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create database and tables
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create changes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS changes (
                    change_id TEXT PRIMARY KEY,
                    change_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    description TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    impact TEXT NOT NULL,
                    author TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    before_hash TEXT,
                    after_hash TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_changes_timestamp 
                ON changes(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_changes_file_path 
                ON changes(file_path)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_changes_type 
                ON changes(change_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_changes_phase 
                ON changes(phase)
            """)
            
            conn.commit()
    
    def _generate_change_id(self, change_type: ChangeType, file_path: str, timestamp: datetime) -> str:
        """Generate a unique change ID."""
        content = f"{change_type.value}_{file_path}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate hash of a file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
        return None
    
    def track_change(
        self,
        change_type: ChangeType,
        file_path: str,
        description: str,
        reason: str,
        impact: str,
        author: str = "system",
        phase: str = "unknown",
        severity: ChangeSeverity = ChangeSeverity.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track a change in the system."""
        timestamp = datetime.now()
        change_id = self._generate_change_id(change_type, file_path, timestamp)
        
        # Calculate file hashes
        before_hash = None
        after_hash = None
        
        if change_type in [ChangeType.FILE_MODIFICATION, ChangeType.FILE_DELETION]:
            before_hash = self._calculate_file_hash(file_path)
        
        if change_type in [ChangeType.FILE_CREATION, ChangeType.FILE_MODIFICATION]:
            after_hash = self._calculate_file_hash(file_path)
        
        # Create change record
        change_record = ChangeRecord(
            change_id=change_id,
            change_type=change_type,
            severity=severity,
            timestamp=timestamp,
            file_path=file_path,
            description=description,
            reason=reason,
            impact=impact,
            author=author,
            phase=phase,
            before_hash=before_hash,
            after_hash=after_hash,
            metadata=metadata
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO changes (
                    change_id, change_type, severity, timestamp, file_path,
                    description, reason, impact, author, phase,
                    before_hash, after_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                change_record.change_id,
                change_record.change_type.value,
                change_record.severity.value,
                change_record.timestamp.isoformat(),
                change_record.file_path,
                change_record.description,
                change_record.reason,
                change_record.impact,
                change_record.author,
                change_record.phase,
                change_record.before_hash,
                change_record.after_hash,
                json.dumps(change_record.metadata) if change_record.metadata else None
            ))
            conn.commit()
        
        logger.info(f"Change tracked: {change_type.value} - {file_path}")
        return change_id
    
    def get_changes(
        self,
        file_path: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        phase: Optional[str] = None,
        author: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ChangeRecord]:
        """Retrieve changes based on filters."""
        query = "SELECT * FROM changes WHERE 1=1"
        params = []
        
        if file_path:
            query += " AND file_path = ?"
            params.append(file_path)
        
        if change_type:
            query += " AND change_type = ?"
            params.append(change_type.value)
        
        if phase:
            query += " AND phase = ?"
            params.append(phase)
        
        if author:
            query += " AND author = ?"
            params.append(author)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        changes = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                change_record = ChangeRecord(
                    change_id=row[0],
                    change_type=ChangeType(row[1]),
                    severity=ChangeSeverity(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    file_path=row[4],
                    description=row[5],
                    reason=row[6],
                    impact=row[7],
                    author=row[8],
                    phase=row[9],
                    before_hash=row[10],
                    after_hash=row[11],
                    metadata=json.loads(row[12]) if row[12] else None
                )
                changes.append(change_record)
        
        return changes
    
    def get_change_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        phase: Optional[str] = None
    ) -> ChangeSummary:
        """Get a summary of changes for a specific period."""
        query = "SELECT * FROM changes WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if phase:
            query += " AND phase = ?"
            params.append(phase)
        
        changes_by_type = {}
        changes_by_severity = {}
        changes_by_phase = {}
        changes_by_author = {}
        file_changes = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                change_type = row[1]
                severity = row[2]
                phase_name = row[9]
                author = row[8]
                file_path = row[4]
                
                # Count by type
                changes_by_type[change_type] = changes_by_type.get(change_type, 0) + 1
                
                # Count by severity
                changes_by_severity[severity] = changes_by_severity.get(severity, 0) + 1
                
                # Count by phase
                changes_by_phase[phase_name] = changes_by_phase.get(phase_name, 0) + 1
                
                # Count by author
                changes_by_author[author] = changes_by_author.get(author, 0) + 1
                
                # Count by file
                file_changes[file_path] = file_changes.get(file_path, 0) + 1
        
        # Get most modified files
        most_modified_files = sorted(file_changes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        total_changes = sum(changes_by_type.values())
        
        return ChangeSummary(
            total_changes=total_changes,
            changes_by_type=changes_by_type,
            changes_by_severity=changes_by_severity,
            changes_by_phase=changes_by_phase,
            changes_by_author=changes_by_author,
            time_range=(start_date, end_date),
            most_modified_files=most_modified_files
        )
    
    def generate_change_report(self, summary: ChangeSummary) -> str:
        """Generate a human-readable change report."""
        report_lines = [
            "=" * 60,
            "CHANGE TRACKING REPORT",
            "=" * 60,
            f"Time Range: {summary.time_range[0]} to {summary.time_range[1]}",
            f"Total Changes: {summary.total_changes}",
            "",
            "CHANGES BY TYPE:",
            "-" * 20
        ]
        
        for change_type, count in summary.changes_by_type.items():
            report_lines.append(f"  {change_type}: {count}")
        
        report_lines.extend([
            "",
            "CHANGES BY SEVERITY:",
            "-" * 20
        ])
        
        for severity, count in summary.changes_by_severity.items():
            report_lines.append(f"  {severity}: {count}")
        
        report_lines.extend([
            "",
            "CHANGES BY PHASE:",
            "-" * 20
        ])
        
        for phase, count in summary.changes_by_phase.items():
            report_lines.append(f"  {phase}: {count}")
        
        report_lines.extend([
            "",
            "CHANGES BY AUTHOR:",
            "-" * 20
        ])
        
        for author, count in summary.changes_by_author.items():
            report_lines.append(f"  {author}: {count}")
        
        report_lines.extend([
            "",
            "MOST MODIFIED FILES:",
            "-" * 20
        ])
        
        for file_path, count in summary.most_modified_files:
            report_lines.append(f"  {file_path}: {count} changes")
        
        return "\n".join(report_lines)
    
    def export_changes(self, output_file: str, format: str = "json") -> bool:
        """Export changes to a file."""
        try:
            changes = self.get_changes(limit=10000)  # Get all changes
            
            if format.lower() == "json":
                with open(output_file, 'w') as f:
                    json.dump([asdict(change) for change in changes], f, indent=2, default=str)
            elif format.lower() == "csv":
                import csv
                with open(output_file, 'w', newline='') as f:
                    if changes:
                        writer = csv.DictWriter(f, fieldnames=asdict(changes[0]).keys())
                        writer.writeheader()
                        for change in changes:
                            writer.writerow(asdict(change))
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Changes exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting changes: {e}")
            return False

# Global change tracker instance
_change_tracker: Optional[ChangeTracker] = None

def get_change_tracker() -> ChangeTracker:
    """Get the global change tracker instance."""
    global _change_tracker
    if _change_tracker is None:
        _change_tracker = ChangeTracker()
    return _change_tracker

def track_change(
    change_type: ChangeType,
    file_path: str,
    description: str,
    reason: str,
    impact: str,
    author: str = "system",
    phase: str = "unknown",
    severity: ChangeSeverity = ChangeSeverity.MEDIUM,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Track a change in the system."""
    return get_change_tracker().track_change(
        change_type, file_path, description, reason, impact,
        author, phase, severity, metadata
    )

def get_changes(
    file_path: Optional[str] = None,
    change_type: Optional[ChangeType] = None,
    phase: Optional[str] = None,
    author: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100
) -> List[ChangeRecord]:
    """Retrieve changes based on filters."""
    return get_change_tracker().get_changes(
        file_path, change_type, phase, author, start_date, end_date, limit
    )

def get_change_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    phase: Optional[str] = None
) -> ChangeSummary:
    """Get a summary of changes for a specific period."""
    return get_change_tracker().get_change_summary(start_date, end_date, phase)
