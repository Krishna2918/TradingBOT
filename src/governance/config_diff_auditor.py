"""
Config Diff Auditor
Tracks and audits configuration changes for compliance and governance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import yaml
import os
import difflib
from pathlib import Path

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of configuration changes"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    RENAMED = "renamed"
    MOVED = "moved"

class ChangeSeverity(Enum):
    """Severity levels for configuration changes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConfigCategory(Enum):
    """Configuration categories"""
    RISK_MANAGEMENT = "risk_management"
    TRADING_PARAMETERS = "trading_parameters"
    AI_MODELS = "ai_models"
    DATA_SOURCES = "data_sources"
    API_KEYS = "api_keys"
    EXECUTION_SETTINGS = "execution_settings"
    MONITORING = "monitoring"
    COMPLIANCE = "compliance"

@dataclass
class ConfigChange:
    """Configuration change record"""
    change_id: str
    timestamp: datetime
    file_path: str
    change_type: ChangeType
    severity: ChangeSeverity
    category: ConfigCategory
    old_value: Any
    new_value: Any
    field_path: str
    description: str
    author: str
    approval_required: bool
    approved: bool = False
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None

@dataclass
class ConfigSnapshot:
    """Configuration snapshot"""
    snapshot_id: str
    timestamp: datetime
    file_path: str
    content_hash: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]

class ConfigDiffAuditor:
    """Audits and tracks configuration changes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.config_history = []
        self.change_history = []
        self.approval_queue = []
        self.watch_paths = config.get('watch_paths', ['config/'])
        self.auto_approve_categories = config.get('auto_approve_categories', [ConfigCategory.MONITORING])
        self.require_approval_categories = config.get('require_approval_categories', [
            ConfigCategory.RISK_MANAGEMENT,
            ConfigCategory.API_KEYS,
            ConfigCategory.COMPLIANCE
        ])
        
        # Initialize baseline snapshots
        self._initialize_baseline_snapshots()
        
        logger.info("Config Diff Auditor initialized")
    
    def _initialize_baseline_snapshots(self):
        """Initialize baseline configuration snapshots"""
        try:
            for watch_path in self.watch_paths:
                if os.path.exists(watch_path):
                    self._create_snapshot(watch_path)
            
            logger.info("Baseline configuration snapshots created")
            
        except Exception as e:
            logger.error(f"Error initializing baseline snapshots: {e}")
    
    def _create_snapshot(self, file_path: str) -> ConfigSnapshot:
        """Create a configuration snapshot"""
        try:
            # Read file content
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'r') as f:
                    content = yaml.safe_load(f) or {}
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    content = json.load(f)
            else:
                # For other file types, read as text
                with open(file_path, 'r') as f:
                    content = f.read()
            
            # Calculate content hash
            content_str = json.dumps(content, sort_keys=True)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            # Create snapshot
            snapshot = ConfigSnapshot(
                snapshot_id=f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                file_path=file_path,
                content_hash=content_hash,
                content=content,
                metadata={
                    'file_size': os.path.getsize(file_path),
                    'file_modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'file_created': datetime.fromtimestamp(os.path.getctime(file_path))
                }
            )
            
            # Store snapshot
            self.config_history.append(snapshot)
            
            # Keep only last 1000 snapshots
            if len(self.config_history) > 1000:
                self.config_history = self.config_history[-1000:]
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating snapshot for {file_path}: {e}")
            return None
    
    def detect_changes(self, file_path: str) -> List[ConfigChange]:
        """Detect configuration changes in a file"""
        try:
            # Get latest snapshot for this file
            latest_snapshot = None
            for snapshot in reversed(self.config_history):
                if snapshot.file_path == file_path:
                    latest_snapshot = snapshot
                    break
            
            if not latest_snapshot:
                # No previous snapshot, treat as new file
                return self._detect_new_file_changes(file_path)
            
            # Create current snapshot
            current_snapshot = self._create_snapshot(file_path)
            if not current_snapshot:
                return []
            
            # Compare snapshots
            changes = self._compare_snapshots(latest_snapshot, current_snapshot)
            
            # Store changes
            for change in changes:
                self.change_history.append(change)
                
                # Add to approval queue if required
                if change.approval_required:
                    self.approval_queue.append(change)
            
            # Keep only last 5000 changes
            if len(self.change_history) > 5000:
                self.change_history = self.change_history[-5000:]
            
            return changes
            
        except Exception as e:
            logger.error(f"Error detecting changes in {file_path}: {e}")
            return []
    
    def _detect_new_file_changes(self, file_path: str) -> List[ConfigChange]:
        """Detect changes for a new file"""
        try:
            current_snapshot = self._create_snapshot(file_path)
            if not current_snapshot:
                return []
            
            # Create change record for new file
            change = ConfigChange(
                change_id=f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                file_path=file_path,
                change_type=ChangeType.ADDED,
                severity=ChangeSeverity.MEDIUM,
                category=self._categorize_file(file_path),
                old_value=None,
                new_value=current_snapshot.content,
                field_path="*",
                description=f"New configuration file added: {file_path}",
                author="system",
                approval_required=self._requires_approval(file_path, ChangeType.ADDED)
            )
            
            return [change]
            
        except Exception as e:
            logger.error(f"Error detecting new file changes: {e}")
            return []
    
    def _compare_snapshots(self, old_snapshot: ConfigSnapshot, 
                          new_snapshot: ConfigSnapshot) -> List[ConfigChange]:
        """Compare two configuration snapshots"""
        try:
            changes = []
            
            # Compare content
            if isinstance(old_snapshot.content, dict) and isinstance(new_snapshot.content, dict):
                changes.extend(self._compare_dicts(
                    old_snapshot.content, 
                    new_snapshot.content, 
                    old_snapshot.file_path,
                    new_snapshot.file_path
                ))
            else:
                # Compare as strings
                changes.extend(self._compare_strings(
                    str(old_snapshot.content),
                    str(new_snapshot.content),
                    old_snapshot.file_path,
                    new_snapshot.file_path
                ))
            
            return changes
            
        except Exception as e:
            logger.error(f"Error comparing snapshots: {e}")
            return []
    
    def _compare_dicts(self, old_dict: Dict, new_dict: Dict, 
                      old_file_path: str, new_file_path: str) -> List[ConfigChange]:
        """Compare two dictionaries recursively"""
        try:
            changes = []
            
            # Find added keys
            for key in new_dict:
                if key not in old_dict:
                    change = ConfigChange(
                        change_id=f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(changes)}",
                        timestamp=datetime.now(),
                        file_path=new_file_path,
                        change_type=ChangeType.ADDED,
                        severity=self._assess_severity(key, None, new_dict[key]),
                        category=self._categorize_field(key),
                        old_value=None,
                        new_value=new_dict[key],
                        field_path=key,
                        description=f"Added field: {key}",
                        author="system",
                        approval_required=self._requires_approval(new_file_path, ChangeType.ADDED, key)
                    )
                    changes.append(change)
            
            # Find removed keys
            for key in old_dict:
                if key not in new_dict:
                    change = ConfigChange(
                        change_id=f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(changes)}",
                        timestamp=datetime.now(),
                        file_path=new_file_path,
                        change_type=ChangeType.REMOVED,
                        severity=self._assess_severity(key, old_dict[key], None),
                        category=self._categorize_field(key),
                        old_value=old_dict[key],
                        new_value=None,
                        field_path=key,
                        description=f"Removed field: {key}",
                        author="system",
                        approval_required=self._requires_approval(new_file_path, ChangeType.REMOVED, key)
                    )
                    changes.append(change)
            
            # Find modified keys
            for key in old_dict:
                if key in new_dict and old_dict[key] != new_dict[key]:
                    change = ConfigChange(
                        change_id=f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(changes)}",
                        timestamp=datetime.now(),
                        file_path=new_file_path,
                        change_type=ChangeType.MODIFIED,
                        severity=self._assess_severity(key, old_dict[key], new_dict[key]),
                        category=self._categorize_field(key),
                        old_value=old_dict[key],
                        new_value=new_dict[key],
                        field_path=key,
                        description=f"Modified field: {key}",
                        author="system",
                        approval_required=self._requires_approval(new_file_path, ChangeType.MODIFIED, key)
                    )
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"Error comparing dictionaries: {e}")
            return []
    
    def _compare_strings(self, old_str: str, new_str: str, 
                        old_file_path: str, new_file_path: str) -> List[ConfigChange]:
        """Compare two strings and create change records"""
        try:
            changes = []
            
            # Use difflib to find differences
            differ = difflib.unified_diff(
                old_str.splitlines(keepends=True),
                new_str.splitlines(keepends=True),
                fromfile=old_file_path,
                tofile=new_file_path
            )
            
            diff_lines = list(differ)
            if diff_lines:
                change = ConfigChange(
                    change_id=f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    file_path=new_file_path,
                    change_type=ChangeType.MODIFIED,
                    severity=ChangeSeverity.MEDIUM,
                    category=self._categorize_file(new_file_path),
                    old_value=old_str,
                    new_value=new_str,
                    field_path="*",
                    description=f"File content modified: {len(diff_lines)} lines changed",
                    author="system",
                    approval_required=self._requires_approval(new_file_path, ChangeType.MODIFIED)
                )
                changes.append(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"Error comparing strings: {e}")
            return []
    
    def _assess_severity(self, field_path: str, old_value: Any, new_value: Any) -> ChangeSeverity:
        """Assess the severity of a configuration change"""
        try:
            # Critical fields
            critical_fields = [
                'api_key', 'password', 'secret', 'token',
                'max_position_size', 'risk_limit', 'kill_switch',
                'daily_loss_limit', 'var_limit'
            ]
            
            # High severity fields
            high_severity_fields = [
                'commission_rate', 'slippage', 'execution_method',
                'model_parameters', 'trading_hours', 'market_data_source'
            ]
            
            # Check field path
            field_lower = field_path.lower()
            
            for critical_field in critical_fields:
                if critical_field in field_lower:
                    return ChangeSeverity.CRITICAL
            
            for high_field in high_severity_fields:
                if high_field in field_lower:
                    return ChangeSeverity.HIGH
            
            # Check value changes for numeric fields
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if old_value != 0:
                    change_ratio = abs(new_value - old_value) / abs(old_value)
                    if change_ratio > 0.5:  # 50% change
                        return ChangeSeverity.HIGH
                    elif change_ratio > 0.2:  # 20% change
                        return ChangeSeverity.MEDIUM
            
            return ChangeSeverity.LOW
            
        except Exception as e:
            logger.error(f"Error assessing severity: {e}")
            return ChangeSeverity.MEDIUM
    
    def _categorize_file(self, file_path: str) -> ConfigCategory:
        """Categorize a configuration file"""
        try:
            file_lower = file_path.lower()
            
            if 'risk' in file_lower or 'capital' in file_lower:
                return ConfigCategory.RISK_MANAGEMENT
            elif 'trading' in file_lower or 'strategy' in file_lower:
                return ConfigCategory.TRADING_PARAMETERS
            elif 'ai' in file_lower or 'model' in file_lower:
                return ConfigCategory.AI_MODELS
            elif 'data' in file_lower or 'pipeline' in file_lower:
                return ConfigCategory.DATA_SOURCES
            elif 'api' in file_lower or 'key' in file_lower:
                return ConfigCategory.API_KEYS
            elif 'execution' in file_lower or 'order' in file_lower:
                return ConfigCategory.EXECUTION_SETTINGS
            elif 'monitor' in file_lower or 'log' in file_lower:
                return ConfigCategory.MONITORING
            elif 'compliance' in file_lower or 'audit' in file_lower:
                return ConfigCategory.COMPLIANCE
            else:
                return ConfigCategory.TRADING_PARAMETERS
                
        except Exception as e:
            logger.error(f"Error categorizing file: {e}")
            return ConfigCategory.TRADING_PARAMETERS
    
    def _categorize_field(self, field_path: str) -> ConfigCategory:
        """Categorize a configuration field"""
        try:
            field_lower = field_path.lower()
            
            if any(risk_term in field_lower for risk_term in ['risk', 'var', 'beta', 'drawdown', 'limit']):
                return ConfigCategory.RISK_MANAGEMENT
            elif any(trading_term in field_lower for trading_term in ['trading', 'strategy', 'signal', 'position']):
                return ConfigCategory.TRADING_PARAMETERS
            elif any(ai_term in field_lower for ai_term in ['ai', 'model', 'neural', 'lstm', 'gru']):
                return ConfigCategory.AI_MODELS
            elif any(data_term in field_lower for data_term in ['data', 'source', 'api', 'feed']):
                return ConfigCategory.DATA_SOURCES
            elif any(api_term in field_lower for api_term in ['key', 'token', 'secret', 'password']):
                return ConfigCategory.API_KEYS
            elif any(exec_term in field_lower for exec_term in ['execution', 'order', 'slippage', 'commission']):
                return ConfigCategory.EXECUTION_SETTINGS
            elif any(monitor_term in field_lower for monitor_term in ['monitor', 'log', 'alert', 'notification']):
                return ConfigCategory.MONITORING
            elif any(comp_term in field_lower for comp_term in ['compliance', 'audit', 'governance']):
                return ConfigCategory.COMPLIANCE
            else:
                return ConfigCategory.TRADING_PARAMETERS
                
        except Exception as e:
            logger.error(f"Error categorizing field: {e}")
            return ConfigCategory.TRADING_PARAMETERS
    
    def _requires_approval(self, file_path: str, change_type: ChangeType, 
                          field_path: str = None) -> bool:
        """Determine if a change requires approval"""
        try:
            category = self._categorize_file(file_path)
            
            # Check if category requires approval
            if category in self.require_approval_categories:
                return True
            
            # Check if category is auto-approved
            if category in self.auto_approve_categories:
                return False
            
            # Check specific field requirements
            if field_path:
                field_category = self._categorize_field(field_path)
                if field_category in self.require_approval_categories:
                    return True
            
            # Default to requiring approval for critical changes
            return change_type in [ChangeType.REMOVED, ChangeType.MODIFIED]
            
        except Exception as e:
            logger.error(f"Error determining approval requirement: {e}")
            return True
    
    def approve_change(self, change_id: str, approver: str) -> bool:
        """Approve a configuration change"""
        try:
            # Find change in approval queue
            change = None
            for c in self.approval_queue:
                if c.change_id == change_id:
                    change = c
                    break
            
            if not change:
                logger.warning(f"Change {change_id} not found in approval queue")
                return False
            
            # Approve change
            change.approved = True
            change.approved_by = approver
            change.approval_timestamp = datetime.now()
            
            # Remove from approval queue
            self.approval_queue = [c for c in self.approval_queue if c.change_id != change_id]
            
            logger.info(f"Change {change_id} approved by {approver}")
            return True
            
        except Exception as e:
            logger.error(f"Error approving change: {e}")
            return False
    
    def reject_change(self, change_id: str, reason: str) -> bool:
        """Reject a configuration change"""
        try:
            # Find change in approval queue
            change = None
            for c in self.approval_queue:
                if c.change_id == change_id:
                    change = c
                    break
            
            if not change:
                logger.warning(f"Change {change_id} not found in approval queue")
                return False
            
            # Reject change
            change.approved = False
            change.approved_by = "system"
            change.approval_timestamp = datetime.now()
            change.description += f" [REJECTED: {reason}]"
            
            # Remove from approval queue
            self.approval_queue = [c for c in self.approval_queue if c.change_id != change_id]
            
            logger.info(f"Change {change_id} rejected: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error rejecting change: {e}")
            return False
    
    def get_pending_approvals(self) -> List[ConfigChange]:
        """Get list of pending approvals"""
        return self.approval_queue.copy()
    
    def get_change_history(self, time_period_hours: int = 24) -> List[ConfigChange]:
        """Get configuration change history"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
            
            recent_changes = [
                change for change in self.change_history
                if change.timestamp >= cutoff_time
            ]
            
            return recent_changes
            
        except Exception as e:
            logger.error(f"Error getting change history: {e}")
            return []
    
    def get_audit_report(self, time_period_hours: int = 24) -> Dict:
        """Generate configuration audit report"""
        try:
            recent_changes = self.get_change_history(time_period_hours)
            
            # Calculate statistics
            total_changes = len(recent_changes)
            severity_counts = {}
            category_counts = {}
            change_type_counts = {}
            approval_counts = {'approved': 0, 'pending': 0, 'rejected': 0}
            
            for change in recent_changes:
                # Severity counts
                severity = change.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Category counts
                category = change.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Change type counts
                change_type = change.change_type.value
                change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
                
                # Approval counts
                if change.approved:
                    approval_counts['approved'] += 1
                elif change.approval_required and not change.approved:
                    approval_counts['pending'] += 1
                else:
                    approval_counts['rejected'] += 1
            
            return {
                'report_period_hours': time_period_hours,
                'total_changes': total_changes,
                'severity_breakdown': severity_counts,
                'category_breakdown': category_counts,
                'change_type_breakdown': change_type_counts,
                'approval_status': approval_counts,
                'pending_approvals': len(self.approval_queue),
                'generated_at': datetime.now(),
                'recent_changes': recent_changes[-10:]  # Last 10 changes
            }
            
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            return {}
    
    def export_audit_data(self, filepath: str, time_period_hours: int = 24):
        """Export audit data to file"""
        try:
            audit_report = self.get_audit_report(time_period_hours)
            
            with open(filepath, 'w') as f:
                json.dump(audit_report, f, indent=2, default=str)
            
            logger.info(f"Exported audit data to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting audit data: {e}")
    
    def get_compliance_status(self) -> Dict:
        """Get current compliance status"""
        try:
            pending_approvals = len(self.approval_queue)
            recent_critical_changes = len([
                c for c in self.change_history[-100:]  # Last 100 changes
                if c.severity == ChangeSeverity.CRITICAL and not c.approved
            ])
            
            compliance_score = 100.0
            if pending_approvals > 0:
                compliance_score -= min(pending_approvals * 10, 50)  # Max 50 point deduction
            if recent_critical_changes > 0:
                compliance_score -= min(recent_critical_changes * 20, 30)  # Max 30 point deduction
            
            return {
                'compliance_score': max(compliance_score, 0),
                'pending_approvals': pending_approvals,
                'unapproved_critical_changes': recent_critical_changes,
                'status': 'compliant' if compliance_score >= 80 else 'non_compliant',
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting compliance status: {e}")
            return {'compliance_score': 0, 'status': 'error'}
