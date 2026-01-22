"""
Auditor Sandbox for GPT-5 Escalation Framework
Provides a safe environment for testing and validating GPT-5 responses
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import copy
import asyncio
import time

logger = logging.getLogger(__name__)

class AuditResult(Enum):
    """Audit result types"""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    PENDING = "pending"

class AuditSeverity(Enum):
    """Audit severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditRule:
    """Audit rule configuration"""
    rule_id: str
    name: str
    description: str
    severity: AuditSeverity
    condition: str  # e.g., "position_size > max_position_size * 0.5"
    action: str  # e.g., "reject", "modify", "warn"
    enabled: bool = True

@dataclass
class AuditFinding:
    """Audit finding record"""
    finding_id: str
    rule_id: str
    severity: AuditSeverity
    description: str
    recommendation: str
    timestamp: datetime
    context: Dict[str, Any]

@dataclass
class AuditReport:
    """Audit report"""
    report_id: str
    timestamp: datetime
    gpt5_response: Dict[str, Any]
    audit_result: AuditResult
    findings: List[AuditFinding]
    approved_actions: List[Dict[str, Any]]
    rejected_actions: List[Dict[str, Any]]
    modified_actions: List[Dict[str, Any]]
    audit_summary: str

class AuditorSandbox:
    """Safe environment for testing and validating GPT-5 responses"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.audit_rules = []
        self.audit_history = []
        self.simulation_environment = {}
        self.risk_limits = {}
        self.position_limits = {}
        
        # Load default audit rules
        self._load_default_audit_rules()
        
        # Initialize risk limits
        self._initialize_risk_limits()
        
        logger.info("Auditor Sandbox initialized")
    
    def _load_default_audit_rules(self):
        """Load default audit rules"""
        try:
            default_rules = [
                # Position size rules
                AuditRule(
                    rule_id="pos_size_001",
                    name="Maximum Position Size",
                    description="Ensure position size doesn't exceed maximum allowed",
                    severity=AuditSeverity.HIGH,
                    condition="position_size > max_position_size",
                    action="reject"
                ),
                AuditRule(
                    rule_id="pos_size_002",
                    name="Large Position Warning",
                    description="Warn about large position sizes",
                    severity=AuditSeverity.MEDIUM,
                    condition="position_size > max_position_size * 0.5",
                    action="warn"
                ),
                
                # Risk rules
                AuditRule(
                    rule_id="risk_001",
                    name="VaR Limit",
                    description="Ensure VaR doesn't exceed limits",
                    severity=AuditSeverity.CRITICAL,
                    condition="portfolio_var > max_var",
                    action="reject"
                ),
                AuditRule(
                    rule_id="risk_002",
                    name="Beta Limit",
                    description="Ensure portfolio beta stays within limits",
                    severity=AuditSeverity.HIGH,
                    condition="portfolio_beta > max_beta",
                    action="reject"
                ),
                
                # Concentration rules
                AuditRule(
                    rule_id="conc_001",
                    name="Single Stock Concentration",
                    description="Limit concentration in single stock",
                    severity=AuditSeverity.HIGH,
                    condition="single_stock_weight > max_single_stock_weight",
                    action="reject"
                ),
                AuditRule(
                    rule_id="conc_002",
                    name="Sector Concentration",
                    description="Limit sector concentration",
                    severity=AuditSeverity.MEDIUM,
                    condition="sector_weight > max_sector_weight",
                    action="warn"
                ),
                
                # Trading rules
                AuditRule(
                    rule_id="trade_001",
                    name="Daily Trading Limit",
                    description="Limit daily trading activity",
                    severity=AuditSeverity.MEDIUM,
                    condition="daily_trades > max_daily_trades",
                    action="warn"
                ),
                AuditRule(
                    rule_id="trade_002",
                    name="Penny Stock Limit",
                    description="Limit penny stock exposure",
                    severity=AuditSeverity.HIGH,
                    condition="penny_stock_weight > max_penny_weight",
                    action="reject"
                ),
                
                # Market condition rules
                AuditRule(
                    rule_id="market_001",
                    name="High Volatility Trading",
                    description="Restrict trading during high volatility",
                    severity=AuditSeverity.MEDIUM,
                    condition="volatility > high_volatility_threshold and action == 'buy'",
                    action="warn"
                ),
                AuditRule(
                    rule_id="market_002",
                    name="Market Hours",
                    description="Ensure trades are within market hours",
                    severity=AuditSeverity.LOW,
                    condition="market_closed == True",
                    action="reject"
                )
            ]
            
            self.audit_rules = default_rules
            logger.info(f"Loaded {len(default_rules)} default audit rules")
            
        except Exception as e:
            logger.error(f"Error loading default audit rules: {e}")
    
    def _initialize_risk_limits(self):
        """Initialize risk limits"""
        try:
            self.risk_limits = {
                'max_var_95': 0.05,  # 5% VaR limit
                'max_var_99': 0.08,  # 8% VaR limit
                'max_beta': 1.5,     # Maximum portfolio beta
                'max_single_stock_weight': 0.10,  # 10% max single stock
                'max_sector_weight': 0.30,        # 30% max sector
                'max_penny_weight': 0.05,         # 5% max penny stocks
                'max_daily_trades': 50,           # 50 max daily trades
                'high_volatility_threshold': 0.03  # 3% volatility threshold
            }
            
            self.position_limits = {
                'max_position_size': 10000,  # $10,000 max position
                'min_position_size': 100,    # $100 min position
                'max_position_count': 20     # 20 max positions
            }
            
            logger.info("Risk limits initialized")
            
        except Exception as e:
            logger.error(f"Error initializing risk limits: {e}")
    
    def add_audit_rule(self, rule: AuditRule):
        """Add a new audit rule"""
        try:
            self.audit_rules.append(rule)
            logger.info(f"Added audit rule: {rule.name}")
            
        except Exception as e:
            logger.error(f"Error adding audit rule: {e}")
    
    def remove_audit_rule(self, rule_id: str):
        """Remove an audit rule by ID"""
        try:
            self.audit_rules = [rule for rule in self.audit_rules if rule.rule_id != rule_id]
            logger.info(f"Removed audit rule: {rule_id}")
            
        except Exception as e:
            logger.error(f"Error removing audit rule: {e}")
    
    def audit_gpt5_response(self, gpt5_response: Dict[str, Any], 
                          system_context: Dict[str, Any]) -> AuditReport:
        """Audit GPT-5 response for safety and compliance"""
        try:
            report_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Extract actions from GPT-5 response
            actions = self._extract_actions(gpt5_response)
            
            # Run audit on each action
            findings = []
            approved_actions = []
            rejected_actions = []
            modified_actions = []
            
            for action in actions:
                action_findings = self._audit_action(action, system_context)
                findings.extend(action_findings)
                
                # Determine action outcome based on findings
                outcome = self._determine_action_outcome(action_findings)
                
                if outcome == "approved":
                    approved_actions.append(action)
                elif outcome == "rejected":
                    rejected_actions.append(action)
                elif outcome == "modified":
                    modified_action = self._modify_action(action, action_findings)
                    modified_actions.append(modified_action)
            
            # Determine overall audit result
            audit_result = self._determine_audit_result(findings)
            
            # Create audit summary
            audit_summary = self._create_audit_summary(
                findings, approved_actions, rejected_actions, modified_actions
            )
            
            # Create audit report
            report = AuditReport(
                report_id=report_id,
                timestamp=datetime.now(),
                gpt5_response=gpt5_response,
                audit_result=audit_result,
                findings=findings,
                approved_actions=approved_actions,
                rejected_actions=rejected_actions,
                modified_actions=modified_actions,
                audit_summary=audit_summary
            )
            
            # Store audit history
            self.audit_history.append(report)
            
            logger.info(f"Audit completed: {audit_result.value} - {len(findings)} findings")
            
            return report
            
        except Exception as e:
            logger.error(f"Error auditing GPT-5 response: {e}")
            return None
    
    def _extract_actions(self, gpt5_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trading actions from GPT-5 response"""
        try:
            actions = []
            
            # Look for trading actions in the response
            if 'actions' in gpt5_response:
                actions = gpt5_response['actions']
            elif 'trades' in gpt5_response:
                actions = gpt5_response['trades']
            elif 'recommendations' in gpt5_response:
                actions = gpt5_response['recommendations']
            
            # Ensure actions is a list
            if not isinstance(actions, list):
                actions = [actions] if actions else []
            
            return actions
            
        except Exception as e:
            logger.error(f"Error extracting actions: {e}")
            return []
    
    def _audit_action(self, action: Dict[str, Any], 
                     system_context: Dict[str, Any]) -> List[AuditFinding]:
        """Audit a single action against all rules"""
        try:
            findings = []
            
            # Combine action and system context for rule evaluation
            audit_context = {**action, **system_context, **self.risk_limits, **self.position_limits}
            
            for rule in self.audit_rules:
                if not rule.enabled:
                    continue
                
                # Evaluate rule condition
                if self._evaluate_rule_condition(rule.condition, audit_context):
                    # Create finding
                    finding = AuditFinding(
                        finding_id=f"finding_{len(findings) + 1}",
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        description=rule.description,
                        recommendation=self._get_rule_recommendation(rule, action),
                        timestamp=datetime.now(),
                        context=audit_context.copy()
                    )
                    
                    findings.append(finding)
            
            return findings
            
        except Exception as e:
            logger.error(f"Error auditing action: {e}")
            return []
    
    def _evaluate_rule_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate audit rule condition"""
        try:
            # Replace context variables in condition
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    condition = condition.replace(key, str(value))
                elif isinstance(value, bool):
                    condition = condition.replace(key, str(value))
                elif isinstance(value, str):
                    condition = condition.replace(f'"{key}"', f'"{value}"')
            
            # Evaluate the condition
            try:
                result = eval(condition)
                return bool(result)
            except:
                logger.warning(f"Could not evaluate rule condition: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return False
    
    def _get_rule_recommendation(self, rule: AuditRule, action: Dict[str, Any]) -> str:
        """Get recommendation for rule violation"""
        try:
            if rule.action == "reject":
                return f"REJECT: {rule.description}"
            elif rule.action == "warn":
                return f"WARNING: {rule.description}"
            elif rule.action == "modify":
                return f"MODIFY: {rule.description} - Consider reducing position size or risk exposure"
            else:
                return f"REVIEW: {rule.description}"
                
        except Exception as e:
            logger.error(f"Error getting rule recommendation: {e}")
            return "Review required"
    
    def _determine_action_outcome(self, findings: List[AuditFinding]) -> str:
        """Determine action outcome based on findings"""
        try:
            if not findings:
                return "approved"
            
            # Check for critical or high severity findings
            critical_findings = [f for f in findings if f.severity == AuditSeverity.CRITICAL]
            high_findings = [f for f in findings if f.severity == AuditSeverity.HIGH]
            
            if critical_findings:
                return "rejected"
            elif high_findings:
                return "rejected"
            else:
                # Medium and low severity findings can be modified
                return "modified"
                
        except Exception as e:
            logger.error(f"Error determining action outcome: {e}")
            return "rejected"
    
    def _modify_action(self, action: Dict[str, Any], 
                      findings: List[AuditFinding]) -> Dict[str, Any]:
        """Modify action based on audit findings"""
        try:
            modified_action = copy.deepcopy(action)
            
            for finding in findings:
                if finding.rule_id == "pos_size_002":  # Large position warning
                    if 'quantity' in modified_action:
                        modified_action['quantity'] = int(modified_action['quantity'] * 0.5)
                elif finding.rule_id == "conc_002":  # Sector concentration
                    if 'quantity' in modified_action:
                        modified_action['quantity'] = int(modified_action['quantity'] * 0.7)
                elif finding.rule_id == "market_001":  # High volatility
                    if 'quantity' in modified_action:
                        modified_action['quantity'] = int(modified_action['quantity'] * 0.3)
            
            # Add modification note
            modified_action['_audit_modified'] = True
            modified_action['_modification_reason'] = "Modified by audit sandbox for risk compliance"
            
            return modified_action
            
        except Exception as e:
            logger.error(f"Error modifying action: {e}")
            return action
    
    def _determine_audit_result(self, findings: List[AuditFinding]) -> AuditResult:
        """Determine overall audit result"""
        try:
            if not findings:
                return AuditResult.APPROVED
            
            # Check for critical findings
            critical_findings = [f for f in findings if f.severity == AuditSeverity.CRITICAL]
            if critical_findings:
                return AuditResult.REJECTED
            
            # Check for high severity findings
            high_findings = [f for f in findings if f.severity == AuditSeverity.HIGH]
            if high_findings:
                return AuditResult.REJECTED
            
            # Check for medium severity findings
            medium_findings = [f for f in findings if f.severity == AuditSeverity.MEDIUM]
            if medium_findings:
                return AuditResult.MODIFIED
            
            # Only low severity findings
            return AuditResult.APPROVED
            
        except Exception as e:
            logger.error(f"Error determining audit result: {e}")
            return AuditResult.REJECTED
    
    def _create_audit_summary(self, findings: List[AuditFinding],
                            approved_actions: List[Dict],
                            rejected_actions: List[Dict],
                            modified_actions: List[Dict]) -> str:
        """Create audit summary"""
        try:
            summary_parts = []
            
            summary_parts.append(f"Audit Summary:")
            summary_parts.append(f"- Total Actions: {len(approved_actions) + len(rejected_actions) + len(modified_actions)}")
            summary_parts.append(f"- Approved: {len(approved_actions)}")
            summary_parts.append(f"- Rejected: {len(rejected_actions)}")
            summary_parts.append(f"- Modified: {len(modified_actions)}")
            summary_parts.append(f"- Total Findings: {len(findings)}")
            
            if findings:
                severity_counts = {}
                for finding in findings:
                    severity = finding.severity.value
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                summary_parts.append(f"- Findings by Severity: {severity_counts}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating audit summary: {e}")
            return "Audit summary unavailable"
    
    def simulate_actions(self, actions: List[Dict[str, Any]], 
                        system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate actions in sandbox environment"""
        try:
            simulation_results = {
                'simulation_id': f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now(),
                'actions_simulated': len(actions),
                'results': []
            }
            
            for i, action in enumerate(actions):
                result = self._simulate_single_action(action, system_context)
                simulation_results['results'].append({
                    'action_index': i,
                    'action': action,
                    'result': result
                })
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Error simulating actions: {e}")
            return {}
    
    def _simulate_single_action(self, action: Dict[str, Any], 
                              system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a single action"""
        try:
            # Simple simulation - in production this would be more sophisticated
            result = {
                'simulated': True,
                'estimated_impact': {
                    'price_impact': 0.001,  # 0.1% price impact
                    'slippage': 0.0005,     # 0.05% slippage
                    'commission': 1.0       # $1 commission
                },
                'risk_metrics': {
                    'var_impact': 0.001,    # 0.1% VaR impact
                    'beta_impact': 0.01     # 0.01 beta impact
                },
                'execution_time': 0.1       # 100ms execution time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating single action: {e}")
            return {'simulated': False, 'error': str(e)}
    
    def get_audit_statistics(self) -> Dict:
        """Get audit statistics"""
        try:
            if not self.audit_history:
                return {}
            
            # Calculate statistics
            total_audits = len(self.audit_history)
            result_counts = {}
            severity_counts = {}
            
            for report in self.audit_history:
                result = report.audit_result.value
                result_counts[result] = result_counts.get(result, 0) + 1
                
                for finding in report.findings:
                    severity = finding.severity.value
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Calculate approval rate
            approved_count = result_counts.get('approved', 0)
            approval_rate = approved_count / total_audits if total_audits > 0 else 0
            
            return {
                'total_audits': total_audits,
                'result_counts': result_counts,
                'severity_counts': severity_counts,
                'approval_rate': approval_rate,
                'active_rules': len([r for r in self.audit_rules if r.enabled]),
                'total_rules': len(self.audit_rules)
            }
            
        except Exception as e:
            logger.error(f"Error getting audit statistics: {e}")
            return {}
    
    def export_audit_rules(self, filepath: str):
        """Export audit rules to file"""
        try:
            rules_data = []
            for rule in self.audit_rules:
                rules_data.append({
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'description': rule.description,
                    'severity': rule.severity.value,
                    'condition': rule.condition,
                    'action': rule.action,
                    'enabled': rule.enabled
                })
            
            with open(filepath, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            logger.info(f"Exported {len(rules_data)} audit rules to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting audit rules: {e}")
    
    def import_audit_rules(self, filepath: str):
        """Import audit rules from file"""
        try:
            with open(filepath, 'r') as f:
                rules_data = json.load(f)
            
            imported_rules = []
            for rule_data in rules_data:
                rule = AuditRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    description=rule_data['description'],
                    severity=AuditSeverity(rule_data['severity']),
                    condition=rule_data['condition'],
                    action=rule_data['action'],
                    enabled=rule_data.get('enabled', True)
                )
                imported_rules.append(rule)
            
            self.audit_rules = imported_rules
            logger.info(f"Imported {len(imported_rules)} audit rules from {filepath}")
            
        except Exception as e:
            logger.error(f"Error importing audit rules: {e}")
