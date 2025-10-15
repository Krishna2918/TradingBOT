"""
Regulatory Reporting Automation System

This module implements comprehensive regulatory reporting automation for
SEC, FINRA, and other regulatory bodies with automated report generation,
submission, and compliance tracking.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import sqlite3
from pathlib import Path
import csv
import xml.etree.ElementTree as ET
from jinja2 import Template
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of regulatory reports."""
    # SEC Reports
    FORM_13F = "FORM_13F"
    FORM_13D = "FORM_13D"
    FORM_13G = "FORM_13G"
    FORM_4 = "FORM_4"
    FORM_8K = "FORM_8K"
    FORM_10K = "FORM_10K"
    FORM_10Q = "FORM_10Q"
    
    # FINRA Reports
    TRACE_REPORT = "TRACE_REPORT"
    OATS_REPORT = "OATS_REPORT"
    CAT_REPORT = "CAT_REPORT"
    
    # CFTC Reports
    CFTC_FORM_40 = "CFTC_FORM_40"
    CFTC_FORM_102 = "CFTC_FORM_102"
    
    # Internal Reports
    COMPLIANCE_REPORT = "COMPLIANCE_REPORT"
    RISK_REPORT = "RISK_REPORT"
    AUDIT_REPORT = "AUDIT_REPORT"
    TRADE_REPORT = "TRADE_REPORT"

class ReportStatus(Enum):
    """Report status."""
    DRAFT = "DRAFT"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    AMENDED = "AMENDED"

class ReportFormat(Enum):
    """Report formats."""
    XML = "XML"
    CSV = "CSV"
    JSON = "JSON"
    PDF = "PDF"
    XBRL = "XBRL"

@dataclass
class ReportDefinition:
    """Regulatory report definition."""
    report_id: str
    report_type: ReportType
    name: str
    description: str
    regulatory_body: str
    frequency: str  # DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUALLY, ON_DEMAND
    due_date_offset: int  # Days before period end
    format: ReportFormat
    template_path: str
    validation_rules: List[Dict[str, Any]]
    submission_requirements: Dict[str, Any]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ReportInstance:
    """Report instance for a specific period."""
    instance_id: str
    report_id: str
    report_type: ReportType
    period_start: datetime
    period_end: datetime
    due_date: datetime
    status: ReportStatus
    generated_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    file_path: Optional[str] = None
    submission_id: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    approval_notes: Optional[str] = None
    submission_notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

class ReportGenerator:
    """
    Regulatory report generator with template engine and validation.
    
    Features:
    - Template-based report generation
    - Data validation and quality checks
    - Multiple output formats (XML, CSV, JSON, PDF)
    - Automated submission workflows
    - Compliance tracking and monitoring
    """
    
    def __init__(self, db_path: str = "data/regulatory_reporting.db"):
        """
        Initialize report generator.
        
        Args:
            db_path: Path to regulatory reporting database
        """
        self.db_path = db_path
        self.report_definitions: Dict[str, ReportDefinition] = {}
        self.report_instances: Dict[str, ReportInstance] = {}
        
        # Initialize database
        self._init_database()
        
        # Load default report definitions
        self._load_default_reports()
        
        logger.info("Regulatory Report Generator initialized")
    
    def _init_database(self) -> None:
        """Initialize regulatory reporting database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create report definitions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS report_definitions (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                regulatory_body TEXT,
                frequency TEXT,
                due_date_offset INTEGER,
                format TEXT,
                template_path TEXT,
                validation_rules TEXT,
                submission_requirements TEXT,
                is_active INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Create report instances table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS report_instances (
                instance_id TEXT PRIMARY KEY,
                report_id TEXT,
                report_type TEXT,
                period_start TEXT,
                period_end TEXT,
                due_date TEXT,
                status TEXT,
                generated_at TEXT,
                submitted_at TEXT,
                file_path TEXT,
                submission_id TEXT,
                validation_errors TEXT,
                approval_notes TEXT,
                submission_notes TEXT,
                created_at TEXT,
                FOREIGN KEY (report_id) REFERENCES report_definitions (report_id)
            )
        """)
        
        # Create report data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS report_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT,
                data_type TEXT,
                data_key TEXT,
                data_value TEXT,
                created_at TEXT,
                FOREIGN KEY (instance_id) REFERENCES report_instances (instance_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_reports(self) -> None:
        """Load default regulatory report definitions."""
        default_reports = [
            ReportDefinition(
                report_id="REPORT_001",
                report_type=ReportType.FORM_13F,
                name="Form 13F - Quarterly Holdings Report",
                description="Quarterly report of institutional investment managers",
                regulatory_body="SEC",
                frequency="QUARTERLY",
                due_date_offset=45,  # 45 days after quarter end
                format=ReportFormat.XML,
                template_path="templates/form_13f.xml",
                validation_rules=[
                    {"field": "total_value", "type": "numeric", "min": 0},
                    {"field": "holdings_count", "type": "integer", "min": 0},
                    {"field": "filing_date", "type": "date", "required": True}
                ],
                submission_requirements={
                    "edgar_system": True,
                    "xbrl_required": True,
                    "signature_required": True
                }
            ),
            ReportDefinition(
                report_id="REPORT_002",
                report_type=ReportType.TRACE_REPORT,
                name="TRACE - Trade Reporting and Compliance Engine",
                description="Daily trade reporting for corporate bonds",
                regulatory_body="FINRA",
                frequency="DAILY",
                due_date_offset=0,  # Same day
                format=ReportFormat.XML,
                template_path="templates/trace_report.xml",
                validation_rules=[
                    {"field": "trade_count", "type": "integer", "min": 0},
                    {"field": "total_volume", "type": "numeric", "min": 0},
                    {"field": "reporting_date", "type": "date", "required": True}
                ],
                submission_requirements={
                    "finra_system": True,
                    "real_time": True,
                    "encryption_required": True
                }
            ),
            ReportDefinition(
                report_id="REPORT_003",
                report_type=ReportType.COMPLIANCE_REPORT,
                name="Monthly Compliance Report",
                description="Internal compliance monitoring report",
                regulatory_body="INTERNAL",
                frequency="MONTHLY",
                due_date_offset=5,  # 5 days after month end
                format=ReportFormat.PDF,
                template_path="templates/compliance_report.html",
                validation_rules=[
                    {"field": "violation_count", "type": "integer", "min": 0},
                    {"field": "compliance_score", "type": "numeric", "min": 0, "max": 100},
                    {"field": "reporting_period", "type": "date", "required": True}
                ],
                submission_requirements={
                    "internal_review": True,
                    "approval_required": True
                }
            ),
            ReportDefinition(
                report_id="REPORT_004",
                report_type=ReportType.RISK_REPORT,
                name="Daily Risk Report",
                description="Daily risk monitoring and limit compliance report",
                regulatory_body="INTERNAL",
                frequency="DAILY",
                due_date_offset=0,  # Same day
                format=ReportFormat.JSON,
                template_path="templates/risk_report.json",
                validation_rules=[
                    {"field": "portfolio_value", "type": "numeric", "min": 0},
                    {"field": "var_95", "type": "numeric", "min": 0},
                    {"field": "max_drawdown", "type": "numeric", "max": 0}
                ],
                submission_requirements={
                    "real_time": True,
                    "alert_on_violations": True
                }
            )
        ]
        
        for report in default_reports:
            self.add_report_definition(report)
    
    def add_report_definition(self, report_def: ReportDefinition) -> None:
        """
        Add a new report definition.
        
        Args:
            report_def: Report definition
        """
        self.report_definitions[report_def.report_id] = report_def
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO report_definitions 
            (report_id, report_type, name, description, regulatory_body, frequency,
             due_date_offset, format, template_path, validation_rules, submission_requirements,
             is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_def.report_id, report_def.report_type.value, report_def.name,
            report_def.description, report_def.regulatory_body, report_def.frequency,
            report_def.due_date_offset, report_def.format.value, report_def.template_path,
            json.dumps(report_def.validation_rules), json.dumps(report_def.submission_requirements),
            report_def.is_active, report_def.created_at.isoformat(), report_def.updated_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added report definition: {report_def.report_id} - {report_def.name}")
    
    def create_report_instance(self, report_id: str, period_start: datetime, 
                             period_end: datetime) -> str:
        """
        Create a new report instance.
        
        Args:
            report_id: Report definition ID
            period_start: Report period start date
            period_end: Report period end date
            
        Returns:
            Report instance ID
        """
        if report_id not in self.report_definitions:
            raise ValueError(f"Report definition not found: {report_id}")
        
        report_def = self.report_definitions[report_id]
        
        # Calculate due date
        due_date = period_end + timedelta(days=report_def.due_date_offset)
        
        instance_id = f"{report_id}_{period_start.strftime('%Y%m%d')}_{period_end.strftime('%Y%m%d')}"
        
        instance = ReportInstance(
            instance_id=instance_id,
            report_id=report_id,
            report_type=report_def.report_type,
            period_start=period_start,
            period_end=period_end,
            due_date=due_date,
            status=ReportStatus.DRAFT
        )
        
        self.report_instances[instance_id] = instance
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO report_instances 
            (instance_id, report_id, report_type, period_start, period_end, due_date,
             status, generated_at, submitted_at, file_path, submission_id,
             validation_errors, approval_notes, submission_notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            instance.instance_id, instance.report_id, instance.report_type.value,
            instance.period_start.isoformat(), instance.period_end.isoformat(),
            instance.due_date.isoformat(), instance.status.value,
            instance.generated_at.isoformat() if instance.generated_at else None,
            instance.submitted_at.isoformat() if instance.submitted_at else None,
            instance.file_path, instance.submission_id,
            json.dumps(instance.validation_errors), instance.approval_notes,
            instance.submission_notes, instance.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created report instance: {instance_id}")
        return instance_id
    
    def generate_report(self, instance_id: str, data: Dict[str, Any]) -> str:
        """
        Generate a regulatory report.
        
        Args:
            instance_id: Report instance ID
            data: Report data
            
        Returns:
            Generated file path
        """
        if instance_id not in self.report_instances:
            raise ValueError(f"Report instance not found: {instance_id}")
        
        instance = self.report_instances[instance_id]
        report_def = self.report_definitions[instance.report_id]
        
        # Validate data
        validation_errors = self._validate_report_data(data, report_def.validation_rules)
        if validation_errors:
            instance.validation_errors = validation_errors
            instance.status = ReportStatus.REJECTED
            self._update_report_instance(instance)
            raise ValueError(f"Report validation failed: {validation_errors}")
        
        # Generate report based on format
        file_path = None
        if report_def.format == ReportFormat.XML:
            file_path = self._generate_xml_report(instance, data)
        elif report_def.format == ReportFormat.CSV:
            file_path = self._generate_csv_report(instance, data)
        elif report_def.format == ReportFormat.JSON:
            file_path = self._generate_json_report(instance, data)
        elif report_def.format == ReportFormat.PDF:
            file_path = self._generate_pdf_report(instance, data)
        elif report_def.format == ReportFormat.XBRL:
            file_path = self._generate_xbrl_report(instance, data)
        
        # Update instance
        instance.file_path = file_path
        instance.generated_at = datetime.now()
        instance.status = ReportStatus.PENDING_REVIEW
        self._update_report_instance(instance)
        
        # Store report data
        self._store_report_data(instance_id, data)
        
        logger.info(f"Generated report: {file_path}")
        return file_path
    
    def _validate_report_data(self, data: Dict[str, Any], validation_rules: List[Dict[str, Any]]) -> List[str]:
        """Validate report data against validation rules."""
        errors = []
        
        for rule in validation_rules:
            field = rule.get('field')
            field_type = rule.get('type')
            required = rule.get('required', False)
            
            if field not in data:
                if required:
                    errors.append(f"Required field missing: {field}")
                continue
            
            value = data[field]
            
            # Type validation
            if field_type == 'numeric':
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(f"Field {field} must be numeric")
                    continue
                
                # Range validation
                if 'min' in rule and float(value) < rule['min']:
                    errors.append(f"Field {field} must be >= {rule['min']}")
                if 'max' in rule and float(value) > rule['max']:
                    errors.append(f"Field {field} must be <= {rule['max']}")
            
            elif field_type == 'integer':
                try:
                    int(value)
                except (ValueError, TypeError):
                    errors.append(f"Field {field} must be an integer")
                    continue
                
                # Range validation
                if 'min' in rule and int(value) < rule['min']:
                    errors.append(f"Field {field} must be >= {rule['min']}")
                if 'max' in rule and int(value) > rule['max']:
                    errors.append(f"Field {field} must be <= {rule['max']}")
            
            elif field_type == 'date':
                try:
                    datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    errors.append(f"Field {field} must be a valid date")
            
            elif field_type == 'string':
                if 'min_length' in rule and len(str(value)) < rule['min_length']:
                    errors.append(f"Field {field} must be at least {rule['min_length']} characters")
                if 'max_length' in rule and len(str(value)) > rule['max_length']:
                    errors.append(f"Field {field} must be at most {rule['max_length']} characters")
        
        return errors
    
    def _generate_xml_report(self, instance: ReportInstance, data: Dict[str, Any]) -> str:
        """Generate XML report."""
        # Create XML structure
        root = ET.Element("Report")
        root.set("instance_id", instance.instance_id)
        root.set("report_type", instance.report_type.value)
        root.set("period_start", instance.period_start.isoformat())
        root.set("period_end", instance.period_end.isoformat())
        root.set("generated_at", datetime.now().isoformat())
        
        # Add data elements
        for key, value in data.items():
            element = ET.SubElement(root, key)
            element.text = str(value)
        
        # Write to file
        file_path = f"reports/{instance.instance_id}.xml"
        Path("reports").mkdir(exist_ok=True)
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
        
        return file_path
    
    def _generate_csv_report(self, instance: ReportInstance, data: Dict[str, Any]) -> str:
        """Generate CSV report."""
        file_path = f"reports/{instance.instance_id}.csv"
        Path("reports").mkdir(exist_ok=True)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Field', 'Value'])
            
            # Write data
            for key, value in data.items():
                writer.writerow([key, value])
        
        return file_path
    
    def _generate_json_report(self, instance: ReportInstance, data: Dict[str, Any]) -> str:
        """Generate JSON report."""
        file_path = f"reports/{instance.instance_id}.json"
        Path("reports").mkdir(exist_ok=True)
        
        report_data = {
            'instance_id': instance.instance_id,
            'report_type': instance.report_type.value,
            'period_start': instance.period_start.isoformat(),
            'period_end': instance.period_end.isoformat(),
            'generated_at': datetime.now().isoformat(),
            'data': data
        }
        
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(report_data, jsonfile, indent=2, ensure_ascii=False)
        
        return file_path
    
    def _generate_pdf_report(self, instance: ReportInstance, data: Dict[str, Any]) -> str:
        """Generate PDF report."""
        # In a real implementation, this would use a PDF library like ReportLab
        # For now, generate a simple text file
        file_path = f"reports/{instance.instance_id}.pdf"
        Path("reports").mkdir(exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as pdffile:
            pdffile.write(f"Report: {instance.instance_id}\n")
            pdffile.write(f"Type: {instance.report_type.value}\n")
            pdffile.write(f"Period: {instance.period_start.date()} to {instance.period_end.date()}\n")
            pdffile.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for key, value in data.items():
                pdffile.write(f"{key}: {value}\n")
        
        return file_path
    
    def _generate_xbrl_report(self, instance: ReportInstance, data: Dict[str, Any]) -> str:
        """Generate XBRL report."""
        # In a real implementation, this would generate proper XBRL
        # For now, generate XML with XBRL-like structure
        file_path = f"reports/{instance.instance_id}.xbrl"
        Path("reports").mkdir(exist_ok=True)
        
        root = ET.Element("xbrl")
        root.set("xmlns", "http://www.xbrl.org/2003/instance")
        
        # Add context
        context = ET.SubElement(root, "context")
        context.set("id", "period")
        
        period = ET.SubElement(context, "period")
        start_date = ET.SubElement(period, "startDate")
        start_date.text = instance.period_start.isoformat()
        end_date = ET.SubElement(period, "endDate")
        end_date.text = instance.period_end.isoformat()
        
        # Add data
        for key, value in data.items():
            element = ET.SubElement(root, key)
            element.set("contextRef", "period")
            element.text = str(value)
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
        
        return file_path
    
    def _store_report_data(self, instance_id: str, data: Dict[str, Any]) -> None:
        """Store report data in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for key, value in data.items():
            cursor.execute("""
                INSERT INTO report_data (instance_id, data_type, data_key, data_value, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (instance_id, "report_data", key, str(value), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def _update_report_instance(self, instance: ReportInstance) -> None:
        """Update report instance in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE report_instances
            SET status = ?, generated_at = ?, submitted_at = ?, file_path = ?,
                submission_id = ?, validation_errors = ?, approval_notes = ?, submission_notes = ?
            WHERE instance_id = ?
        """, (
            instance.status.value,
            instance.generated_at.isoformat() if instance.generated_at else None,
            instance.submitted_at.isoformat() if instance.submitted_at else None,
            instance.file_path, instance.submission_id,
            json.dumps(instance.validation_errors), instance.approval_notes,
            instance.submission_notes, instance.instance_id
        ))
        
        conn.commit()
        conn.close()
    
    def submit_report(self, instance_id: str, submission_notes: str = None) -> str:
        """
        Submit a report to regulatory body.
        
        Args:
            instance_id: Report instance ID
            submission_notes: Optional submission notes
            
        Returns:
            Submission ID
        """
        if instance_id not in self.report_instances:
            raise ValueError(f"Report instance not found: {instance_id}")
        
        instance = self.report_instances[instance_id]
        
        if instance.status != ReportStatus.APPROVED:
            raise ValueError(f"Report must be approved before submission. Current status: {instance.status}")
        
        if not instance.file_path:
            raise ValueError("Report file not found")
        
        # Generate submission ID
        submission_id = f"SUB_{instance_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # In a real implementation, this would submit to the actual regulatory system
        # For now, just update the status
        instance.submission_id = submission_id
        instance.submitted_at = datetime.now()
        instance.status = ReportStatus.SUBMITTED
        instance.submission_notes = submission_notes
        
        self._update_report_instance(instance)
        
        logger.info(f"Report submitted: {instance_id} with submission ID: {submission_id}")
        return submission_id
    
    def approve_report(self, instance_id: str, approval_notes: str = None) -> bool:
        """
        Approve a report for submission.
        
        Args:
            instance_id: Report instance ID
            approval_notes: Optional approval notes
            
        Returns:
            True if approved successfully
        """
        if instance_id not in self.report_instances:
            return False
        
        instance = self.report_instances[instance_id]
        
        if instance.status != ReportStatus.PENDING_REVIEW:
            return False
        
        instance.status = ReportStatus.APPROVED
        instance.approval_notes = approval_notes
        
        self._update_report_instance(instance)
        
        logger.info(f"Report approved: {instance_id}")
        return True
    
    def get_report_status(self, instance_id: str) -> Optional[ReportStatus]:
        """Get report status."""
        if instance_id in self.report_instances:
            return self.report_instances[instance_id].status
        return None
    
    def get_due_reports(self, days_ahead: int = 7) -> List[ReportInstance]:
        """
        Get reports due within specified days.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of due reports
        """
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        
        due_reports = []
        for instance in self.report_instances.values():
            if instance.due_date <= cutoff_date and instance.status in [ReportStatus.DRAFT, ReportStatus.PENDING_REVIEW]:
                due_reports.append(instance)
        
        return sorted(due_reports, key=lambda x: x.due_date)
    
    def get_report_history(self, report_id: str = None, start_date: datetime = None, 
                          end_date: datetime = None) -> List[ReportInstance]:
        """
        Get report history with optional filters.
        
        Args:
            report_id: Filter by report ID
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of report instances
        """
        reports = list(self.report_instances.values())
        
        if report_id:
            reports = [r for r in reports if r.report_id == report_id]
        
        if start_date:
            reports = [r for r in reports if r.period_start >= start_date]
        
        if end_date:
            reports = [r for r in reports if r.period_end <= end_date]
        
        return sorted(reports, key=lambda x: x.period_end, reverse=True)
    
    def generate_compliance_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance summary for reporting.
        
        Args:
            start_date: Start date for summary
            end_date: End date for summary
            
        Returns:
            Compliance summary dictionary
        """
        reports = self.get_report_history(start_date=start_date, end_date=end_date)
        
        # Calculate statistics
        total_reports = len(reports)
        submitted_reports = len([r for r in reports if r.status == ReportStatus.SUBMITTED])
        approved_reports = len([r for r in reports if r.status == ReportStatus.APPROVED])
        pending_reports = len([r for r in reports if r.status == ReportStatus.PENDING_REVIEW])
        rejected_reports = len([r for r in reports if r.status == ReportStatus.REJECTED])
        
        # Group by report type
        by_type = {}
        for report in reports:
            report_type = report.report_type.value
            if report_type not in by_type:
                by_type[report_type] = {'total': 0, 'submitted': 0, 'pending': 0, 'rejected': 0}
            
            by_type[report_type]['total'] += 1
            if report.status == ReportStatus.SUBMITTED:
                by_type[report_type]['submitted'] += 1
            elif report.status == ReportStatus.PENDING_REVIEW:
                by_type[report_type]['pending'] += 1
            elif report.status == ReportStatus.REJECTED:
                by_type[report_type]['rejected'] += 1
        
        # Calculate compliance rate
        compliance_rate = (submitted_reports / total_reports * 100) if total_reports > 0 else 100
        
        return {
            'summary': {
                'total_reports': total_reports,
                'submitted_reports': submitted_reports,
                'approved_reports': approved_reports,
                'pending_reports': pending_reports,
                'rejected_reports': rejected_reports,
                'compliance_rate': compliance_rate
            },
            'by_report_type': by_type,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat()
        }

class RegulatoryReporter:
    """
    High-level regulatory reporting interface.
    
    This class provides a simplified interface for common regulatory
    reporting tasks and automated report generation workflows.
    """
    
    def __init__(self, db_path: str = "data/regulatory_reporting.db"):
        """
        Initialize regulatory reporter.
        
        Args:
            db_path: Path to regulatory reporting database
        """
        self.generator = ReportGenerator(db_path)
        logger.info("Regulatory Reporter initialized")
    
    def generate_form_13f(self, period_start: datetime, period_end: datetime, 
                         holdings_data: List[Dict[str, Any]]) -> str:
        """
        Generate Form 13F report.
        
        Args:
            period_start: Report period start
            period_end: Report period end
            holdings_data: Holdings data
            
        Returns:
            Report instance ID
        """
        # Create report instance
        instance_id = self.generator.create_report_instance("REPORT_001", period_start, period_end)
        
        # Prepare data
        total_value = sum(holding.get('market_value', 0) for holding in holdings_data)
        data = {
            'filing_date': datetime.now().isoformat(),
            'reporting_period': period_end.isoformat(),
            'total_value': total_value,
            'holdings_count': len(holdings_data),
            'holdings': holdings_data
        }
        
        # Generate report
        file_path = self.generator.generate_report(instance_id, data)
        
        logger.info(f"Generated Form 13F: {file_path}")
        return instance_id
    
    def generate_trace_report(self, trade_date: datetime, trade_data: List[Dict[str, Any]]) -> str:
        """
        Generate TRACE report.
        
        Args:
            trade_date: Trade date
            trade_data: Trade data
            
        Returns:
            Report instance ID
        """
        # Create report instance
        instance_id = self.generator.create_report_instance("REPORT_002", trade_date, trade_date)
        
        # Prepare data
        total_volume = sum(trade.get('volume', 0) for trade in trade_data)
        data = {
            'reporting_date': trade_date.isoformat(),
            'trade_count': len(trade_data),
            'total_volume': total_volume,
            'trades': trade_data
        }
        
        # Generate report
        file_path = self.generator.generate_report(instance_id, data)
        
        logger.info(f"Generated TRACE report: {file_path}")
        return instance_id
    
    def generate_compliance_report(self, period_start: datetime, period_end: datetime,
                                  compliance_data: Dict[str, Any]) -> str:
        """
        Generate compliance report.
        
        Args:
            period_start: Report period start
            period_end: Report period end
            compliance_data: Compliance data
            
        Returns:
            Report instance ID
        """
        # Create report instance
        instance_id = self.generator.create_report_instance("REPORT_003", period_start, period_end)
        
        # Prepare data
        data = {
            'reporting_period': f"{period_start.date()} to {period_end.date()}",
            'violation_count': compliance_data.get('violation_count', 0),
            'compliance_score': compliance_data.get('compliance_score', 100),
            'risk_violations': compliance_data.get('risk_violations', []),
            'sec_violations': compliance_data.get('sec_violations', []),
            'recommendations': compliance_data.get('recommendations', [])
        }
        
        # Generate report
        file_path = self.generator.generate_report(instance_id, data)
        
        logger.info(f"Generated compliance report: {file_path}")
        return instance_id
    
    def generate_risk_report(self, report_date: datetime, risk_data: Dict[str, Any]) -> str:
        """
        Generate risk report.
        
        Args:
            report_date: Report date
            risk_data: Risk data
            
        Returns:
            Report instance ID
        """
        # Create report instance
        instance_id = self.generator.create_report_instance("REPORT_004", report_date, report_date)
        
        # Prepare data
        data = {
            'report_date': report_date.isoformat(),
            'portfolio_value': risk_data.get('portfolio_value', 0),
            'var_95': risk_data.get('var_95', 0),
            'var_99': risk_data.get('var_99', 0),
            'max_drawdown': risk_data.get('max_drawdown', 0),
            'sharpe_ratio': risk_data.get('sharpe_ratio', 0),
            'beta': risk_data.get('beta', 0),
            'volatility': risk_data.get('volatility', 0),
            'leverage': risk_data.get('leverage', 0),
            'violations': risk_data.get('violations', [])
        }
        
        # Generate report
        file_path = self.generator.generate_report(instance_id, data)
        
        logger.info(f"Generated risk report: {file_path}")
        return instance_id
    
    def submit_report(self, instance_id: str, notes: str = None) -> str:
        """Submit a report."""
        return self.generator.submit_report(instance_id, notes)
    
    def approve_report(self, instance_id: str, notes: str = None) -> bool:
        """Approve a report."""
        return self.generator.approve_report(instance_id, notes)
    
    def get_due_reports(self, days_ahead: int = 7) -> List[ReportInstance]:
        """Get due reports."""
        return self.generator.get_due_reports(days_ahead)
    
    def get_compliance_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get compliance summary."""
        return self.generator.generate_compliance_summary(start_date, end_date)
