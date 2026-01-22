"""
Encrypted Backup Mirror
Implements secure, encrypted backup and disaster recovery for the trading system
"""

import os
import json
import pickle
import hashlib
import hmac
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import shutil
import gzip
import base64
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import uuid
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import schedule

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Backup types"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    CONFIGURATION = "configuration"
    DATA = "data"
    MODELS = "models"
    LOGS = "logs"
    TRADES = "trades"

class BackupStatus(Enum):
    """Backup status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"

@dataclass
class BackupConfig:
    """Backup configuration"""
    backup_id: str
    backup_type: BackupType
    source_paths: List[str]
    destination_path: str
    encryption_key: Optional[str] = None
    compression: bool = True
    verification: bool = True
    retention_days: int = 30
    metadata: Dict[str, Any] = None

@dataclass
class BackupRecord:
    """Backup record"""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    status: BackupStatus
    source_paths: List[str]
    destination_path: str
    file_count: int
    total_size: int
    compressed_size: int
    checksum: str
    encryption_key_id: str
    verification_status: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class EncryptedBackupMirror:
    """Secure encrypted backup and disaster recovery system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.backup_root = Path(config.get('backup_root', 'backups'))
        self.encryption_key = self._generate_or_load_encryption_key()
        self.backup_queue = queue.Queue()
        self.backup_records = {}
        self.verification_results = {}
        
        # Create backup directory structure
        self._create_backup_structure()
        
        # Initialize database
        self._init_backup_database()
        
        # Start backup worker thread
        self.backup_worker_thread = threading.Thread(target=self._backup_worker, daemon=True)
        self.backup_worker_thread.start()
        
        # Schedule automatic backups
        self._schedule_automatic_backups()
        
        logger.info("Encrypted Backup Mirror initialized")
    
    def _generate_or_load_encryption_key(self) -> bytes:
        """Generate or load encryption key"""
        try:
            key_file = Path(self.config.get('key_file', 'backup_key.key'))
            
            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    key = f.read()
                logger.info("Loaded existing encryption key")
            else:
                # Generate new key
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                logger.info("Generated new encryption key")
            
            return key
            
        except Exception as e:
            logger.error(f"Error with encryption key: {e}")
            # Generate temporary key
            return Fernet.generate_key()
    
    def _create_backup_structure(self):
        """Create backup directory structure"""
        try:
            # Create main backup directory
            self.backup_root.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            subdirs = [
                'full_backups',
                'incremental_backups',
                'differential_backups',
                'config_backups',
                'data_backups',
                'model_backups',
                'log_backups',
                'trade_backups',
                'verification',
                'metadata'
            ]
            
            for subdir in subdirs:
                (self.backup_root / subdir).mkdir(exist_ok=True)
            
            logger.info(f"Created backup structure at {self.backup_root}")
            
        except Exception as e:
            logger.error(f"Error creating backup structure: {e}")
    
    def _init_backup_database(self):
        """Initialize backup database"""
        try:
            db_path = self.backup_root / 'backup_records.db'
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Create backup records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS backup_records (
                        backup_id TEXT PRIMARY KEY,
                        backup_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        source_paths TEXT NOT NULL,
                        destination_path TEXT NOT NULL,
                        file_count INTEGER,
                        total_size INTEGER,
                        compressed_size INTEGER,
                        checksum TEXT,
                        encryption_key_id TEXT,
                        verification_status TEXT,
                        error_message TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create verification results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS verification_results (
                        backup_id TEXT PRIMARY KEY,
                        verification_timestamp TEXT NOT NULL,
                        verification_status TEXT NOT NULL,
                        file_integrity_check TEXT,
                        checksum_verification TEXT,
                        encryption_verification TEXT,
                        error_message TEXT
                    )
                ''')
                
                conn.commit()
            
            logger.info("Backup database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing backup database: {e}")
    
    def _schedule_automatic_backups(self):
        """Schedule automatic backups"""
        try:
            # Schedule full backup daily at 2 AM
            schedule.every().day.at("02:00").do(self._automatic_full_backup)
            
            # Schedule incremental backup every 4 hours
            schedule.every(4).hours.do(self._automatic_incremental_backup)
            
            # Schedule configuration backup every hour
            schedule.every().hour.do(self._automatic_config_backup)
            
            # Schedule verification daily at 3 AM
            schedule.every().day.at("03:00").do(self._automatic_verification)
            
            # Schedule cleanup weekly
            schedule.every().week.do(self._automatic_cleanup)
            
            logger.info("Automatic backups scheduled")
            
        except Exception as e:
            logger.error(f"Error scheduling automatic backups: {e}")
    
    def create_backup(self, backup_config: BackupConfig) -> str:
        """Create a backup"""
        try:
            # Generate backup ID if not provided
            if not backup_config.backup_id:
                backup_config.backup_id = f"backup_{uuid.uuid4().hex[:8]}"
            
            # Add to queue
            self.backup_queue.put(backup_config)
            
            logger.info(f"Backup queued: {backup_config.backup_id} ({backup_config.backup_type.value})")
            return backup_config.backup_id
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
    def _backup_worker(self):
        """Backup worker thread"""
        while True:
            try:
                # Get backup config from queue
                backup_config = self.backup_queue.get(timeout=1)
                
                # Process backup
                self._process_backup(backup_config)
                
                # Mark task as done
                self.backup_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in backup worker: {e}")
    
    def _process_backup(self, backup_config: BackupConfig):
        """Process a backup"""
        try:
            backup_id = backup_config.backup_id
            
            # Create backup record
            backup_record = BackupRecord(
                backup_id=backup_id,
                backup_type=backup_config.backup_type,
                timestamp=datetime.now(),
                status=BackupStatus.IN_PROGRESS,
                source_paths=backup_config.source_paths,
                destination_path=backup_config.destination_path,
                file_count=0,
                total_size=0,
                compressed_size=0,
                checksum="",
                encryption_key_id="",
                verification_status="pending",
                metadata=backup_config.metadata or {}
            )
            
            # Store record
            self.backup_records[backup_id] = backup_record
            
            # Create destination directory
            dest_path = Path(backup_config.destination_path)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Process backup based on type
            if backup_config.backup_type == BackupType.FULL:
                result = self._create_full_backup(backup_config)
            elif backup_config.backup_type == BackupType.INCREMENTAL:
                result = self._create_incremental_backup(backup_config)
            elif backup_config.backup_type == BackupType.DIFFERENTIAL:
                result = self._create_differential_backup(backup_config)
            else:
                result = self._create_specific_backup(backup_config)
            
            # Update record
            backup_record.status = BackupStatus.COMPLETED if result['success'] else BackupStatus.FAILED
            backup_record.file_count = result.get('file_count', 0)
            backup_record.total_size = result.get('total_size', 0)
            backup_record.compressed_size = result.get('compressed_size', 0)
            backup_record.checksum = result.get('checksum', '')
            backup_record.encryption_key_id = result.get('encryption_key_id', '')
            backup_record.error_message = result.get('error_message')
            
            # Store in database
            self._store_backup_record(backup_record)
            
            # Verify backup if requested
            if backup_config.verification and result['success']:
                self._verify_backup(backup_id)
            
            logger.info(f"Backup completed: {backup_id} - {backup_record.status.value}")
            
        except Exception as e:
            logger.error(f"Error processing backup: {e}")
            if backup_id in self.backup_records:
                self.backup_records[backup_id].status = BackupStatus.FAILED
                self.backup_records[backup_id].error_message = str(e)
    
    def _create_full_backup(self, backup_config: BackupConfig) -> Dict[str, Any]:
        """Create full backup"""
        try:
            dest_path = Path(backup_config.destination_path)
            backup_file = dest_path / f"full_backup_{backup_config.backup_id}.tar.gz.enc"
            
            # Collect all files
            all_files = []
            total_size = 0
            
            for source_path in backup_config.source_paths:
                source = Path(source_path)
                if source.exists():
                    if source.is_file():
                        all_files.append(source)
                        total_size += source.stat().st_size
                    elif source.is_dir():
                        for file_path in source.rglob('*'):
                            if file_path.is_file():
                                all_files.append(file_path)
                                total_size += file_path.stat().st_size
            
            # Create backup archive
            backup_data = self._create_backup_archive(all_files)
            
            # Compress if requested
            if backup_config.compression:
                backup_data = gzip.compress(backup_data)
            
            # Encrypt
            encrypted_data = self._encrypt_data(backup_data)
            
            # Write to file
            with open(backup_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Calculate checksum
            checksum = hashlib.sha256(encrypted_data).hexdigest()
            
            return {
                'success': True,
                'file_count': len(all_files),
                'total_size': total_size,
                'compressed_size': len(encrypted_data),
                'checksum': checksum,
                'encryption_key_id': 'default'
            }
            
        except Exception as e:
            logger.error(f"Error creating full backup: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _create_incremental_backup(self, backup_config: BackupConfig) -> Dict[str, Any]:
        """Create incremental backup"""
        try:
            # Find last full backup
            last_full_backup = self._find_last_backup(BackupType.FULL)
            if not last_full_backup:
                logger.warning("No full backup found, creating full backup instead")
                return self._create_full_backup(backup_config)
            
            # Get files modified since last backup
            modified_files = self._get_modified_files_since(last_full_backup.timestamp)
            
            if not modified_files:
                logger.info("No files modified since last backup")
                return {
                    'success': True,
                    'file_count': 0,
                    'total_size': 0,
                    'compressed_size': 0,
                    'checksum': '',
                    'encryption_key_id': 'default'
                }
            
            # Create backup with modified files only
            dest_path = Path(backup_config.destination_path)
            backup_file = dest_path / f"incremental_backup_{backup_config.backup_id}.tar.gz.enc"
            
            # Create backup archive
            backup_data = self._create_backup_archive(modified_files)
            
            # Compress if requested
            if backup_config.compression:
                backup_data = gzip.compress(backup_data)
            
            # Encrypt
            encrypted_data = self._encrypt_data(backup_data)
            
            # Write to file
            with open(backup_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Calculate checksum
            checksum = hashlib.sha256(encrypted_data).hexdigest()
            
            return {
                'success': True,
                'file_count': len(modified_files),
                'total_size': sum(f.stat().st_size for f in modified_files),
                'compressed_size': len(encrypted_data),
                'checksum': checksum,
                'encryption_key_id': 'default'
            }
            
        except Exception as e:
            logger.error(f"Error creating incremental backup: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _create_differential_backup(self, backup_config: BackupConfig) -> Dict[str, Any]:
        """Create differential backup"""
        try:
            # Find last full backup
            last_full_backup = self._find_last_backup(BackupType.FULL)
            if not last_full_backup:
                logger.warning("No full backup found, creating full backup instead")
                return self._create_full_backup(backup_config)
            
            # Get files modified since last full backup
            modified_files = self._get_modified_files_since(last_full_backup.timestamp)
            
            # Create backup with modified files
            return self._create_backup_with_files(backup_config, modified_files, "differential")
            
        except Exception as e:
            logger.error(f"Error creating differential backup: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _create_specific_backup(self, backup_config: BackupConfig) -> Dict[str, Any]:
        """Create specific type backup (config, data, models, etc.)"""
        try:
            # Filter files based on backup type
            filtered_files = self._filter_files_by_type(backup_config.source_paths, backup_config.backup_type)
            
            # Create backup with filtered files
            return self._create_backup_with_files(backup_config, filtered_files, backup_config.backup_type.value)
            
        except Exception as e:
            logger.error(f"Error creating specific backup: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _create_backup_with_files(self, backup_config: BackupConfig, files: List[Path], 
                                 backup_type: str) -> Dict[str, Any]:
        """Create backup with specific files"""
        try:
            if not files:
                return {
                    'success': True,
                    'file_count': 0,
                    'total_size': 0,
                    'compressed_size': 0,
                    'checksum': '',
                    'encryption_key_id': 'default'
                }
            
            dest_path = Path(backup_config.destination_path)
            backup_file = dest_path / f"{backup_type}_backup_{backup_config.backup_id}.tar.gz.enc"
            
            # Create backup archive
            backup_data = self._create_backup_archive(files)
            
            # Compress if requested
            if backup_config.compression:
                backup_data = gzip.compress(backup_data)
            
            # Encrypt
            encrypted_data = self._encrypt_data(backup_data)
            
            # Write to file
            with open(backup_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Calculate checksum
            checksum = hashlib.sha256(encrypted_data).hexdigest()
            
            return {
                'success': True,
                'file_count': len(files),
                'total_size': sum(f.stat().st_size for f in files),
                'compressed_size': len(encrypted_data),
                'checksum': checksum,
                'encryption_key_id': 'default'
            }
            
        except Exception as e:
            logger.error(f"Error creating backup with files: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _create_backup_archive(self, files: List[Path]) -> bytes:
        """Create backup archive from files"""
        try:
            import tarfile
            import io
            
            # Create tar archive in memory
            tar_buffer = io.BytesIO()
            
            with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                for file_path in files:
                    try:
                        # Add file to archive
                        tar.add(file_path, arcname=file_path.name)
                    except Exception as e:
                        logger.warning(f"Could not add file {file_path} to archive: {e}")
            
            tar_buffer.seek(0)
            return tar_buffer.read()
            
        except Exception as e:
            logger.error(f"Error creating backup archive: {e}")
            return b''
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data"""
        try:
            fernet = Fernet(self.encryption_key)
            return fernet.encrypt(data)
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return data
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        try:
            fernet = Fernet(self.encryption_key)
            return fernet.decrypt(encrypted_data)
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return encrypted_data
    
    def _find_last_backup(self, backup_type: BackupType) -> Optional[BackupRecord]:
        """Find last backup of specific type"""
        try:
            db_path = self.backup_root / 'backup_records.db'
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM backup_records 
                    WHERE backup_type = ? AND status = 'completed'
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (backup_type.value,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_backup_record(row)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding last backup: {e}")
            return None
    
    def _get_modified_files_since(self, since_timestamp: datetime) -> List[Path]:
        """Get files modified since timestamp"""
        try:
            modified_files = []
            
            for source_path in self.config.get('backup_paths', []):
                source = Path(source_path)
                if source.exists():
                    for file_path in source.rglob('*'):
                        if file_path.is_file():
                            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_mtime > since_timestamp:
                                modified_files.append(file_path)
            
            return modified_files
            
        except Exception as e:
            logger.error(f"Error getting modified files: {e}")
            return []
    
    def _filter_files_by_type(self, source_paths: List[str], backup_type: BackupType) -> List[Path]:
        """Filter files by backup type"""
        try:
            filtered_files = []
            
            for source_path in source_paths:
                source = Path(source_path)
                if source.exists():
                    if backup_type == BackupType.CONFIGURATION:
                        # Filter for config files
                        for file_path in source.rglob('*.yaml'):
                            if file_path.is_file():
                                filtered_files.append(file_path)
                        for file_path in source.rglob('*.json'):
                            if file_path.is_file():
                                filtered_files.append(file_path)
                    elif backup_type == BackupType.MODELS:
                        # Filter for model files
                        for file_path in source.rglob('*.pkl'):
                            if file_path.is_file():
                                filtered_files.append(file_path)
                        for file_path in source.rglob('*.h5'):
                            if file_path.is_file():
                                filtered_files.append(file_path)
                    elif backup_type == BackupType.LOGS:
                        # Filter for log files
                        for file_path in source.rglob('*.log'):
                            if file_path.is_file():
                                filtered_files.append(file_path)
                    elif backup_type == BackupType.TRADES:
                        # Filter for trade data files
                        for file_path in source.rglob('*trade*'):
                            if file_path.is_file():
                                filtered_files.append(file_path)
                    else:
                        # Include all files
                        for file_path in source.rglob('*'):
                            if file_path.is_file():
                                filtered_files.append(file_path)
            
            return filtered_files
            
        except Exception as e:
            logger.error(f"Error filtering files by type: {e}")
            return []
    
    def _verify_backup(self, backup_id: str):
        """Verify backup integrity"""
        try:
            if backup_id not in self.backup_records:
                logger.error(f"Backup record not found: {backup_id}")
                return
            
            backup_record = self.backup_records[backup_id]
            
            # Check if backup file exists
            backup_file = Path(backup_record.destination_path)
            if not backup_file.exists():
                backup_record.verification_status = "failed"
                backup_record.error_message = "Backup file not found"
                return
            
            # Verify checksum
            with open(backup_file, 'rb') as f:
                file_data = f.read()
            
            calculated_checksum = hashlib.sha256(file_data).hexdigest()
            if calculated_checksum != backup_record.checksum:
                backup_record.verification_status = "failed"
                backup_record.error_message = "Checksum mismatch"
                return
            
            # Test decryption
            try:
                decrypted_data = self._decrypt_data(file_data)
                if backup_record.metadata.get('compression', True):
                    decompressed_data = gzip.decompress(decrypted_data)
                else:
                    decompressed_data = decrypted_data
            except Exception as e:
                backup_record.verification_status = "failed"
                backup_record.error_message = f"Decryption failed: {e}"
                return
            
            # Update verification status
            backup_record.verification_status = "verified"
            backup_record.status = BackupStatus.VERIFIED
            
            # Store verification result
            self.verification_results[backup_id] = {
                'verification_timestamp': datetime.now(),
                'verification_status': 'verified',
                'file_integrity_check': 'passed',
                'checksum_verification': 'passed',
                'encryption_verification': 'passed'
            }
            
            logger.info(f"Backup verified: {backup_id}")
            
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            if backup_id in self.backup_records:
                self.backup_records[backup_id].verification_status = "failed"
                self.backup_records[backup_id].error_message = str(e)
    
    def restore_backup(self, backup_id: str, restore_path: str) -> bool:
        """Restore backup"""
        try:
            if backup_id not in self.backup_records:
                logger.error(f"Backup record not found: {backup_id}")
                return False
            
            backup_record = self.backup_records[backup_id]
            
            if backup_record.status != BackupStatus.VERIFIED:
                logger.error(f"Backup not verified: {backup_id}")
                return False
            
            # Read backup file
            backup_file = Path(backup_record.destination_path)
            with open(backup_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt
            decrypted_data = self._decrypt_data(encrypted_data)
            
            # Decompress if needed
            if backup_record.metadata.get('compression', True):
                archive_data = gzip.decompress(decrypted_data)
            else:
                archive_data = decrypted_data
            
            # Extract archive
            restore_dir = Path(restore_path)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            import tarfile
            import io
            
            tar_buffer = io.BytesIO(archive_data)
            with tarfile.open(fileobj=tar_buffer, mode='r') as tar:
                tar.extractall(restore_dir)
            
            logger.info(f"Backup restored: {backup_id} to {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def _store_backup_record(self, backup_record: BackupRecord):
        """Store backup record in database"""
        try:
            db_path = self.backup_root / 'backup_records.db'
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO backup_records 
                    (backup_id, backup_type, timestamp, status, source_paths, 
                     destination_path, file_count, total_size, compressed_size, 
                     checksum, encryption_key_id, verification_status, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    backup_record.backup_id,
                    backup_record.backup_type.value,
                    backup_record.timestamp.isoformat(),
                    backup_record.status.value,
                    json.dumps(backup_record.source_paths),
                    backup_record.destination_path,
                    backup_record.file_count,
                    backup_record.total_size,
                    backup_record.compressed_size,
                    backup_record.checksum,
                    backup_record.encryption_key_id,
                    backup_record.verification_status,
                    backup_record.error_message,
                    json.dumps(backup_record.metadata or {})
                ))
                
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing backup record: {e}")
    
    def _row_to_backup_record(self, row) -> BackupRecord:
        """Convert database row to BackupRecord"""
        try:
            return BackupRecord(
                backup_id=row[0],
                backup_type=BackupType(row[1]),
                timestamp=datetime.fromisoformat(row[2]),
                status=BackupStatus(row[3]),
                source_paths=json.loads(row[4]),
                destination_path=row[5],
                file_count=row[6],
                total_size=row[7],
                compressed_size=row[8],
                checksum=row[9],
                encryption_key_id=row[10],
                verification_status=row[11],
                error_message=row[12],
                metadata=json.loads(row[13]) if row[13] else {}
            )
        except Exception as e:
            logger.error(f"Error converting row to BackupRecord: {e}")
            return None
    
    def _automatic_full_backup(self):
        """Automatic full backup"""
        try:
            backup_config = BackupConfig(
                backup_id=f"auto_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                backup_type=BackupType.FULL,
                source_paths=self.config.get('backup_paths', []),
                destination_path=str(self.backup_root / 'full_backups'),
                compression=True,
                verification=True,
                retention_days=30
            )
            
            self.create_backup(backup_config)
            logger.info("Automatic full backup scheduled")
            
        except Exception as e:
            logger.error(f"Error in automatic full backup: {e}")
    
    def _automatic_incremental_backup(self):
        """Automatic incremental backup"""
        try:
            backup_config = BackupConfig(
                backup_id=f"auto_inc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                backup_type=BackupType.INCREMENTAL,
                source_paths=self.config.get('backup_paths', []),
                destination_path=str(self.backup_root / 'incremental_backups'),
                compression=True,
                verification=True,
                retention_days=7
            )
            
            self.create_backup(backup_config)
            logger.info("Automatic incremental backup scheduled")
            
        except Exception as e:
            logger.error(f"Error in automatic incremental backup: {e}")
    
    def _automatic_config_backup(self):
        """Automatic configuration backup"""
        try:
            backup_config = BackupConfig(
                backup_id=f"auto_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                backup_type=BackupType.CONFIGURATION,
                source_paths=self.config.get('config_paths', ['config/']),
                destination_path=str(self.backup_root / 'config_backups'),
                compression=True,
                verification=True,
                retention_days=90
            )
            
            self.create_backup(backup_config)
            logger.info("Automatic configuration backup scheduled")
            
        except Exception as e:
            logger.error(f"Error in automatic configuration backup: {e}")
    
    def _automatic_verification(self):
        """Automatic verification of recent backups"""
        try:
            # Get recent backups
            recent_backups = self._get_recent_backups(days=7)
            
            for backup_id in recent_backups:
                if backup_id not in self.verification_results:
                    self._verify_backup(backup_id)
            
            logger.info("Automatic verification completed")
            
        except Exception as e:
            logger.error(f"Error in automatic verification: {e}")
    
    def _automatic_cleanup(self):
        """Automatic cleanup of old backups"""
        try:
            # Get old backups
            old_backups = self._get_old_backups(days=30)
            
            for backup_id in old_backups:
                self._delete_backup(backup_id)
            
            logger.info("Automatic cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in automatic cleanup: {e}")
    
    def _get_recent_backups(self, days: int = 7) -> List[str]:
        """Get recent backups"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            db_path = self.backup_root / 'backup_records.db'
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT backup_id FROM backup_records 
                    WHERE timestamp > ? AND status = 'completed'
                    ORDER BY timestamp DESC
                ''', (cutoff_date.isoformat(),))
                
                return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting recent backups: {e}")
            return []
    
    def _get_old_backups(self, days: int = 30) -> List[str]:
        """Get old backups"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            db_path = self.backup_root / 'backup_records.db'
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT backup_id FROM backup_records 
                    WHERE timestamp < ? AND status = 'completed'
                    ORDER BY timestamp ASC
                ''', (cutoff_date.isoformat(),))
                
                return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting old backups: {e}")
            return []
    
    def _delete_backup(self, backup_id: str):
        """Delete backup"""
        try:
            if backup_id not in self.backup_records:
                logger.warning(f"Backup record not found: {backup_id}")
                return
            
            backup_record = self.backup_records[backup_id]
            
            # Delete backup file
            backup_file = Path(backup_record.destination_path)
            if backup_file.exists():
                backup_file.unlink()
            
            # Remove from database
            db_path = self.backup_root / 'backup_records.db'
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM backup_records WHERE backup_id = ?', (backup_id,))
                conn.commit()
            
            # Remove from memory
            del self.backup_records[backup_id]
            
            logger.info(f"Backup deleted: {backup_id}")
            
        except Exception as e:
            logger.error(f"Error deleting backup: {e}")
    
    def get_backup_statistics(self) -> Dict:
        """Get backup statistics"""
        try:
            db_path = self.backup_root / 'backup_records.db'
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get total backups
                cursor.execute('SELECT COUNT(*) FROM backup_records')
                total_backups = cursor.fetchone()[0]
                
                # Get successful backups
                cursor.execute('SELECT COUNT(*) FROM backup_records WHERE status = "completed"')
                successful_backups = cursor.fetchone()[0]
                
                # Get failed backups
                cursor.execute('SELECT COUNT(*) FROM backup_records WHERE status = "failed"')
                failed_backups = cursor.fetchone()[0]
                
                # Get total size
                cursor.execute('SELECT SUM(compressed_size) FROM backup_records WHERE status = "completed"')
                total_size = cursor.fetchone()[0] or 0
                
                # Get backup types
                cursor.execute('SELECT backup_type, COUNT(*) FROM backup_records GROUP BY backup_type')
                backup_types = dict(cursor.fetchall())
            
            return {
                'total_backups': total_backups,
                'successful_backups': successful_backups,
                'failed_backups': failed_backups,
                'success_rate': successful_backups / total_backups if total_backups > 0 else 0,
                'total_size_mb': total_size / (1024 * 1024),
                'backup_types': backup_types,
                'queue_size': self.backup_queue.qsize(),
                'verification_results': len(self.verification_results)
            }
            
        except Exception as e:
            logger.error(f"Error getting backup statistics: {e}")
            return {}
    
    def run_scheduled_tasks(self):
        """Run scheduled tasks"""
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Error running scheduled tasks: {e}")
