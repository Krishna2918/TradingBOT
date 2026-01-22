"""
Training Data Collection Orchestrator

Manages the complete 20-year data collection process for AI training.
Coordinates between different data sources and ensures comprehensive coverage.
"""

import logging
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .comprehensive_data_collector import ComprehensiveDataCollector
from .alpha_vantage_key_manager import get_alpha_vantage_key_manager

logger = logging.getLogger(__name__)

class TrainingDataOrchestrator:
    """
    Orchestrates the complete training data collection process
    
    Manages:
    1. Data collection scheduling and prioritization
    2. Progress monitoring and reporting
    3. Quality validation and completeness checks
    4. Recovery from failures and retries
    5. Final dataset preparation for AI training
    """
    
    def __init__(self, base_path: str = "TrainingData"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.collector = ComprehensiveDataCollector(str(self.base_path))
        self.key_manager = get_alpha_vantage_key_manager()
        
        # Collection configuration
        self.config = {
            'target_symbols': 200,
            'min_years_history': 15,
            'required_completeness': 0.95,
            'max_retries': 3,
            'collection_phases': [
                'market_data',
                'fundamentals', 
                'macro_economics',
                'sentiment',
                'validation'
            ]
        }
        
        # Status tracking
        self.status_file = self.base_path / 'orchestrator_status.json'
        self.status = self._load_status()
        
        logger.info("Training Data Orchestrator initialized")
    
    def _load_status(self) -> Dict[str, Any]:
        """Load orchestrator status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading status: {e}")
        
        return {
            'collection_started': False,
            'current_phase': None,
            'phases_completed': [],
            'start_time': None,
            'estimated_completion': None,
            'total_progress': 0.0,
            'errors': [],
            'retries': {}
        }
    
    def _save_status(self):
        """Save orchestrator status to file"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status: {e}")
    
    async def start_full_collection(self) -> Dict[str, Any]:
        """
        Start the complete training data collection process
        
        Returns comprehensive results and status
        """
        logger.info("Starting full training data collection")
        
        self.status['collection_started'] = True
        self.status['start_time'] = datetime.now().isoformat()
        self._save_status()
        
        results = {
            'success': False,
            'start_time': datetime.now(),
            'phases': {},
            'final_validation': {},
            'dataset_ready': False
        }
        
        try:
            # Execute collection phases in order
            for phase in self.config['collection_phases']:
                logger.info(f"📋 Starting phase: {phase}")
                self.status['current_phase'] = phase
                self._save_status()
                
                phase_result = await self._execute_phase(phase)
                results['phases'][phase] = phase_result
                
                if phase_result.get('success', False):
                    self.status['phases_completed'].append(phase)
                    logger.info(f"Phase {phase} completed successfully")
                else:
                    logger.error(f"Phase {phase} failed")
                    # Continue with other phases but mark overall failure
                
                self._update_progress()
                self._save_status()
            
            # Final validation and dataset preparation
            logger.info("🔍 Performing final validation")
            final_validation = await self._final_validation()
            results['final_validation'] = final_validation
            results['dataset_ready'] = final_validation.get('training_ready', False)
            
            results['success'] = results['dataset_ready']
            results['end_time'] = datetime.now()
            results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
            
            if results['success']:
                logger.info(f"Training data collection completed successfully in {results['duration']:.0f} seconds")
            else:
                logger.warning(f"Training data collection completed with issues in {results['duration']:.0f} seconds")
            
        except Exception as e:
            logger.error(f"Training data collection failed: {e}")
            results['error'] = str(e)
            results['end_time'] = datetime.now()
        
        return results
    
    async def _execute_phase(self, phase: str) -> Dict[str, Any]:
        """Execute a specific collection phase"""
        phase_start = datetime.now()
        
        try:
            if phase == 'market_data':
                result = await self.collector.collect_market_data()
            elif phase == 'fundamentals':
                result = await self.collector.collect_fundamentals_data()
            elif phase == 'macro_economics':
                result = await self.collector.collect_macro_data()
            elif phase == 'sentiment':
                result = await self.collector.collect_sentiment_data()
            elif phase == 'validation':
                result = await self.collector.validate_collected_data()
            else:
                raise ValueError(f"Unknown phase: {phase}")
            
            result['success'] = True
            result['duration'] = (datetime.now() - phase_start).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"Phase {phase} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': (datetime.now() - phase_start).total_seconds()
            }
    
    def _update_progress(self):
        """Update overall progress percentage"""
        completed_phases = len(self.status['phases_completed'])
        total_phases = len(self.config['collection_phases'])
        self.status['total_progress'] = (completed_phases / total_phases) * 100
        
        # Estimate completion time
        if completed_phases > 0 and self.status['start_time']:
            start_time = datetime.fromisoformat(self.status['start_time'])
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = completed_phases / elapsed
            remaining_time = (total_phases - completed_phases) / rate
            self.status['estimated_completion'] = (datetime.now() + timedelta(seconds=remaining_time)).isoformat()
    
    async def _final_validation(self) -> Dict[str, Any]:
        """
        Perform final validation of all collected data
        
        Ensures dataset is ready for AI training
        """
        logger.info("🔍 Performing comprehensive final validation")
        
        validation = {
            'data_completeness': {},
            'quality_scores': {},
            'training_readiness': {},
            'recommendations': [],
            'overall_score': 0.0,
            'training_ready': False
        }
        
        try:
            # Check market data completeness
            market_data_path = self.base_path / 'market_data'
            if market_data_path.exists():
                market_files = list(market_data_path.glob("*.parquet"))
                expected_market_files = len(self.collector.us_symbols) * 3  # daily + 1min + 5min
                
                validation['data_completeness']['market_data'] = {
                    'files_found': len(market_files),
                    'files_expected': expected_market_files,
                    'completeness': len(market_files) / expected_market_files if expected_market_files > 0 else 0
                }
            
            # Check fundamentals data
            fundamentals_path = self.base_path / 'fundamentals'
            if fundamentals_path.exists():
                fundamentals_files = list(fundamentals_path.glob("*.parquet"))
                expected_fundamentals = len(self.collector.us_symbols) * 4  # overview, income, balance, cash_flow
                
                validation['data_completeness']['fundamentals'] = {
                    'files_found': len(fundamentals_files),
                    'files_expected': expected_fundamentals,
                    'completeness': len(fundamentals_files) / expected_fundamentals if expected_fundamentals > 0 else 0
                }
            
            # Check macro economics data
            macro_path = self.base_path / 'macro_economics'
            if macro_path.exists():
                macro_files = list(macro_path.glob("*.parquet"))
                validation['data_completeness']['macro_economics'] = {
                    'files_found': len(macro_files),
                    'indicators_expected': 20,  # Various macro indicators
                    'completeness': min(len(macro_files) / 20, 1.0)
                }
            
            # Check sentiment data
            sentiment_path = self.base_path / 'sentiment'
            if sentiment_path.exists():
                sentiment_files = list(sentiment_path.glob("*.parquet"))
                validation['data_completeness']['sentiment'] = {
                    'files_found': len(sentiment_files),
                    'completeness': min(len(sentiment_files) / 100, 1.0)  # Sentiment files
                }
            
            # Calculate overall completeness
            completeness_scores = []
            for data_type, metrics in validation['data_completeness'].items():
                if 'completeness' in metrics:
                    completeness_scores.append(metrics['completeness'])
            
            if completeness_scores:
                validation['overall_score'] = sum(completeness_scores) / len(completeness_scores)
            
            # Determine training readiness
            validation['training_ready'] = validation['overall_score'] >= self.config['required_completeness']
            
            # Generate recommendations
            if validation['overall_score'] < 0.8:
                validation['recommendations'].append("Data collection is incomplete. Consider re-running failed collections.")
            elif validation['overall_score'] < 0.95:
                validation['recommendations'].append("Data collection is mostly complete but some gaps exist.")
            else:
                validation['recommendations'].append("Data collection is comprehensive and ready for AI training.")
            
            if not validation['training_ready']:
                validation['recommendations'].append(f"Minimum completeness of {self.config['required_completeness']:.1%} not met.")
            
            logger.info(f"🔍 Final validation completed - Overall score: {validation['overall_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            validation['error'] = str(e)
        
        return validation
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        collector_status = self.collector.get_collection_status()
        key_usage = self.key_manager.get_usage_summary()
        
        return {
            'orchestrator': self.status,
            'collector': collector_status,
            'key_usage': key_usage,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_collection_report(self) -> str:
        """Generate a human-readable collection report"""
        status = self.get_detailed_status()
        
        report = []
        report.append("=" * 60)
        report.append("TRAINING DATA COLLECTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall status
        report.append(f"Collection Status: {'[IN PROGRESS]' if status['orchestrator']['collection_started'] else '[NOT STARTED]'}")
        if status['orchestrator']['current_phase']:
            report.append(f"Current Phase: {status['orchestrator']['current_phase']}")
        report.append(f"Progress: {status['orchestrator']['total_progress']:.1f}%")
        report.append("")
        
        # Phases completed
        report.append("Phases Completed:")
        for phase in status['orchestrator']['phases_completed']:
            report.append(f"  [DONE] {phase}")
        
        remaining_phases = set(self.config['collection_phases']) - set(status['orchestrator']['phases_completed'])
        if remaining_phases:
            report.append("Phases Remaining:")
            for phase in remaining_phases:
                report.append(f"  [PENDING] {phase}")
        report.append("")
        
        # Key usage
        report.append("API Key Usage:")
        for key in status['key_usage']['keys']:
            report.append(f"  {key['name']} ({key['purpose']}): {key['used']}/{key['limit']} ({key['remaining']} remaining)")
        report.append("")
        
        # Data collected
        report.append("Data Files Collected:")
        for data_type, count in status['collector']['data_directories'].items():
            report.append(f"  {data_type}: {count} files")
        report.append("")
        
        # Recommendations
        if status['orchestrator']['total_progress'] < 100:
            report.append("Next Steps:")
            report.append("  1. Continue with current collection phase")
            report.append("  2. Monitor API key usage to avoid limits")
            report.append("  3. Check for any failed collections and retry")
        else:
            report.append("Collection Complete!")
            report.append("  1. Run final validation")
            report.append("  2. Prepare dataset for AI training")
            report.append("  3. Begin model training pipeline")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

# Convenience functions
async def start_training_data_collection(base_path: str = "TrainingData") -> Dict[str, Any]:
    """Start the complete training data collection process"""
    orchestrator = TrainingDataOrchestrator(base_path)
    return await orchestrator.start_full_collection()

def get_collection_status(base_path: str = "TrainingData") -> Dict[str, Any]:
    """Get current collection status"""
    orchestrator = TrainingDataOrchestrator(base_path)
    return orchestrator.get_detailed_status()

def generate_collection_report(base_path: str = "TrainingData") -> str:
    """Generate collection report"""
    orchestrator = TrainingDataOrchestrator(base_path)
    return orchestrator.generate_collection_report()

if __name__ == '__main__':
    # Test the orchestrator
    logging.basicConfig(level=logging.INFO)
    
    async def test_orchestrator():
        orchestrator = TrainingDataOrchestrator()
        
        # Generate initial report
        print("Initial Status Report:")
        print(orchestrator.generate_collection_report())
        
        # Test status
        status = orchestrator.get_detailed_status()
        print(f"\nOrchestrator initialized")
        print(f"Target symbols: {len(orchestrator.collector.us_symbols)}")
        print(f"Collection phases: {orchestrator.config['collection_phases']}")
    
    asyncio.run(test_orchestrator())