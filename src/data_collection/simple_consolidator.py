"""
Simple Data Consolidator - Working Version

Basic data consolidation functionality for testing.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataConsolidator:
    """Simple data consolidator for merging multiple sources"""
    
    def __init__(self):
        self.priority_rules = {
            'alpha_vantage': 1,  # Highest priority
            'yahoo': 2,          # Lower priority
            'questrade': 3       # Lowest priority
        }
        logger.info("🔄 Simple Data Consolidator initialized")
    
    def merge_price_data(self, primary, secondary, primary_source="primary", secondary_source="secondary"):
        """Merge two datasets with priority-based conflict resolution"""
        
        if primary.empty and secondary.empty:
            return pd.DataFrame()
        
        if primary.empty:
            return secondary.copy()
        
        if secondary.empty:
            return primary.copy()
        
        # Determine priority
        primary_priority = self.priority_rules.get(primary_source, 999)
        secondary_priority = self.priority_rules.get(secondary_source, 999)
        
        if primary_priority <= secondary_priority:
            high_priority, low_priority = primary, secondary
        else:
            high_priority, low_priority = secondary, primary
        
        # Start with high priority data
        result = high_priority.copy()
        
        # Fill missing values with low priority data
        for col in high_priority.columns:
            if col in low_priority.columns:
                missing_mask = result[col].isna()
                available_mask = low_priority[col].notna()
                fill_mask = missing_mask & available_mask
                
                if fill_mask.any():
                    result.loc[fill_mask, col] = low_priority.loc[fill_mask, col]
        
        return result

# Test the class
if __name__ == "__main__":
    consolidator = DataConsolidator()
    print("✅ DataConsolidator created successfully")