#!/usr/bin/env python
"""
Merge Kaggle Data Files
=======================
Combine all downloaded Kaggle CSV files into one large dataset.

Usage:
    python src/merge_kaggle_data.py
"""

import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def find_csv_files(kaggle_dir: Path) -> List[Path]:
    """Find all CSV files in kaggle directory recursively."""
    csv_files = []
    
    for pattern in ['*.csv', '**/*.csv']:
        csv_files.extend(kaggle_dir.glob(pattern))
    
    # Remove duplicates and sort by size (largest first)
    csv_files = list(set(csv_files))
    csv_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    return csv_files


def load_and_standardize(file_path: Path) -> pd.DataFrame:
    """Load CSV and standardize column names."""
    logger.info(f"Loading: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.error(f"Could not read {file_path}")
        return None
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Map common column variations
    column_map = {
        'usd_goal_real': 'goal',
        'usd_pledged_real': 'pledged',
        'usd_pledged': 'pledged',
        'main_category': 'category',
        'backers_count': 'backers',
    }
    
    for old, new in column_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple dataframes, handling different schemas."""
    if not dfs:
        return None
    
    # Find common columns
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols = common_cols.intersection(set(df.columns))
    
    logger.info(f"Common columns across all files: {len(common_cols)}")
    
    # Select important columns that exist
    important_cols = ['id', 'name', 'category', 'subcategory', 'goal', 'pledged', 
                      'backers', 'state', 'country', 'launched', 'deadline']
    
    cols_to_keep = [c for c in important_cols if c in common_cols]
    
    # Merge
    merged_dfs = []
    for df in dfs:
        available = [c for c in cols_to_keep if c in df.columns]
        merged_dfs.append(df[available])
    
    merged = pd.concat(merged_dfs, ignore_index=True)
    
    # Remove duplicates by ID if available
    if 'id' in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=['id'], keep='first')
        logger.info(f"Removed {before - len(merged)} duplicates")
    
    return merged


def process_merged_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process merged data into standard format."""
    # Filter to completed campaigns
    if 'state' in df.columns:
        valid_states = ['successful', 'failed', 'Successful', 'Failed']
        df = df[df['state'].isin(valid_states)]
        df['status'] = df['state'].str.lower()
    
    # Rename columns
    rename_map = {
        'goal': 'funding_goal',
        'pledged': 'pledged_amount',
        'backers': 'backers_count'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Parse dates
    if 'launched' in df.columns:
        df['launch_date'] = pd.to_datetime(df['launched'], errors='coerce')
    if 'deadline' in df.columns:
        df['end_date'] = pd.to_datetime(df['deadline'], errors='coerce')
    
    # Calculate duration
    if 'launch_date' in df.columns and 'end_date' in df.columns:
        df['campaign_duration_days'] = (df['end_date'] - df['launch_date']).dt.days
    
    # Filter valid
    if 'funding_goal' in df.columns:
        df = df[df['funding_goal'] > 0]
    if 'campaign_duration_days' in df.columns:
        df = df[(df['campaign_duration_days'] > 0) & (df['campaign_duration_days'] <= 90)]
    
    # Calculate metrics
    if 'pledged_amount' in df.columns and 'funding_goal' in df.columns:
        df['is_successful'] = (df['pledged_amount'] >= df['funding_goal']).astype(int)
        df['funding_ratio'] = df['pledged_amount'] / df['funding_goal'].replace(0, 1)
    
    return df


def main():
    """Main function."""
    print("=" * 60)
    print("MERGING KAGGLE DATA FILES")
    print("=" * 60)
    
    try:
        config = load_config()
        base_dir = Path(__file__).parent.parent
        
        kaggle_dir = base_dir / config.get('data', {}).get('kaggle_dir', 'data/raw/kaggle')
        
        if not kaggle_dir.exists():
            logger.error(f"Kaggle directory not found: {kaggle_dir}")
            return False
        
        # Find all CSV files
        csv_files = find_csv_files(kaggle_dir)
        
        if not csv_files:
            logger.error("No CSV files found")
            return False
        
        print(f"Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Load all files
        dataframes = []
        for file_path in csv_files:
            df = load_and_standardize(file_path)
            if df is not None:
                dataframes.append(df)
        
        if not dataframes:
            logger.error("No data loaded")
            return False
        
        # Merge
        merged = merge_dataframes(dataframes)
        logger.info(f"Merged dataset: {len(merged)} rows")
        
        # Process
        processed = process_merged_data(merged)
        logger.info(f"Processed dataset: {len(processed)} rows")
        
        # Sample if configured
        sample_size = config.get('processing', {}).get('sample_size')
        if sample_size and len(processed) > sample_size:
            processed = processed.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled to: {len(processed)} rows")
        
        # Add synthetic fields not in Kaggle data
        import numpy as np
        n = len(processed)
        np.random.seed(42)
        
        processed['description_length'] = (processed['name'].fillna('').str.len() * 
                                           np.random.randint(10, 50, n))
        processed['num_updates'] = np.random.randint(0, 15, n)
        
        if 'backers_count' in processed.columns:
            processed['num_comments'] = (processed['backers_count'].fillna(0) * 
                                        np.random.uniform(0.02, 0.08, n)).astype(int)
        else:
            processed['num_comments'] = np.random.randint(0, 50, n)
        
        def gen_tiers(goal):
            if pd.isna(goal) or goal <= 0:
                return str([25, 50, 100])
            base = max(10, int(goal / 500))
            return str([max(10, base), max(25, base*2), max(50, base*4), max(100, base*8)])
        
        if 'funding_goal' in processed.columns:
            processed['reward_tiers'] = processed['funding_goal'].apply(gen_tiers)
        
        # Save
        output_path = base_dir / config.get('data', {}).get('raw_dir', 'data/raw')
        output_file = output_path / config.get('data', {}).get('raw_output', 'kickstarter_raw.csv')
        
        processed.to_csv(output_file, index=False)
        
        print()
        print("=" * 60)
        print("MERGE COMPLETE")
        print("=" * 60)
        print(f"Total campaigns: {len(processed)}")
        if 'is_successful' in processed.columns:
            print(f"Successful: {processed['is_successful'].sum()}")
            print(f"Failed: {(~processed['is_successful'].astype(bool)).sum()}")
        if 'category' in processed.columns:
            print(f"Categories: {processed['category'].nunique()}")
        print(f"Saved to: {output_file}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
