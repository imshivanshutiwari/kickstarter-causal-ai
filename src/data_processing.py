"""
Data Processing Module for Kickstarter Counterfactual Analysis

This module handles data cleaning, transformation, and feature enrichment
for the scraped Kickstarter campaign data.

Functions:
- load_raw_data: Load raw scraped data
- clean_data: Clean and validate data
- enrich_with_trends: Add Google Trends data
- create_causal_features: Engineer features for causal inference
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
import json

# Try to import pytrends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw Kickstarter data from CSV.
    
    Args:
        filepath: Path to CSV file (uses default if None)
        
    Returns:
        DataFrame with raw campaign data
    """
    if filepath is None:
        base_dir = Path(__file__).parent.parent
        filepath = base_dir / "data" / "raw" / "kickstarter_raw.csv"
        
        # Try alternate filename
        if not Path(filepath).exists():
            filepath = base_dir / "data" / "raw" / "kickstarter_raw_data.csv"
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} campaigns from {filepath}")
    return df


# =============================================================================
# Data Cleaning
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate campaign data.
    
    Operations:
    - Remove duplicates
    - Handle missing values
    - Convert data types
    - Validate ranges
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    original_len = len(df)
    
    # Remove duplicates
    if 'project_id' in df.columns:
        df = df.drop_duplicates(subset=['project_id'], keep='first')
    
    # Convert dates
    date_cols = ['launch_date', 'end_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = ['funding_goal', 'pledged_amount', 'backers_count', 
                    'campaign_duration_days', 'num_updates', 'num_comments']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid funding goals
    if 'funding_goal' in df.columns:
        df = df[df['funding_goal'] > 0]
    
    # Remove rows with negative pledged amounts
    if 'pledged_amount' in df.columns:
        df = df[df['pledged_amount'] >= 0]
    
    # Fill missing values
    fill_zeros = ['num_updates', 'num_comments', 'backers_count']
    for col in fill_zeros:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    if 'description_length' in df.columns:
        df['description_length'] = df['description_length'].fillna(df['description_length'].median())
    
    # Parse reward_tiers JSON if needed
    if 'reward_tiers' in df.columns:
        df['reward_tiers_list'] = df['reward_tiers'].apply(_parse_reward_tiers)
        df['num_reward_tiers'] = df['reward_tiers_list'].apply(len)
        df['min_reward_price'] = df['reward_tiers_list'].apply(
            lambda x: min(x) if x else 0
        )
        df['max_reward_price'] = df['reward_tiers_list'].apply(
            lambda x: max(x) if x else 0
        )
        df['avg_reward_price'] = df['reward_tiers_list'].apply(
            lambda x: np.mean(x) if x else 0
        )
    
    removed = original_len - len(df)
    logger.info(f"Cleaned data: removed {removed} rows, {len(df)} remaining")
    
    return df


def _parse_reward_tiers(val) -> List[float]:
    """Parse reward tiers from JSON string."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


# =============================================================================
# Feature Enrichment
# =============================================================================

def enrich_with_trends(df: pd.DataFrame, keywords: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Enrich data with Google Trends information.
    
    Adds trend_index column showing search interest during campaign period.
    
    Args:
        df: Campaign DataFrame
        keywords: Keywords to search (uses category names if None)
        
    Returns:
        DataFrame with trend_index column
    """
    if not PYTRENDS_AVAILABLE:
        logger.warning("pytrends not available, using synthetic trend data")
        df['trend_index'] = np.random.randint(20, 80, size=len(df))
        return df
    
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Use category as keyword
        if 'category' in df.columns:
            keyword = df['category'].mode()[0] if len(df) > 0 else 'kickstarter'
        else:
            keyword = 'kickstarter'
        
        # Get trends for the period
        pytrends.build_payload([keyword], cat=0, timeframe='today 5-y')
        trends_df = pytrends.interest_over_time()
        
        if not trends_df.empty:
            df['trend_index'] = df['launch_date'].apply(
                lambda x: _get_trend_value(x, trends_df, keyword)
            )
        else:
            df['trend_index'] = 50
            
    except Exception as e:
        logger.warning(f"Failed to get trends: {e}")
        df['trend_index'] = np.random.randint(30, 70, size=len(df))
    
    return df


def _get_trend_value(date, trends_df: pd.DataFrame, keyword: str) -> int:
    """Get trend value for a specific date."""
    if pd.isna(date):
        return 50
    try:
        if keyword in trends_df.columns:
            closest_date = trends_df.index[trends_df.index.get_indexer([date], method='nearest')[0]]
            return int(trends_df.loc[closest_date, keyword])
    except Exception:
        pass
    return 50


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add competition-related features.
    
    Features added:
    - concurrent_campaigns: Number of campaigns launched within ±7 days
    
    Args:
        df: Campaign DataFrame
        
    Returns:
        DataFrame with competition features
    """
    if 'launch_date' not in df.columns:
        logger.warning("launch_date not found, skipping competition features")
        return df
    
    # Ensure datetime
    df['launch_date'] = pd.to_datetime(df['launch_date'])
    
    # Count concurrent campaigns
    def count_concurrent(row):
        date = row['launch_date']
        if pd.isna(date):
            return 10
        
        window_start = date - timedelta(days=7)
        window_end = date + timedelta(days=7)
        
        concurrent = df[
            (df['launch_date'] >= window_start) & 
            (df['launch_date'] <= window_end) &
            (df['category'] == row.get('category', ''))
        ]
        return len(concurrent) - 1  # Exclude self
    
    df['concurrent_campaigns'] = df.apply(count_concurrent, axis=1)
    
    return df


# =============================================================================
# Causal Feature Engineering
# =============================================================================

def create_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features specifically designed for causal inference.
    
    Features added:
    - Treatment variables (endogenous)
    - Instrumental variables (exogenous)
    - Outcome variables
    - Confounders
    
    Args:
        df: Campaign DataFrame
        
    Returns:
        DataFrame with causal features
    """
    # Ensure datetime columns
    if 'launch_date' in df.columns:
        df['launch_date'] = pd.to_datetime(df['launch_date'])
        df['day_of_week'] = df['launch_date'].dt.dayofweek
        df['month'] = df['launch_date'].dt.month
        df['year'] = df['launch_date'].dt.year
    
    # Holiday proximity (within 7 days of major holiday)
    df['holiday_proximity'] = df['month'].apply(
        lambda m: 1 if m in [1, 11, 12] else 0
    )
    
    # Status as binary
    if 'status' in df.columns:
        df['is_successful'] = (df['status'] == 'successful').astype(int)
    
    # Funding ratio (outcome)
    if 'funding_goal' in df.columns and 'pledged_amount' in df.columns:
        df['funding_ratio'] = df['pledged_amount'] / df['funding_goal'].replace(0, 1)
        df['funding_ratio'] = df['funding_ratio'].clip(0, 20)  # Cap extreme values
    
    # Goal ambition by category
    if 'category' in df.columns and 'funding_goal' in df.columns:
        df['goal_ambition'] = df.groupby('category')['funding_goal'].transform(
            lambda x: x / x.median()
        )
    
    # Price to goal ratio
    if 'avg_reward_price' in df.columns and 'funding_goal' in df.columns:
        df['price_to_goal_ratio'] = df['avg_reward_price'] / (df['funding_goal'] / 100).replace(0, 1)
    
    # Demand censoring indicator
    if 'funding_ratio' in df.columns:
        df['hit_funding_cap'] = (df['funding_ratio'] > 3.0).astype(int)
    
    # Backers per day
    if 'backers_count' in df.columns and 'campaign_duration_days' in df.columns:
        df['backers_per_day'] = df['backers_count'] / df['campaign_duration_days'].replace(0, 1)
    
    # Pledged per backer
    if 'pledged_amount' in df.columns and 'backers_count' in df.columns:
        df['pledged_per_backer'] = df['pledged_amount'] / df['backers_count'].replace(0, 1)
    
    logger.info(f"Created causal features. Total columns: {len(df.columns)}")
    
    return df


# =============================================================================
# Full Processing Pipeline
# =============================================================================

def process_data(input_path: Optional[str] = None, 
                 output_path: Optional[str] = None,
                 add_trends: bool = False) -> pd.DataFrame:
    """
    Run full data processing pipeline.
    
    Steps:
    1. Load raw data
    2. Clean data
    3. Add competition features
    4. Optionally add trend data
    5. Create causal features
    6. Save processed data
    
    Args:
        input_path: Path to raw data (uses default if None)
        output_path: Path for output (uses default if None)
        add_trends: Whether to add Google Trends data
        
    Returns:
        Processed DataFrame
    """
    base_dir = Path(__file__).parent.parent
    
    if output_path is None:
        output_path = base_dir / "data" / "processed" / "kickstarter_processed.csv"
    
    # Pipeline
    logger.info("Starting data processing pipeline...")
    
    df = load_raw_data(input_path)
    df = clean_data(df)
    df = add_competition_features(df)
    
    if add_trends:
        df = enrich_with_trends(df)
    
    df = create_causal_features(df)
    
    # Save processed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    return df


if __name__ == "__main__":
    df = process_data()
    print(f"\nProcessed {len(df)} campaigns")
    print(f"Columns: {list(df.columns)}")
