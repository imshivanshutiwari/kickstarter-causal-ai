"""
Data Enrichment Module for Kickstarter Campaigns

This module enriches the raw Kickstarter data with additional features:
1. Google Trends data (trend_index)
2. Competition proxy (concurrent_campaigns)
3. Seasonality features (month, day_of_week, is_holiday_week, quarter)
4. Reward tier analysis (num_reward_tiers, min/max/avg prices, price_spread)
5. Campaign characteristics (update_frequency, engagement_ratio)

Author: Kickstarter Counterfactual Project
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import time
from tqdm import tqdm

# Try to import pytrends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("Warning: pytrends not available. Using synthetic trend data.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Major US holidays for holiday proximity detection
US_HOLIDAYS = [
    # 2020
    (2020, 1, 1), (2020, 1, 20), (2020, 2, 17), (2020, 5, 25), (2020, 7, 4),
    (2020, 9, 7), (2020, 10, 12), (2020, 11, 11), (2020, 11, 26), (2020, 12, 25),
    # 2021
    (2021, 1, 1), (2021, 1, 18), (2021, 2, 15), (2021, 5, 31), (2021, 7, 4),
    (2021, 9, 6), (2021, 10, 11), (2021, 11, 11), (2021, 11, 25), (2021, 12, 25),
    # 2022
    (2022, 1, 1), (2022, 1, 17), (2022, 2, 21), (2022, 5, 30), (2022, 7, 4),
    (2022, 9, 5), (2022, 10, 10), (2022, 11, 11), (2022, 11, 24), (2022, 12, 25),
    # 2023
    (2023, 1, 1), (2023, 1, 16), (2023, 2, 20), (2023, 5, 29), (2023, 7, 4),
    (2023, 9, 4), (2023, 10, 9), (2023, 11, 11), (2023, 11, 23), (2023, 12, 25),
    # 2024
    (2024, 1, 1), (2024, 1, 15), (2024, 2, 19), (2024, 5, 27), (2024, 7, 4),
    (2024, 9, 2), (2024, 10, 14), (2024, 11, 11), (2024, 11, 28), (2024, 12, 25),
]

# Convert to datetime objects
HOLIDAY_DATES = [datetime(y, m, d) for y, m, d in US_HOLIDAYS]


# =============================================================================
# Google Trends Functions
# =============================================================================

def get_trend_index(category: str, launch_date: datetime, 
                   pytrends: Optional['TrendReq'] = None) -> Optional[int]:
    """
    Get Google Trends index for a category around a launch date.
    
    Args:
        category: Campaign category name
        launch_date: Campaign launch date
        pytrends: Optional pytrends instance for reuse
        
    Returns:
        Trend index (0-100) or None if failed
    """
    if not PYTRENDS_AVAILABLE:
        return None
    
    if pytrends is None:
        pytrends = TrendReq(hl='en-US', tz=360)
    
    try:
        # Map category to search keyword
        keyword = category.lower().replace(' & ', ' ').replace('_', ' ')
        
        # Build timeframe (1 month around launch date)
        start = (launch_date - timedelta(days=14)).strftime('%Y-%m-%d')
        end = (launch_date + timedelta(days=14)).strftime('%Y-%m-%d')
        timeframe = f'{start} {end}'
        
        # Get trend data
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe)
        trends_df = pytrends.interest_over_time()
        
        if not trends_df.empty and keyword in trends_df.columns:
            return int(trends_df[keyword].mean())
        
    except Exception as e:
        logger.debug(f"Trends fetch failed for {category}: {e}")
    
    return None


def batch_get_trends(df: pd.DataFrame, batch_size: int = 5) -> pd.Series:
    """
    Get Google Trends for all campaigns with rate limiting.
    
    Args:
        df: DataFrame with category and launch_date columns
        batch_size: How many to process before pausing
        
    Returns:
        Series of trend indices
    """
    if not PYTRENDS_AVAILABLE:
        logger.warning("pytrends not available, generating synthetic trends")
        return pd.Series(np.random.randint(20, 80, size=len(df)))
    
    trends = []
    pytrends = TrendReq(hl='en-US', tz=360, retries=2, backoff_factor=0.5)
    
    # Get unique category-month combinations to reduce API calls
    df['launch_month'] = pd.to_datetime(df['launch_date']).dt.to_period('M')
    unique_combos = df[['category', 'launch_month']].drop_duplicates()
    
    trend_cache = {}
    
    for idx, row in tqdm(unique_combos.iterrows(), total=len(unique_combos), 
                         desc="Fetching trends"):
        key = (row['category'], str(row['launch_month']))
        
        # Use mid-month date for the query
        mid_date = row['launch_month'].to_timestamp() + timedelta(days=15)
        trend_val = get_trend_index(row['category'], mid_date, pytrends)
        
        if trend_val is not None:
            trend_cache[key] = trend_val
        else:
            # Use category median or default
            trend_cache[key] = 50
        
        # Rate limiting
        if (idx + 1) % batch_size == 0:
            time.sleep(2)
    
    # Map back to all campaigns
    for _, row in df.iterrows():
        key = (row['category'], str(row['launch_month']))
        trends.append(trend_cache.get(key, 50))
    
    return pd.Series(trends)


# =============================================================================
# Competition Features
# =============================================================================

def calculate_concurrent_campaigns(df: pd.DataFrame, days_window: int = 14) -> pd.Series:
    """
    Count campaigns in same category launched within ±N days.
    
    Args:
        df: DataFrame with category and launch_date columns
        days_window: Days before/after to consider
        
    Returns:
        Series of concurrent campaign counts
    """
    logger.info(f"Calculating concurrent campaigns (±{days_window} days)...")
    
    df['launch_date'] = pd.to_datetime(df['launch_date'])
    concurrent = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing competition"):
        date = row['launch_date']
        category = row['category']
        
        if pd.isna(date):
            concurrent.append(0)
            continue
        
        window_start = date - timedelta(days=days_window)
        window_end = date + timedelta(days=days_window)
        
        # Count campaigns in same category within window
        count = df[
            (df['category'] == category) &
            (df['launch_date'] >= window_start) &
            (df['launch_date'] <= window_end) &
            (df.index != idx)
        ].shape[0]
        
        concurrent.append(count)
    
    return pd.Series(concurrent)


# =============================================================================
# Seasonality Features
# =============================================================================

def is_holiday_week(date: datetime, days_proximity: int = 7) -> int:
    """
    Check if date is within N days of a major US holiday.
    
    Args:
        date: Date to check
        days_proximity: Days proximity to consider
        
    Returns:
        1 if near holiday, 0 otherwise
    """
    if pd.isna(date):
        return 0
    
    for holiday in HOLIDAY_DATES:
        if abs((date - holiday).days) <= days_proximity:
            return 1
    return 0


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all seasonality features to DataFrame.
    
    Features added:
    - month (1-12)
    - day_of_week (0=Monday, 6=Sunday)
    - is_holiday_week (0 or 1)
    - quarter (Q1, Q2, Q3, Q4)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with seasonality features
    """
    logger.info("Adding seasonality features...")
    
    df['launch_date'] = pd.to_datetime(df['launch_date'])
    
    # Basic extractions
    df['month'] = df['launch_date'].dt.month
    df['day_of_week'] = df['launch_date'].dt.dayofweek
    df['quarter'] = 'Q' + df['launch_date'].dt.quarter.astype(str)
    
    # Holiday proximity
    df['is_holiday_week'] = df['launch_date'].apply(is_holiday_week)
    
    logger.info(f"  Campaigns near holidays: {df['is_holiday_week'].sum()}")
    
    return df


# =============================================================================
# Reward Tier Analysis
# =============================================================================

def parse_reward_tiers(val) -> List[float]:
    """Parse reward tiers from various formats."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [float(x) for x in val if x > 0]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [float(x) for x in parsed if x > 0]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def add_reward_tier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate reward tier statistics.
    
    Features added:
    - num_reward_tiers
    - min_reward_price
    - max_reward_price
    - avg_reward_price
    - price_spread (max - min)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with reward tier features
    """
    logger.info("Analyzing reward tiers...")
    
    # Parse reward tiers
    if 'reward_tiers' in df.columns:
        df['reward_tiers_list'] = df['reward_tiers'].apply(parse_reward_tiers)
    else:
        df['reward_tiers_list'] = [[] for _ in range(len(df))]
    
    # Calculate statistics
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
    df['price_spread'] = df['max_reward_price'] - df['min_reward_price']
    
    # Drop intermediate column
    df = df.drop(columns=['reward_tiers_list'], errors='ignore')
    
    logger.info(f"  Campaigns with reward tiers: {(df['num_reward_tiers'] > 0).sum()}")
    
    return df


# =============================================================================
# Campaign Characteristics
# =============================================================================

def add_campaign_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived campaign characteristics.
    
    Features added:
    - update_frequency (updates per day)
    - engagement_ratio (comments per backer)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with campaign characteristics
    """
    logger.info("Calculating campaign characteristics...")
    
    # Update frequency
    duration_col = 'campaign_duration_days' if 'campaign_duration_days' in df.columns else 'duration_days'
    updates_col = 'num_updates' if 'num_updates' in df.columns else 'updates_count'
    
    if duration_col in df.columns and updates_col in df.columns:
        df['update_frequency'] = df[updates_col] / df[duration_col].replace(0, 1)
    else:
        df['update_frequency'] = 0
    
    # Engagement ratio
    comments_col = 'num_comments' if 'num_comments' in df.columns else 'comments_count'
    backers_col = 'backers_count'
    
    if comments_col in df.columns and backers_col in df.columns:
        df['engagement_ratio'] = df[comments_col] / df[backers_col].replace(0, 1)
    else:
        df['engagement_ratio'] = 0
    
    return df


# =============================================================================
# Main Enrichment Pipeline
# =============================================================================

def generate_enrichment_report(df: pd.DataFrame, trends_missing: int, 
                               output_path: Path) -> str:
    """
    Generate a text report about the enrichment process.
    
    Args:
        df: Enriched DataFrame
        trends_missing: Number of campaigns with missing trends
        output_path: Path to save report
        
    Returns:
        Report text
    """
    report_lines = [
        "=" * 60,
        "KICKSTARTER DATA ENRICHMENT REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "OVERVIEW",
        "-" * 40,
        f"Total campaigns enriched: {len(df)}",
        f"Campaigns with missing Google Trends: {trends_missing}",
        f"Campaigns near holidays: {df.get('is_holiday_week', pd.Series([0])).sum()}",
        f"Campaigns with reward tier data: {(df.get('num_reward_tiers', pd.Series([0])) > 0).sum()}",
        "",
        "NEW VARIABLES SUMMARY",
        "-" * 40,
    ]
    
    # Summary statistics for new features
    new_features = [
        'trend_index', 'concurrent_campaigns', 'month', 'day_of_week',
        'is_holiday_week', 'num_reward_tiers', 'min_reward_price', 
        'max_reward_price', 'avg_reward_price', 'price_spread',
        'update_frequency', 'engagement_ratio'
    ]
    
    for feature in new_features:
        if feature in df.columns:
            stats = df[feature].describe()
            report_lines.append(f"\n{feature}:")
            report_lines.append(f"  Count: {stats['count']:.0f}")
            report_lines.append(f"  Mean:  {stats['mean']:.2f}")
            report_lines.append(f"  Std:   {stats['std']:.2f}")
            report_lines.append(f"  Min:   {stats['min']:.2f}")
            report_lines.append(f"  Max:   {stats['max']:.2f}")
    
    report_lines.extend([
        "",
        "QUARTER DISTRIBUTION",
        "-" * 40,
    ])
    
    if 'quarter' in df.columns:
        quarter_counts = df['quarter'].value_counts().sort_index()
        for q, count in quarter_counts.items():
            report_lines.append(f"  {q}: {count} campaigns ({count/len(df)*100:.1f}%)")
    
    report_lines.extend([
        "",
        "DAY OF WEEK DISTRIBUTION",
        "-" * 40,
    ])
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if 'day_of_week' in df.columns:
        dow_counts = df['day_of_week'].value_counts().sort_index()
        for dow, count in dow_counts.items():
            report_lines.append(f"  {day_names[dow]}: {count} campaigns")
    
    report_lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def enrich_data(input_path: Optional[str] = None,
                output_path: Optional[str] = None,
                report_path: Optional[str] = None,
                use_google_trends: bool = False) -> pd.DataFrame:
    """
    Run the full data enrichment pipeline.
    
    Args:
        input_path: Path to raw data
        output_path: Path for enriched data output
        report_path: Path for enrichment report
        use_google_trends: Whether to fetch real Google Trends (slow)
        
    Returns:
        Enriched DataFrame
    """
    base_dir = Path(__file__).parent.parent
    
    # Set default paths
    if input_path is None:
        input_path = base_dir / "data" / "raw" / "kickstarter_raw.csv"
        if not Path(input_path).exists():
            input_path = base_dir / "data" / "raw" / "kickstarter_raw_data.csv"
    
    if output_path is None:
        output_path = base_dir / "data" / "processed" / "kickstarter_enriched.csv"
    
    if report_path is None:
        report_path = base_dir / "data" / "processed" / "enrichment_report.txt"
    
    # Load data
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} campaigns")
    
    # Track trends missing count
    trends_missing = 0
    
    # 1. Google Trends
    logger.info("\n1. Processing Google Trends...")
    if use_google_trends and PYTRENDS_AVAILABLE:
        df['trend_index'] = batch_get_trends(df)
        trends_missing = df['trend_index'].isna().sum()
        df['trend_index'] = df['trend_index'].fillna(df.groupby('category')['trend_index'].transform('median'))
    else:
        # Use synthetic trends if not available or not requested
        if 'trend_index' not in df.columns:
            np.random.seed(42)
            df['trend_index'] = np.random.randint(20, 80, size=len(df))
        trends_missing = 0
        logger.info("  Using existing/synthetic trend data")
    
    # 2. Competition proxy
    logger.info("\n2. Processing competition features...")
    if 'concurrent_campaigns' not in df.columns:
        df['concurrent_campaigns'] = calculate_concurrent_campaigns(df, days_window=14)
    else:
        logger.info("  concurrent_campaigns already exists, skipping")
    
    # 3. Seasonality features
    logger.info("\n3. Processing seasonality features...")
    df = add_seasonality_features(df)
    
    # 4. Reward tier analysis
    logger.info("\n4. Processing reward tier analysis...")
    df = add_reward_tier_features(df)
    
    # 5. Campaign characteristics
    logger.info("\n5. Processing campaign characteristics...")
    df = add_campaign_characteristics(df)
    
    # Save enriched data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved enriched data to {output_path}")
    
    # Generate report
    report = generate_enrichment_report(df, trends_missing, Path(report_path))
    logger.info(f"Saved enrichment report to {report_path}")
    
    return df


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run data enrichment and display results."""
    print("=" * 60)
    print("KICKSTARTER DATA ENRICHMENT")
    print("=" * 60)
    
    # Run enrichment
    df = enrich_data(use_google_trends=False)  # Set True for real API calls
    
    # Display summary
    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"\nTotal campaigns: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    print("\n--- New Features Added ---")
    new_cols = ['trend_index', 'concurrent_campaigns', 'month', 'day_of_week',
                'is_holiday_week', 'quarter', 'num_reward_tiers', 'min_reward_price',
                'max_reward_price', 'avg_reward_price', 'price_spread',
                'update_frequency', 'engagement_ratio']
    
    for col in new_cols:
        if col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                print(f"\n{col}:")
                print(f"  Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
                print(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            else:
                print(f"\n{col}: {df[col].nunique()} unique values")
    
    print("\n" + "=" * 60)
    print("Data saved to: data/processed/kickstarter_enriched.csv")
    print("Report saved to: data/processed/enrichment_report.txt")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    main()
