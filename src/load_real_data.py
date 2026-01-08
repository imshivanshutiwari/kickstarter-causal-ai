"""
Kickstarter Data Loader

This module provides functions to load and process Kickstarter data from various sources:
1. Kaggle datasets (recommended - must be downloaded manually)
2. Webrobots.io archives
3. Synthetic data fallback

GETTING REAL DATA:
==================

Since Kickstarter blocks direct scraping and the Webrobots.io S3 bucket requires 
authentication, you need to download data manually from Kaggle:

Option 1: Kaggle (Recommended)
------------------------------
1. Go to: https://www.kaggle.com/datasets/kemical/kickstarter-projects
2. Click "Download" (requires free Kaggle account)
3. Extract the zip file
4. Place the CSV file(s) in: data/raw/kaggle/

Option 2: Alternative Kaggle Dataset
------------------------------------
1. Go to: https://www.kaggle.com/datasets/domingosun/2025-crowdfunding-kickstarter-648400-projects
2. Download and extract
3. Place in: data/raw/kaggle/

After downloading, run:
    python src/load_real_data.py

Author: Kickstarter Counterfactual Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_kickstarter_csv(data_dir: Path = None) -> Path:
    """
    Find Kickstarter CSV files in the data directory.
    
    Searches in order:
    1. data/raw/kaggle/
    2. data/raw/
    3. data/
    
    Returns:
        Path to the first CSV file found, or None
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    
    search_dirs = [
        data_dir / "raw" / "kaggle",
        data_dir / "raw",
        data_dir,
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Look for Kickstarter CSVs
        patterns = ["ks-projects*.csv", "kickstarter*.csv", "Kickstarter*.csv", "*.csv"]
        for pattern in patterns:
            files = list(search_dir.glob(pattern))
            if files:
                # Return largest file (likely the main dataset)
                largest = max(files, key=lambda f: f.stat().st_size)
                logger.info(f"Found Kickstarter data: {largest}")
                return largest
    
    return None


def load_kaggle_kickstarter(csv_path: Path) -> pd.DataFrame:
    """
    Load and process Kickstarter data from Kaggle CSV.
    
    Kaggle dataset columns typically include:
    - ID, name, category, main_category
    - currency, deadline, goal, launched
    - pledged, state, backers, country
    - usd_pledged, usd_pledged_real, usd_goal_real
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Processed DataFrame in standard format
    """
    logger.info(f"Loading data from {csv_path}...")
    
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not read {csv_path} with any encoding")
    
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Create standard format DataFrame
    processed = pd.DataFrame()
    
    # Name
    if 'name' in df.columns:
        processed['name'] = df['name']
    
    # Funding goal (prefer USD columns)
    if 'usd_goal_real' in df.columns:
        processed['funding_goal'] = pd.to_numeric(df['usd_goal_real'], errors='coerce')
    elif 'goal' in df.columns:
        processed['funding_goal'] = pd.to_numeric(df['goal'], errors='coerce')
    
    # Pledged amount (prefer USD columns)
    if 'usd_pledged_real' in df.columns:
        processed['pledged_amount'] = pd.to_numeric(df['usd_pledged_real'], errors='coerce')
    elif 'usd pledged' in df.columns:
        processed['pledged_amount'] = pd.to_numeric(df['usd pledged'], errors='coerce')
    elif 'pledged' in df.columns:
        processed['pledged_amount'] = pd.to_numeric(df['pledged'], errors='coerce')
    
    # Backers
    if 'backers' in df.columns:
        processed['backers_count'] = pd.to_numeric(df['backers'], errors='coerce')
    elif 'backers_count' in df.columns:
        processed['backers_count'] = pd.to_numeric(df['backers_count'], errors='coerce')
    
    # Status
    if 'state' in df.columns:
        processed['status'] = df['state'].str.lower()
    
    # Category
    if 'main_category' in df.columns:
        processed['category'] = df['main_category']
    elif 'category' in df.columns:
        processed['category'] = df['category']
    
    # Subcategory
    if 'category' in df.columns and 'main_category' in df.columns:
        processed['subcategory'] = df['category']
    
    # Country
    if 'country' in df.columns:
        processed['country'] = df['country']
    
    # Dates
    if 'launched' in df.columns:
        processed['launch_date'] = pd.to_datetime(df['launched'], errors='coerce')
    
    if 'deadline' in df.columns:
        processed['end_date'] = pd.to_datetime(df['deadline'], errors='coerce')
    
    # Calculate duration
    if 'launch_date' in processed.columns and 'end_date' in processed.columns:
        processed['campaign_duration_days'] = (processed['end_date'] - processed['launch_date']).dt.days
    
    # Filter to completed campaigns
    if 'status' in processed.columns:
        valid_states = ['successful', 'failed']
        processed = processed[processed['status'].isin(valid_states)]
        logger.info(f"Filtered to {len(processed)} completed campaigns")
    
    # Remove invalid rows
    processed = processed.dropna(subset=['funding_goal', 'pledged_amount'])
    processed = processed[processed['funding_goal'] > 0]
    
    if 'campaign_duration_days' in processed.columns:
        processed = processed[processed['campaign_duration_days'] > 0]
        processed = processed[processed['campaign_duration_days'] <= 90]
    
    # Add derived fields
    processed['is_successful'] = (processed['pledged_amount'] >= processed['funding_goal']).astype(int)
    processed['funding_ratio'] = processed['pledged_amount'] / processed['funding_goal']
    
    # Generate synthetic fields that aren't in Kaggle data
    n = len(processed)
    
    # Estimate updates (not available)
    processed['num_updates'] = np.random.randint(0, 15, n)
    
    # Estimate comments (fraction of backers)
    if 'backers_count' in processed.columns:
        processed['num_comments'] = (processed['backers_count'] * np.random.uniform(0.02, 0.08, n)).astype(int)
    else:
        processed['num_comments'] = np.random.randint(0, 50, n)
    
    # Estimate description length
    if 'name' in processed.columns:
        processed['description_length'] = processed['name'].str.len() * np.random.randint(10, 50, n)
    else:
        processed['description_length'] = np.random.randint(500, 5000, n)
    
    # Generate reward tiers (not in Kaggle data)
    def generate_reward_tiers(goal, pledged):
        if pd.isna(goal) or goal <= 0:
            return [25, 50, 100]
        
        # Estimate avg reward from pledged/backers if available
        avg_pledge = 50  # default
        
        n_tiers = np.random.randint(3, 8)
        base = max(10, int(goal / 1000))
        
        tiers = sorted(list(set([
            max(10, base + np.random.randint(-5, 10)),
            max(25, base * 2 + np.random.randint(-10, 20)),
            max(50, base * 4 + np.random.randint(-15, 30)),
            max(100, base * 8 + np.random.randint(-20, 50)),
            max(200, base * 15 + np.random.randint(-30, 100))
        ])))[:n_tiers]
        
        return tiers
    
    processed['reward_tiers'] = processed.apply(
        lambda x: generate_reward_tiers(x['funding_goal'], x['pledged_amount']), axis=1
    )
    
    logger.info(f"Processed {len(processed)} campaigns")
    
    return processed


def process_and_save(output_path: Path = None, sample_size: int = None) -> pd.DataFrame:
    """
    Load, process, and save Kickstarter data.
    
    Args:
        output_path: Where to save processed data
        sample_size: Optional limit on number of campaigns
        
    Returns:
        Processed DataFrame
    """
    base_dir = Path(__file__).parent.parent
    
    if output_path is None:
        output_path = base_dir / "data" / "raw" / "kickstarter_raw.csv"
    
    # Find CSV file
    csv_path = find_kickstarter_csv()
    
    if csv_path is None:
        print()
        print("=" * 60)
        print("NO KICKSTARTER DATA FOUND")
        print("=" * 60)
        print()
        print("Please download Kickstarter data from Kaggle:")
        print()
        print("1. Go to: https://www.kaggle.com/datasets/kemical/kickstarter-projects")
        print("2. Click 'Download' (requires free Kaggle account)")
        print("3. Extract the zip file")
        print("4. Copy the CSV to: data/raw/kaggle/")
        print()
        print("Then run this script again.")
        return None
    
    # Load and process
    df = load_kaggle_kickstarter(csv_path)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled to {len(df)} campaigns")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print()
    print("=" * 60)
    print("DATA LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"Source: {csv_path}")
    print(f"Total campaigns: {len(df)}")
    print(f"Successful: {df['is_successful'].sum()}")
    print(f"Failed: {(~df['is_successful'].astype(bool)).sum()}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Saved to: {output_path}")
    print()
    print("Category distribution:")
    print(df['category'].value_counts().head(10))
    print()
    print("Next step: Run the enrichment pipeline:")
    print("  python src/data_enrichment.py")
    
    return df


def main():
    """Main entry point."""
    print()
    print("=" * 60)
    print("KICKSTARTER REAL DATA LOADER")
    print("=" * 60)
    print()
    
    df = process_and_save(sample_size=1000)
    
    if df is not None:
        print("\n✅ Real data loaded successfully!")
    else:
        print("\n❌ Please download data from Kaggle first.")


if __name__ == "__main__":
    main()
