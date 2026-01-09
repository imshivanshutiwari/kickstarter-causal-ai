"""
Process Downloaded Kaggle Data
Converts real Kickstarter data to our standard format.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 60)
    print("PROCESSING REAL KICKSTARTER DATA")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    # Load
    input_file = base_dir / "data" / "raw" / "kaggle" / "kickstarter_projects.csv"
    df = pd.read_csv(input_file, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    
    print(f"Loaded {len(df)} campaigns")
    
    # Filter to completed campaigns
    df = df[df['state'].isin(['Failed', 'Successful'])]
    df['status'] = df['state'].str.lower()
    print(f"Completed campaigns: {len(df)}")
    
    # Rename columns
    rename_map = {
        'id': 'id',
        'name': 'name',
        'category': 'subcategory',
        'subcategory': 'category',
        'country': 'country',
        'goal': 'funding_goal',
        'pledged': 'pledged_amount',
        'backers': 'backers_count'
    }
    df = df.rename(columns=rename_map)
    
    # Parse dates
    df['launch_date'] = pd.to_datetime(df['launched'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['deadline'], errors='coerce')
    df['campaign_duration_days'] = (df['end_date'] - df['launch_date']).dt.days
    
    # Filter valid
    df = df[(df['campaign_duration_days'] > 0) & (df['campaign_duration_days'] <= 90)]
    df = df[df['funding_goal'] > 0]
    df = df[df['pledged_amount'] >= 0]
    
    # --- MERGE LIVE DATA ---
    print("Checking for live scraped data...")
    live_dfs = []
    
    # 1. Kickstarter Live
    ks_live_path = base_dir / "data" / "raw" / "kickstarter_live.csv"
    if ks_live_path.exists():
        print(f"  Found Kickstarter Live data: {ks_live_path}")
        ks_live = pd.read_csv(ks_live_path)
        ks_live['source'] = 'kickstarter_live'
        live_dfs.append(ks_live)
        
    # 2. Indiegogo Live
    igg_live_path = base_dir / "data" / "raw" / "indiegogo_live.csv"
    if igg_live_path.exists():
        print(f"  Found Indiegogo Live data: {igg_live_path}")
        igg_live = pd.read_csv(igg_live_path)
        igg_live['source'] = 'indiegogo_live'
        # Indiegogo structure is a bit different, Ensure mappings
        # In scraper we used: name, category, funding_goal, pledged_amount, backers_count, status, campaign_duration_days
        live_dfs.append(igg_live)
        
    if live_dfs:
        df_live = pd.concat(live_dfs, ignore_index=True)
        # Harmonize columns for live data
        # Live data already uses standard names from our scrapers, but ensuring 'status' is lowercase
        df_live['status'] = df_live['status'].str.lower()
        df_live['is_successful'] = (df_live['status'] == 'successful').astype(int)
        
        # Add missing columns with defaults
        for col in ['country', 'subcategory', 'launch_date', 'end_date']:
            if col not in df_live.columns:
                df_live[col] = None # Or reasonable defaults
                
        print(f"  Merging {len(df_live)} live campaigns...")
        df['source'] = 'kaggle' # Mark origin
        
        # Combine!
        # Align columns
        common_cols = list(set(df.columns) & set(df_live.columns))
        df = pd.concat([df[common_cols], df_live[common_cols]], ignore_index=True)
        print(f"Total campaigns after merge: {len(df)}")
    
    # --- END MERGE ---
    
    print(f"After filtering and merging: {len(df)} campaigns")
    
    # Calculate metrics (Re-calc for safety)
    df['is_successful'] = (df['status'] == 'successful').astype(int)
    df['funding_ratio'] = df['pledged_amount'] / df['funding_goal'].replace(0, 1)
    
    # Sample 2000 for manageable size (Prioritize live data if needed, or just mix)
    # Let's keep ALL live data and sample from Kaggle if needed? 
    # For now, just simple sample to keep size consistent for user speed
    df_sample = df.sample(n=min(2500, len(df)), random_state=42).copy()
    
    # Generate synthetic fields not in Kaggle data
    n = len(df_sample)
    np.random.seed(42)
    
    # Ensure name is string
    df_sample['name'] = df_sample['name'].astype(str).fillna("Unknown Project")
    
    df_sample['description_length'] = df_sample['name'].str.len() * np.random.randint(10, 50, n)
    df_sample['num_updates'] = np.random.randint(0, 15, n)
    df_sample['num_comments'] = (df_sample['backers_count'].fillna(0) * np.random.uniform(0.02, 0.08, n)).astype(int)
    
    def gen_tiers(goal):
        base = max(10, int(goal / 500))
        return str([max(10, base), max(25, base*2), max(50, base*4), max(100, base*8)])
    
    df_sample['reward_tiers'] = df_sample['funding_goal'].apply(gen_tiers)
    
    # Save
    output_file = base_dir / "data" / "raw" / "kickstarter_raw.csv"
    df_sample.to_csv(output_file, index=False)
    
    print()
    print("=" * 60)
    print("REAL DATA SAVED")
    print("=" * 60)
    print(f"Total campaigns: {len(df_sample)}")
    print(f"Successful: {df_sample['is_successful'].sum()}")
    print(f"Failed: {(~df_sample['is_successful'].astype(bool)).sum()}")
    print(f"Categories: {df_sample['category'].nunique()}")
    
    min_date = df_sample['launch_date'].min()
    max_date = df_sample['launch_date'].max()
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    
    print()
    print("Category distribution:")
    print(df_sample['category'].value_counts().head(10))
    print()
    
    # --- SOURCE BREAKDOWN ---
    if 'source' in df_sample.columns:
        print("Data Source Breakdown:")
        print(df_sample['source'].value_counts())
        print()
    # ------------------------
    
    print(f"Saved to: {output_file}")
    print()
    print("Next step: Run enrichment pipeline")
    print("  python src/data_enrichment.py")

if __name__ == "__main__":
    main()
