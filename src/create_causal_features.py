"""
Create Causal Features from Enriched Data
=========================================
Generates all variables needed for causal inference:
- Outcome: funding_ratio
- Treatment: price_positioning, avg_reward_price
- Endogenous: goal_ambition
- Instruments: is_weekend_launch, holiday_proximity, trend_spike
- Confounders: competition_intensity, creator_experience_proxy
- Deep NLP: BERT embeddings of campaign text (New)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add src to path if needed
base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

def main():
    print("=" * 60)
    print("CREATING CAUSAL FEATURES FROM REAL DATA")
    print("=" * 60)
    
    # Load data
    input_path = base_dir / 'data' / 'processed' / 'kickstarter_enriched.csv'
    if not input_path.exists():
        # Fallback to raw if enriched doesn't exist (e.g. step skipped)
        print("Enriched data not found, trying raw...")
        input_path = base_dir / 'data' / 'raw' / 'kickstarter_raw.csv'
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} campaigns")
    
    dur_col = 'campaign_duration_days' if 'campaign_duration_days' in df.columns else 'duration_days'
    
    # Parse dates
    if 'launch_date' in df.columns:
        df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
    
    # 1. OUTCOME VARIABLES
    # --------------------
    df['funding_ratio'] = df['pledged_amount'] / df['funding_goal'].replace(0, 1)
    df['success_binary'] = (df['status'] == 'successful').astype(int)
    # Avoid division by zero
    df['backers_per_day'] = df['backers_count'] / df[dur_col].replace(0, 1)
    
    # 2. TREATMENT VARIABLES (Price)
    # --------------------
    # Normalized price relative to category
    cat_median_price = df.groupby('category')['avg_reward_price'].transform('median')
    df['price_positioning'] = df['avg_reward_price'] / cat_median_price.replace(0, 1)
    # Price relative to goal (ambition per unit)
    df['price_ambition'] = df['avg_reward_price'] / (df['funding_goal'] / 100).replace(0, 1)
    
    # 3. ENDOGENOUS VARIABLES (Goal)
    # --------------------
    cat_median_goal = df.groupby('category')['funding_goal'].transform('median')
    df['goal_ambition'] = df['funding_goal'] / cat_median_goal.replace(0, 1)
    
    # 4. INSTRUMENTAL VARIABLES (External Shocks)
    # --------------------
    if 'launch_date' in df.columns and df['launch_date'].notna().any():
        df['launch_day_of_week'] = df['launch_date'].dt.dayofweek
    else:
        df['launch_day_of_week'] = df.get('day_of_week', 0)
    
    df['is_weekend_launch'] = (df['launch_day_of_week'] >= 5).astype(int)
    
    if 'is_holiday_week' not in df.columns:
        df['is_holiday_week'] = 0
    df['holiday_proximity'] = df['is_holiday_week']
    
    if 'trend_index' in df.columns:
        median_trend = df['trend_index'].median()
        df['trend_spike'] = (df['trend_index'] > median_trend * 1.2).astype(int)
    else:
        df['trend_index'] = 50
        df['trend_spike'] = 0
    
    # 5. CONFOUNDERS (Controls)
    # --------------------
    if 'concurrent_campaigns' in df.columns:
        mean_comp = df['concurrent_campaigns'].mean()
        df['competition_intensity'] = df['concurrent_campaigns'] / mean_comp if mean_comp > 0 else 1
    else:
        df['concurrent_campaigns'] = 10
        df['competition_intensity'] = 1.0
        
    df['creator_experience_proxy'] = df['description_length'] if 'description_length' in df.columns else 100
    
    # 6. DEEP NLP FEATURES (New)
    # --------------------
    try:
        from src.nlp_features import add_nlp_features
        print("Generating Deep NLP embeddings (BERT) from Campaign Names...")
        # Use 'name' as description is not in this dataset
        df, nlp_cols = add_nlp_features(df, text_cols=['name'], n_components=10)
        print(f"Added {len(nlp_cols)} NLP embedding features")
    except ImportError:
        print("NLP module not found or dependencies missing (sentence-transformers). Skipping.")
        nlp_cols = []
    except Exception as e:
        print(f"Failed to generate NLP features: {e}")
        nlp_cols = []

    # 7. CENSORING & OUTLIERS
    # --------------------
    df['demand_censored'] = (df['funding_ratio'] > 3.0).astype(int)
    df['is_outlier'] = ((df['funding_ratio'] > 10) | 
                        (df['avg_reward_price'] > 1000) | 
                        (df['backers_count'] > 10000)).astype(int)
    
    # Save
    output_path = base_dir / 'data' / 'processed' / 'kickstarter_causal_features.csv'
    df.to_csv(output_path, index=False)
    
    print()
    print("=" * 60)
    print("CAUSAL FEATURES CREATED (WITH NLP)")
    print("=" * 60)
    print(f"Total campaigns: {len(df)}")
    print(f"Success rate: {df['success_binary'].mean()*100:.1f}%")
    print(f"NLP Features: {len(nlp_cols)} dims")
    print("Saved to: data/processed/kickstarter_causal_features.csv")

if __name__ == "__main__":
    main()
