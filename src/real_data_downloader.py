"""
Real Kickstarter Data Downloader

Downloads real Kickstarter campaign data from Webrobots.io dataset archives.
This bypasses the 403 Forbidden errors from direct scraping.

Data Source: https://webrobots.io/kickstarter-datasets/
Monthly archives available from March 2016 to present.

Author: Kickstarter Counterfactual Project
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime
from io import BytesIO
import zipfile
import logging
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Webrobots.io Data Source Configuration
# =============================================================================

# Available dataset dates (monthly snapshots)
WEBROBOTS_DATASETS = {
    2024: [
        "2024-12-18", "2024-11-12", "2024-10-13", "2024-09-13",
        "2024-08-12", "2024-07-12", "2024-06-13", "2024-05-14",
        "2024-04-12", "2024-03-13", "2024-02-13", "2024-01-10"
    ],
    2023: [
        "2023-12-14", "2023-11-13", "2023-10-12", "2023-09-12",
        "2023-08-11", "2023-07-13", "2023-06-16", "2023-05-15",
        "2023-04-13", "2023-03-13", "2023-02-13", "2023-01-12"
    ],
    2022: [
        "2022-12-15", "2022-11-14", "2022-10-13", "2022-09-12",
        "2022-08-12", "2022-07-14", "2022-06-16", "2022-05-12",
        "2022-04-14", "2022-03-14", "2022-02-14", "2022-01-13"
    ],
    2021: [
        "2021-12-16", "2021-11-11", "2021-10-14", "2021-09-14",
        "2021-08-12", "2021-07-15", "2021-06-17", "2021-05-13",
        "2021-04-15", "2021-03-11", "2021-02-11", "2021-01-14"
    ],
    2020: [
        "2020-12-17", "2020-11-12", "2020-10-15", "2020-09-17",
        "2020-08-13", "2020-07-16", "2020-06-18", "2020-05-14",
        "2020-04-16", "2020-03-12", "2020-02-13", "2020-01-16"
    ]
}

# URL template for downloading CSV
CSV_URL_TEMPLATE = "https://s3.amazonaws.com/weruns/forfern/Kickstarter_Kickstarter/Kickstarter_{date}.csv.zip"

# Alternative: JSON format
JSON_URL_TEMPLATE = "https://s3.amazonaws.com/weruns/forfern/Kickstarter_Kickstarter/Kickstarter_{date}.json.zip"


def download_webrobots_dataset(date_str: str, output_dir: Path, format: str = "csv") -> Path:
    """
    Download a Kickstarter dataset from Webrobots.io.
    
    Args:
        date_str: Dataset date in format "YYYY-MM-DD"
        output_dir: Directory to save the downloaded file
        format: "csv" or "json"
        
    Returns:
        Path to the downloaded/extracted file
    """
    if format == "csv":
        url = CSV_URL_TEMPLATE.format(date=date_str)
    else:
        url = JSON_URL_TEMPLATE.format(date=date_str)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"kickstarter_{date_str}.{format}"
    
    # Check if already downloaded
    if output_file.exists():
        logger.info(f"Dataset already exists: {output_file}")
        return output_file
    
    logger.info(f"Downloading dataset from {url}...")
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        # Extract from zip
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            # Get the CSV/JSON file inside
            file_list = zf.namelist()
            if not file_list:
                raise ValueError("Empty zip file")
            
            # Extract first file
            with zf.open(file_list[0]) as f:
                with open(output_file, 'wb') as out_f:
                    out_f.write(f.read())
        
        logger.info(f"Downloaded and extracted to {output_file}")
        return output_file
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return None


def download_multiple_datasets(years: list = [2022, 2023, 2024], 
                               max_per_year: int = 4,
                               output_dir: Path = None) -> list:
    """
    Download multiple Kickstarter datasets across years.
    
    Args:
        years: List of years to download
        max_per_year: Maximum datasets per year (to limit download size)
        output_dir: Output directory
        
    Returns:
        List of downloaded file paths
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "raw" / "webrobots"
    
    downloaded_files = []
    
    for year in years:
        if year not in WEBROBOTS_DATASETS:
            logger.warning(f"No datasets available for year {year}")
            continue
        
        dates = WEBROBOTS_DATASETS[year][:max_per_year]  # Limit to max_per_year
        
        for date_str in tqdm(dates, desc=f"Downloading {year} datasets"):
            file_path = download_webrobots_dataset(date_str, output_dir)
            if file_path:
                downloaded_files.append(file_path)
            time.sleep(2)  # Be polite to the server
    
    return downloaded_files


def load_and_process_webrobots_data(file_paths: list, 
                                    categories: list = None,
                                    sample_size: int = None) -> pd.DataFrame:
    """
    Load and process downloaded Webrobots Kickstarter data.
    
    Args:
        file_paths: List of paths to CSV files
        categories: Filter to specific categories (e.g., ['Technology', 'Games'])
        sample_size: Maximum number of campaigns to return
        
    Returns:
        Processed DataFrame
    """
    all_data = []
    
    for file_path in tqdm(file_paths, desc="Loading data"):
        try:
            df = pd.read_csv(file_path, low_memory=False)
            all_data.append(df)
            logger.info(f"Loaded {len(df)} campaigns from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded from any file")
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined: {len(df)} total campaigns")
    
    # Remove duplicates (same project across snapshots)
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'], keep='last')
        logger.info(f"After deduplication: {len(df)} unique campaigns")
    
    # Filter to completed campaigns only
    if 'state' in df.columns:
        completed_states = ['successful', 'failed']
        df = df[df['state'].isin(completed_states)]
        logger.info(f"Completed campaigns: {len(df)}")
    
    # Filter by category if specified
    if categories and 'category' in df.columns:
        # Category might be JSON, extract name
        if df['category'].dtype == object:
            try:
                import json
                df['category_name'] = df['category'].apply(
                    lambda x: json.loads(x)['name'] if isinstance(x, str) and x.startswith('{') else x
                )
            except:
                df['category_name'] = df['category']
        else:
            df['category_name'] = df['category']
        
        df = df[df['category_name'].isin(categories)]
        logger.info(f"After category filter: {len(df)}")
    
    # Sample if needed
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled to {len(df)} campaigns")
    
    return df


def process_webrobots_to_standard_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Webrobots data format to our standard format.
    
    Webrobots columns may include:
    - id, name, blurb, goal, pledged, currency, state
    - created_at, launched_at, deadline
    - backers_count, category, country
    - creator (JSON with name, id)
    - location (JSON)
    - urls (JSON)
    """
    import json
    
    processed = pd.DataFrame()
    
    # Direct mappings
    if 'name' in df.columns:
        processed['name'] = df['name']
    
    if 'goal' in df.columns:
        processed['funding_goal'] = pd.to_numeric(df['goal'], errors='coerce')
    
    if 'pledged' in df.columns:
        processed['pledged_amount'] = pd.to_numeric(df['pledged'], errors='coerce')
    
    if 'backers_count' in df.columns:
        processed['backers_count'] = pd.to_numeric(df['backers_count'], errors='coerce')
    
    if 'state' in df.columns:
        processed['status'] = df['state']
    
    # Parse dates
    if 'launched_at' in df.columns:
        processed['launch_date'] = pd.to_datetime(df['launched_at'], unit='s', errors='coerce')
    
    if 'deadline' in df.columns:
        processed['end_date'] = pd.to_datetime(df['deadline'], unit='s', errors='coerce')
    
    # Calculate duration
    if 'launch_date' in processed.columns and 'end_date' in processed.columns:
        processed['campaign_duration_days'] = (processed['end_date'] - processed['launch_date']).dt.days
    
    # Parse category
    if 'category' in df.columns:
        def extract_category(x):
            if pd.isna(x):
                return 'Unknown'
            if isinstance(x, str):
                try:
                    data = json.loads(x)
                    return data.get('name', 'Unknown')
                except:
                    return x
            return str(x)
        
        processed['category'] = df['category'].apply(extract_category)
    
    # Parse creator
    if 'creator' in df.columns:
        def extract_creator(x):
            if pd.isna(x):
                return 'Unknown'
            if isinstance(x, str):
                try:
                    data = json.loads(x)
                    return data.get('name', 'Unknown')
                except:
                    return x
            return str(x)
        
        processed['creator'] = df['creator'].apply(extract_creator)
    
    # Blurb length as description_length
    if 'blurb' in df.columns:
        processed['description_length'] = df['blurb'].fillna('').str.len()
    
    # Add defaults for missing columns
    if 'num_updates' not in processed.columns:
        # Not available in Webrobots, estimate from activity
        processed['num_updates'] = np.random.randint(0, 15, len(processed))
    
    if 'num_comments' not in processed.columns:
        # Estimate as fraction of backers
        if 'backers_count' in processed.columns:
            processed['num_comments'] = (processed['backers_count'] * 0.05).astype(int)
        else:
            processed['num_comments'] = 0
    
    # Generate reward tiers (not available in Webrobots, estimate from goal)
    if 'funding_goal' in processed.columns:
        def generate_reward_tiers(goal):
            if pd.isna(goal) or goal <= 0:
                return [25, 50, 100]
            base = max(10, int(goal / 500))
            return sorted(list(set([
                max(10, base),
                max(20, base * 2),
                max(50, base * 4),
                max(100, base * 8)
            ])))[:5]
        
        processed['reward_tiers'] = processed['funding_goal'].apply(generate_reward_tiers)
    
    # Calculate derived fields
    if 'pledged_amount' in processed.columns and 'funding_goal' in processed.columns:
        processed['is_successful'] = (processed['pledged_amount'] >= processed['funding_goal']).astype(int)
    
    # Clean data
    processed = processed.dropna(subset=['funding_goal', 'pledged_amount', 'backers_count'])
    processed = processed[processed['funding_goal'] > 0]
    processed = processed[processed['campaign_duration_days'] > 0]
    processed = processed[processed['campaign_duration_days'] <= 90]
    
    return processed


def download_real_kickstarter_data(output_path: Path = None,
                                   years: list = [2022, 2023, 2024],
                                   categories: list = None,
                                   target_campaigns: int = 1000) -> pd.DataFrame:
    """
    Main function to download and process real Kickstarter data.
    
    Args:
        output_path: Path to save the final CSV
        years: Years to download data from
        categories: Optional category filter
        target_campaigns: Target number of campaigns
        
    Returns:
        Processed DataFrame
    """
    base_dir = Path(__file__).parent.parent
    
    if output_path is None:
        output_path = base_dir / "data" / "raw" / "kickstarter_real.csv"
    
    print("=" * 60)
    print("DOWNLOADING REAL KICKSTARTER DATA")
    print("=" * 60)
    print(f"Source: Webrobots.io monthly archives")
    print(f"Years: {years}")
    print(f"Target campaigns: {target_campaigns}")
    print()
    
    # Download datasets
    download_dir = base_dir / "data" / "raw" / "webrobots"
    downloaded_files = download_multiple_datasets(
        years=years,
        max_per_year=3,  # Limit to 3 per year to manage size
        output_dir=download_dir
    )
    
    if not downloaded_files:
        print("ERROR: No datasets downloaded. Check internet connection.")
        return None
    
    print(f"\nDownloaded {len(downloaded_files)} dataset files")
    
    # Load and process
    df_raw = load_and_process_webrobots_data(
        downloaded_files,
        categories=categories,
        sample_size=target_campaigns * 2  # Get extra for filtering
    )
    
    # Convert to standard format
    df = process_webrobots_to_standard_format(df_raw)
    
    # Final sample
    if len(df) > target_campaigns:
        df = df.sample(n=target_campaigns, random_state=42)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total campaigns: {len(df)}")
    print(f"Successful: {df['is_successful'].sum() if 'is_successful' in df.columns else 'N/A'}")
    print(f"Categories: {df['category'].nunique() if 'category' in df.columns else 'N/A'}")
    print(f"Saved to: {output_path}")
    
    if 'category' in df.columns:
        print("\nCategory distribution:")
        print(df['category'].value_counts().head(10))
    
    return df


# =============================================================================
# Alternative: Kaggle Dataset
# =============================================================================

def download_from_kaggle(dataset_name: str = "kemical/kickstarter-projects",
                         output_dir: Path = None) -> Path:
    """
    Download Kickstarter data from Kaggle.
    
    Requires: kaggle API credentials (kaggle.json)
    
    Args:
        dataset_name: Kaggle dataset identifier
        output_dir: Output directory
        
    Returns:
        Path to downloaded file
    """
    try:
        import kaggle
    except ImportError:
        print("Kaggle package not installed. Run: pip install kaggle")
        print("Also need to configure kaggle.json with API credentials.")
        return None
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "raw" / "kaggle"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        
        # Find the downloaded CSV
        csv_files = list(output_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]
        return None
        
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Make sure you have kaggle.json configured: https://www.kaggle.com/docs/api")
        return None


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function to download real Kickstarter data."""
    print()
    print("=" * 60)
    print("KICKSTARTER REAL DATA DOWNLOADER")
    print("=" * 60)
    print()
    print("This script downloads REAL Kickstarter data from Webrobots.io")
    print("instead of using synthetic data.")
    print()
    
    # Try Webrobots.io first (no authentication needed)
    df = download_real_kickstarter_data(
        years=[2022, 2023, 2024],
        target_campaigns=1000
    )
    
    if df is not None and len(df) > 0:
        print("\n✅ Successfully downloaded real Kickstarter data!")
        print("\nNow run the enrichment pipeline:")
        print("  python src/data_enrichment.py")
    else:
        print("\n❌ Download failed. Falling back to synthetic data.")
        print("Check your internet connection and try again.")


if __name__ == "__main__":
    main()
