#!/usr/bin/env python
"""
Kaggle Auto-Downloader
======================
Automatically download fresh Kickstarter data from Kaggle using their API.

This is the SOLUTION for automated data updates since direct scraping is blocked.

Setup:
    1. Create Kaggle account at kaggle.com
    2. Go to Account -> Create New API Token
    3. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:/Users/YOU/.kaggle/ (Windows)
    
Usage:
    python src/kaggle_auto_downloader.py
    python src/kaggle_auto_downloader.py --check-new  # Only download if newer

The script will:
    1. Authenticate with Kaggle API
    2. Download the latest Kickstarter dataset
    3. Extract and place in data/raw/kaggle/
    4. Optionally trigger the pipeline
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Known Kickstarter datasets on Kaggle
KAGGLE_DATASETS = [
    "ulrikthygepedersen/kickstarter-projects",  # Best quality, 10.0 usability
    "kemical/kickstarter-projects",              # Most popular, 94K downloads
]


def check_kaggle_installed():
    """Check if kaggle package is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        logger.error("Kaggle package not installed!")
        logger.error("Install with: pip install kaggle")
        return False


def check_kaggle_credentials():
    """Check if Kaggle API credentials exist."""
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get('KAGGLE_CONFIG_DIR', '')) / "kaggle.json" if os.environ.get('KAGGLE_CONFIG_DIR') else None,
    ]
    
    for path in kaggle_json_paths:
        if path and path.exists():
            logger.info(f"Found Kaggle credentials at: {path}")
            return True
    
    # Check environment variables
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        logger.info("Using Kaggle credentials from environment variables")
        return True
    
    logger.error("Kaggle API credentials not found!")
    logger.error("")
    logger.error("To set up Kaggle API:")
    logger.error("  1. Go to https://www.kaggle.com/account")
    logger.error("  2. Click 'Create New API Token'")
    logger.error("  3. Save kaggle.json to:")
    logger.error(f"     Windows: {Path.home() / '.kaggle' / 'kaggle.json'}")
    logger.error(f"     Linux/Mac: ~/.kaggle/kaggle.json")
    logger.error("")
    logger.error("Or set environment variables:")
    logger.error("  KAGGLE_USERNAME=your_username")
    logger.error("  KAGGLE_KEY=your_api_key")
    
    return False


def download_dataset(dataset_name: str, output_dir: Path) -> bool:
    """
    Download a dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'user/dataset-name')
        output_dir: Directory to save the downloaded files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info(f"Output directory: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True
        )
        
        logger.info(f"[OK] Downloaded and extracted: {dataset_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        return False


def get_dataset_info(dataset_name: str) -> dict:
    """Get metadata about a Kaggle dataset."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Parse dataset name
        owner, name = dataset_name.split('/')
        
        # Get dataset details
        datasets = api.dataset_list(search=name)
        
        for ds in datasets:
            if ds.ref == dataset_name:
                return {
                    'name': dataset_name,
                    'title': ds.title,
                    'size': getattr(ds, 'total_bytes', 0),
                    'last_updated': getattr(ds, 'last_updated', None),
                    'download_count': getattr(ds, 'download_count', 0),
                    'usability': getattr(ds, 'usability_rating', 0),
                }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get info for {dataset_name}: {e}")
        return None


def check_for_updates(output_dir: Path) -> bool:
    """Check if there are newer datasets available."""
    last_download_file = output_dir / ".last_download"
    
    if not last_download_file.exists():
        logger.info("No previous download recorded. Will download fresh data.")
        return True
    
    with open(last_download_file, 'r') as f:
        last_download = datetime.fromisoformat(f.read().strip())
    
    logger.info(f"Last download: {last_download}")
    
    # Check each dataset
    for dataset_name in KAGGLE_DATASETS:
        info = get_dataset_info(dataset_name)
        if info and info.get('last_updated'):
            # Parse the date
            try:
                updated = datetime.fromisoformat(str(info['last_updated']).replace('Z', '+00:00'))
                if updated > last_download:
                    logger.info(f"Newer data available: {dataset_name}")
                    logger.info(f"  Dataset updated: {updated}")
                    return True
            except:
                pass
    
    logger.info("No new updates available.")
    return False


def record_download(output_dir: Path):
    """Record the timestamp of the last successful download."""
    last_download_file = output_dir / ".last_download"
    with open(last_download_file, 'w') as f:
        f.write(datetime.now().isoformat())


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download Kickstarter data from Kaggle")
    parser.add_argument('--check-new', action='store_true', 
                        help='Only download if newer data available')
    parser.add_argument('--dataset', type=str, default=KAGGLE_DATASETS[0],
                        help=f'Kaggle dataset to download (default: {KAGGLE_DATASETS[0]})')
    parser.add_argument('--run-pipeline', action='store_true',
                        help='Run the full pipeline after downloading')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KAGGLE AUTO-DOWNLOADER")
    print("=" * 60)
    print()
    
    # Check prerequisites
    if not check_kaggle_installed():
        print("\nInstall kaggle with: pip install kaggle")
        sys.exit(1)
    
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Determine output directory
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "raw" / "kaggle"
    
    # Check for updates if requested
    if args.check_new:
        if not check_for_updates(output_dir):
            print("\n[OK] Data is up to date. No download needed.")
            sys.exit(0)
    
    # Download dataset
    success = download_dataset(args.dataset, output_dir)
    
    if success:
        record_download(output_dir)
        
        # List downloaded files
        print("\nDownloaded files:")
        for f in output_dir.glob("*.csv"):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        print()
        print("=" * 60)
        print("[OK] DOWNLOAD COMPLETE")
        print("=" * 60)
        print()
        print("Next step: Run the pipeline")
        print("  python run_pipeline.py")
        
        # Optionally run pipeline
        if args.run_pipeline:
            print("\nRunning pipeline...")
            import subprocess
            subprocess.run([sys.executable, str(base_dir / "run_pipeline.py")])
    else:
        print()
        print("[FAIL] Download failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
