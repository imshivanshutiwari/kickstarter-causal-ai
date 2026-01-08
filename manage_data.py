#!/usr/bin/env python
"""
Kickstarter Data Manager 🚀
=========================
Interactive CLI to manage data updates and model training.
Integrates Kaggle Auto-Downloader and Pipeline Orchestration.
"""

import sys
import time
import subprocess
from pathlib import Path
from src.kaggle_auto_downloader import check_for_updates, download_dataset, KAGGLE_DATASETS, record_download
from src.live_scraper import scrape_live_data

def clear_screen():
    print("\n" * 2)

def run_pipeline():
    print("\n🚀 Starting Pipeline...")
    try:
        subprocess.run([sys.executable, "run_pipeline.py"], check=True)
        print("\n✅ Pipeline Completed Successfully!")
        
        # Auto-launch Dashboard
        print("\n📊 Launching Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
    except subprocess.CalledProcessError:
        print("\n❌ Pipeline Failed!")
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def run_live_scrape():
    print("\n🕸️  Starting Live Scraper (Selenium)...")
    print("This will open a background browser to bypass protections.")
    try:
        csv_path = scrape_live_data(pages=2) # Scrape 2 pages
        if csv_path:
            print("\n✅ Scraped new data!")
            choice = input("Run pipeline with this data? (y/n): ").lower().strip()
            if choice == 'y':
                run_pipeline()
    except Exception as e:
        print(f"\n❌ Scraper Failed: {e}")

def check_and_update():
    print("\n🔍 Checking Kaggle for new data...")
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data" / "raw" / "kaggle"
    
    # 1. Check
    if check_for_updates(output_dir):
        print("\n✨ New data found!")
        choice = input("Do you want to download and retrain? (y/n): ").lower().strip()
        if choice == 'y':
            # 2. Download
            print("\n⬇️ Downloading...")
            if download_dataset(KAGGLE_DATASETS[0], output_dir):
                record_download(output_dir)
                print("✅ Download Complete.")
                
                # 3. Retrain
                run_pipeline()
            else:
                print("❌ Download Failed.")
        else:
            print("Skipping update.")
    else:
        print("\n✅ Your data is up to date (No new version on Kaggle).")
        choice = input("Do you want to force run the pipeline anyway? (y/n): ").lower().strip()
        if choice == 'y':
            run_pipeline()

def force_fresh_start():
    print("\n⚠️  FORCE FRESH START")
    print("This will re-download the dataset even if not new.")
    choice = input("Are you sure? (y/n): ").lower().strip()
    if choice == 'y':
        base_dir = Path(__file__).parent
        output_dir = base_dir / "data" / "raw" / "kaggle"
        
        print("\n⬇️ Downloading fresh data...")
        if download_dataset(KAGGLE_DATASETS[0], output_dir):
            record_download(output_dir)
            run_pipeline()
        else:
            print("❌ Download Failed.")

def main_menu():
    while True:
        clear_screen()
        print("========================================")
        print("   🤖 Kickstarter Data Manager")
        print("========================================")
        print("1. Check for Smart Updates (Recommended)")
        print("   - Checks Kaggle for new data")
        print("   - Using API (Fast & Reliable)")
        print("")
        print("2. Force Fresh Download & Retrain")
        print("   - Re-downloads latest data")
        print("")
        print("3. 🕸️ Live Scraper (Native Technology) [NEW]")
        print("   - Bypasses blocks using Selenium")
        print("   - Scrapes newest campaigns directly")
        print("")
        print("4. Run Pipeline Only")
        print("   - Retrains models on current data")
        print("")
        print("5. Exit")
        print("========================================")
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == '1':
            check_and_update()
            input("\nPress Enter to continue...")
        elif choice == '2':
            force_fresh_start()
            input("\nPress Enter to continue...")
        elif choice == '3':
            run_live_scrape()
            input("\nPress Enter to continue...")
        elif choice == '4':
            run_pipeline()
            input("\nPress Enter to continue...")
        elif choice == '5':
            print("Goodbye! 👋")
            break
        else:
            print("Invalid option.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        # Ensure we can import from src
        sys.path.append(str(Path(__file__).parent))
        main_menu()
    except KeyboardInterrupt:
        print("\nCancelled.")
