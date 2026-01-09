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
import webbrowser
from pathlib import Path
from colorama import Fore, Style, init
from src.kaggle_auto_downloader import check_for_updates, download_dataset, KAGGLE_DATASETS, record_download
from src.live_scraper import scrape_live_data
from src.indiegogo_scraper import scrape_indiegogo

# Initialize Colorama
init()

def clear_screen():
    print("\n" * 2)

def run_pipeline():
    print("\n🚀 Starting Pipeline...")
    try:
        # Run training pipeline (blocking is fine here as it finishes quickly)
        subprocess.run([sys.executable, "run_pipeline.py"], check=True)
        print("\n✅ Pipeline Completed Successfully!")
        
        # Auto-launch Dashboard (NON-BLOCKING / HANDLED)
        print("\n📊 Launching Dashboard...")
        print("Press Ctrl+C to stop the dashboard and return to menu.")
        
        # Use Popen instead of run so we can control it
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.address", "localhost", "--server.port", "8501"]
        process = subprocess.Popen(cmd)
        
        # Give server a moment to start, then open browser
        print("⏳ Waiting for server to initialize...")
        time.sleep(5) 
        
        if process.poll() is None:
            print("🌐 Opening http://localhost:8501 in your browser...")
            webbrowser.open("http://localhost:8501")
        else:
            print("❌ Streamlit crashed early. Check output above for errors.")
        
        # Wait for user to stop it
        process.wait()
        
    except subprocess.CalledProcessError:
        print("\n❌ Pipeline Failed!")
    except KeyboardInterrupt:
        print("\n🛑 Stopping Dashboard...")
        if 'process' in locals():
            process.terminate()
            process.wait() # Ensure it dies
        time.sleep(1) # Give it a moment
    except Exception as e:
        print(f"\n❌ Error: {e}")

def run_live_scrape():
    # ... (omitted for brevity, just keeping placeholder if needed, but the original code had this logic inline or separate)
    # The previous edit deleted this too. I need to restore the logic used in main_menu
    pass

def main_menu():
    while True:
        clear_screen()
        print(f"{Fore.BLUE}========================================")
        print(f"   🤖 Kickstarter Data Manager")
        print(f"========================================{Style.RESET_ALL}")
        print(f"{Fore.GREEN}1. Smart Update (Kaggle){Style.RESET_ALL} - Check for new Kaggle data & retrain.")
        print(f"{Fore.YELLOW}2. Force Full Update (Kaggle){Style.RESET_ALL} - Re-download Kaggle data & retrain.")
        print(f"{Fore.MAGENTA}3. Scrape Kickstarter Live (NEW!){Style.RESET_ALL} - Get trending campaigns from Kickstarter.")
        print(f"{Fore.CYAN}4. Scrape Indiegogo (NEW!){Style.RESET_ALL} - Get trending campaigns from Indiegogo.")
        print(f"{Fore.WHITE}5. Run Pipeline (Train Models){Style.RESET_ALL} - Process local data & launch dashboard.")
        print(f"{Fore.RED}6. Exit{Style.RESET_ALL}")
        print(f"{Fore.BLUE}========================================{Style.RESET_ALL}")

        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print(f"\n{Fore.YELLOW}Checking for new Kaggle data...{Style.RESET_ALL}")
            check_and_update()
            input("\nPress Enter to continue...")
        elif choice == '2':
            print(f"\n{Fore.YELLOW}Forcing full update...{Style.RESET_ALL}")
            force_fresh_start() # Renamed from force_update to match existing function
            input("\nPress Enter to continue...")
        elif choice == '3':
            print(f"\n{Fore.YELLOW}Launching Live Kickstarter Scraper...{Style.RESET_ALL}")
            from src.live_scraper import scrape_kickstarter_live
            df = scrape_kickstarter_live()
            if not df.empty:
                print(f"{Fore.GREEN}Scraping complete! Processing data...{Style.RESET_ALL}")
                run_pipeline()
            input("\nPress Enter to continue...")
        elif choice == '4':
            print(f"\n{Fore.YELLOW}Launching Live Indiegogo Scraper...{Style.RESET_ALL}")
            from src.indiegogo_scraper import scrape_indiegogo
            df = scrape_indiegogo()
            if not df.empty:
                print(f"{Fore.GREEN}Indiegogo scrap complete! (Note: Saved to data/raw/indiegogo_live.csv){Style.RESET_ALL}")
                print(f"{Fore.CYAN}To use this in the main model, we need to harmonize schemas in 'process_kaggle_data.py'.{Style.RESET_ALL}")
            input("\nPress Enter to continue...")
        elif choice == '5':
            run_pipeline()
            input("\nPress Enter to continue...")
        elif choice == '6':
            print("Exiting...")
            sys.exit(0)
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
