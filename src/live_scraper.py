
import logging
import time
import random
import pandas as pd
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_driver():
    """Setup Chrome Driver with anti-detection options."""
    options = Options()
    options.add_argument("--headless=new")  # Run in background
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scrape_live_data(pages=1):
    """Scrape the newest projects from Kickstarter."""
    driver = None
    data = []
    
    try:
        logger.info("Initializing Browser...")
        driver = setup_driver()
        
        base_url = "https://www.kickstarter.com/discover/advanced?sort=newest&seed=2843981"
        logger.info(f"Navigating to: {base_url}")
        driver.get(base_url)
        
        # Human-like pause
        time.sleep(random.uniform(3, 6))
        
        for page in range(pages):
            logger.info(f"Scraping page {page + 1}...")
            
            # Scroll to trigger lazy loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(2)
            
            # Parse content
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            projects = soup.find_all('div', class_='js-react-proj-card')
            
            if not projects:
                # Fallback implementation for different page structures
                projects = soup.select('div[data-test-id="project-card"]')
            
            logger.info(f"Found {len(projects)} projects on page {page + 1}")
            
            for proj in projects:
                try:
                    # Extract Data safely
                    try:
                        data_json = proj.get('data-project')
                        import json
                        if data_json:
                            p = json.loads(data_json)
                            name = p.get('name')
                            blurb = p.get('blurb')
                            goal = p.get('goal')
                            pledged = p.get('pledged')
                            state = p.get('state')
                            category = p.get('category', {}).get('name')
                            country = p.get('country')
                            
                            data.append({
                                'name': name,
                                'desc': blurb,  # Important for our NLP
                                'goal': goal,
                                'pledged': pledged,
                                'state': state,
                                'category': category,
                                'country': country,
                                'source': 'live_scrape'
                            })
                            continue
                    except:
                        pass
                        
                    # Backup parsing if JSON fails
                    title_elem = proj.find('h3', class_='type-13') or proj.select_one('[data-test-id="project-link"]')
                    desc_elem = proj.find('p', class_='type-12') or proj.select_one('.project-description')
                    
                    if title_elem:
                        data.append({
                            'name': title_elem.get_text(strip=True),
                            'desc': desc_elem.get_text(strip=True) if desc_elem else "",
                            'goal': 0, # Placeholder
                            'pledged': 0,
                            'state': 'live',
                            'category': 'Unknown',
                            'country': 'US',
                            'source': 'live_scrape_fallback'
                        })
                        
                except Exception as e:
                    logger.warning(f"Error parsing project: {e}")
            
            # Next page logic could go here (click 'Load more')
            
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
    finally:
        if driver:
            driver.quit()
            
    # Save Data
    if data:
        df = pd.DataFrame(data)
        output_dir = Path(__file__).parent.parent / "data" / "raw" / "scraped"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"live_kickstarter_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"✅ Successfully scraped {len(df)} campaigns to {filename}")
        return str(filename)
    else:
        logger.warning("❌ No data scraped.")
        return None

if __name__ == "__main__":
    scrape_live_data()
