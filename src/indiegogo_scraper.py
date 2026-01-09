import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import logging
import re

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_driver():
    """Setup Headless Chrome Driver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Mask automation
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scrape_indiegogo(target_count=50):
    """
    Scrapes project data from Indiegogo Trending page.
    """
    url = "https://www.indiegogo.com/explore/all?project_type=campaign&sort=trending"
    driver = setup_driver()
    
    projects = []
    
    try:
        logger.info(f"🚀 Navigating to {url}")
        driver.get(url)
        
        # Wait for cards to load
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".gfu-project-card")))
        
        # Scroll to load more (basic implementation)
        logger.info("Scrolling to load more projects...")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
        card_elements = driver.find_elements(By.CSS_SELECTOR, ".gfu-project-card")
        logger.info(f"Found {len(card_elements)} cards. Extracting data...")
        
        for card in card_elements:
            if len(projects) >= target_count:
                break
                
            try:
                # Title
                title_el = card.find_element(By.CSS_SELECTOR, ".gfu-project-card__title")
                title = title_el.text.strip()
                
                # Category (approximate selector based on analysis)
                try:
                    cat_el = card.find_elements(By.CSS_SELECTOR, "strong._ttu._tc--accent")
                    category = cat_el[0].text.strip() if cat_el else "Unknown"
                except:
                    category = "Unknown"
                    
                # Amount
                try:
                    # Look for the H3 header which usually contains the amount
                    amount_header = card.find_element(By.CSS_SELECTOR, "div.gfu-hd.gfu-hd--h3")
                    amount_text = amount_header.text.strip()
                    # Clean amount: "$1,234 USD" -> 1234.0
                    clean_amount = float(re.sub(r'[^\d.]', '', amount_text.split(' ')[0]))
                except:
                    clean_amount = 0.0
                    
                # Status/Percentage
                # Indiegogo structure is complex here, we'll look for percentage text in the whole card
                try:
                    card_text = card.text
                    match = re.search(r'(\d+)%', card_text)
                    percent_funded = int(match.group(1)) if match else 0
                except:
                    percent_funded = 0
                
                # Indiegogo specific: Goal is not always shown, but we can back-calculate if we have % and amount
                # If 5000 raised is 50%, goal was 10000.
                if percent_funded > 0:
                    goal = int(clean_amount / (percent_funded / 100))
                else:
                    goal = clean_amount # Fallback
                
                project = {
                    'name': title,
                    'category': category,
                    'funding_goal': goal,
                    'pledged_amount': clean_amount,
                    'backers_count': 0, # Difficult to find reliably on summary card
                    'status': 'successful' if percent_funded >= 100 else 'live',
                    'campaign_duration_days': 30, # Default estimate
                    'source': 'indiegogo'
                }
                
                projects.append(project)
                print(f"   Collected: {title[:30]}... (${clean_amount})")
                
            except Exception as e:
                continue
                
    except Exception as e:
        logger.error(f"❌ Scraping Failed: {e}")
    finally:
        driver.quit()
        
    # Save
    if projects:
        df = pd.DataFrame(projects)
        output_path = "data/raw/indiegogo_live.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved {len(df)} Indiegogo projects to {output_path}")
        return df
    else:
        logger.warning("No projects collected.")
        return pd.DataFrame()

if __name__ == "__main__":
    scrape_indiegogo()
