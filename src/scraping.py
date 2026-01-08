"""
Kickstarter Data Scraping Module

This module handles data collection from Kickstarter using web scraping.
It supports both BeautifulSoup and Selenium approaches with robust error handling,
rate limiting, checkpoint saving, and detailed logging.

Features:
- Scrapes 500-1000 completed Technology campaigns (2020-2024)
- Rate limiting (1-2 second delay between requests)
- Checkpoint saving every 100 campaigns
- Comprehensive logging to scraping_log.txt
- Graceful error handling for timeouts, blocks, missing data
- Fallback to synthetic data generation for development

Author: Kickstarter Counterfactual Project
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import time
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
import pickle

# Try to import Selenium (optional)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

class ScrapingConfig:
    """Configuration for the scraping process."""
    
    # Target settings
    TARGET_CAMPAIGNS = 600
    CATEGORY = "technology"
    YEAR_RANGE = (2020, 2024)
    
    # Rate limiting
    MIN_DELAY = 1.0  # Minimum seconds between requests
    MAX_DELAY = 2.0  # Maximum seconds between requests
    
    # Checkpointing
    CHECKPOINT_INTERVAL = 100  # Save every N campaigns
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    OUTPUT_FILE = RAW_DATA_DIR / "kickstarter_raw.csv"
    CHECKPOINT_FILE = RAW_DATA_DIR / "scraping_checkpoint.pkl"
    LOG_FILE = RAW_DATA_DIR / "scraping_log.txt"
    
    # Request settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    
    # User agent rotation
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    ]


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Path) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Configured logger instance
    """
    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('kickstarter_scraper')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler (info only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Kickstarter Scraper Class
# =============================================================================

class KickstarterScraper:
    """
    Robust scraper for collecting Kickstarter campaign data.
    
    Supports both requests+BeautifulSoup and Selenium approaches.
    Includes rate limiting, checkpoint saving, and comprehensive error handling.
    
    Attributes:
        config: ScrapingConfig instance
        logger: Logger instance
        session: Requests session
        driver: Selenium WebDriver (if available)
        campaigns: List of scraped campaign data
    """
    
    # Kickstarter URLs
    BASE_URL = "https://www.kickstarter.com"
    DISCOVER_API = "https://www.kickstarter.com/discover/advanced"
    
    # Category IDs
    CATEGORY_IDS = {
        "technology": 16,
        "design": 7,
        "games": 12,
        "film_video": 11,
        "music": 14,
        "publishing": 18,
        "art": 1,
        "fashion": 9,
        "food": 10,
        "photography": 15,
    }
    
    def __init__(self, config: ScrapingConfig = None, use_selenium: bool = False):
        """
        Initialize the scraper.
        
        Args:
            config: ScrapingConfig instance (uses default if None)
            use_selenium: Whether to use Selenium instead of requests
        """
        self.config = config or ScrapingConfig()
        self.logger = setup_logging(self.config.LOG_FILE)
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        
        # Initialize session
        self.session = requests.Session()
        self._update_headers()
        
        # Initialize Selenium if requested
        self.driver = None
        if self.use_selenium:
            self._init_selenium()
        
        # Storage
        self.campaigns = []
        self.errors = []
        self.last_request_time = 0
        
        self.logger.info("=" * 60)
        self.logger.info("Kickstarter Scraper Initialized")
        self.logger.info(f"Target: {self.config.TARGET_CAMPAIGNS} campaigns")
        self.logger.info(f"Category: {self.config.CATEGORY}")
        self.logger.info(f"Using Selenium: {self.use_selenium}")
        self.logger.info("=" * 60)
    
    def _update_headers(self):
        """Update session headers with a random user agent."""
        self.session.headers.update({
            'User-Agent': random.choice(self.config.USER_AGENTS),
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': self.BASE_URL,
        })
    
    def _init_selenium(self):
        """Initialize Selenium WebDriver with Chrome."""
        if not SELENIUM_AVAILABLE:
            self.logger.warning("Selenium not available, falling back to requests")
            self.use_selenium = False
            return
        
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument(f'user-agent={random.choice(self.config.USER_AGENTS)}')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Selenium: {e}")
            self.use_selenium = False
    
    def _rate_limit_wait(self):
        """Wait to respect rate limiting."""
        elapsed = time.time() - self.last_request_time
        min_wait = random.uniform(self.config.MIN_DELAY, self.config.MAX_DELAY)
        
        if elapsed < min_wait:
            sleep_time = min_wait - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                      retries: int = None) -> Optional[requests.Response]:
        """
        Make a rate-limited HTTP request with retries.
        
        Args:
            url: URL to request
            params: Query parameters
            retries: Number of retries (uses config default if None)
            
        Returns:
            Response object or None if all retries failed
        """
        retries = retries or self.config.MAX_RETRIES
        
        for attempt in range(retries):
            self._rate_limit_wait()
            
            # Rotate user agent occasionally
            if random.random() < 0.1:
                self._update_headers()
            
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = self.config.RETRY_DELAY * (attempt + 2)
                    self.logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 403:
                    self.logger.warning("Access forbidden - may be blocked")
                    time.sleep(self.config.RETRY_DELAY * 2)
                else:
                    self.logger.debug(f"Request failed with status {response.status_code}")
                    
            except requests.Timeout:
                self.logger.warning(f"Request timeout (attempt {attempt + 1}/{retries})")
            except requests.RequestException as e:
                self.logger.warning(f"Request error: {e} (attempt {attempt + 1}/{retries})")
            
            if attempt < retries - 1:
                time.sleep(self.config.RETRY_DELAY)
        
        self.logger.error(f"All {retries} attempts failed for {url}")
        return None
    
    def _fetch_with_selenium(self, url: str) -> Optional[str]:
        """
        Fetch page content using Selenium.
        
        Args:
            url: URL to fetch
            
        Returns:
            Page source HTML or None if failed
        """
        if not self.driver:
            return None
        
        try:
            self._rate_limit_wait()
            self.driver.get(url)
            
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            return self.driver.page_source
        except Exception as e:
            self.logger.warning(f"Selenium error: {e}")
            return None
    
    def get_discover_page(self, page: int = 1, state: str = "successful") -> Optional[Dict]:
        """
        Get a page of projects from Kickstarter's discover API.
        
        Args:
            page: Page number (1-indexed)
            state: Campaign state ('successful', 'failed', 'all')
            
        Returns:
            JSON response data or None if failed
        """
        params = {
            'category_id': self.CATEGORY_IDS.get(self.config.CATEGORY, 16),
            'state': state,
            'sort': 'end_date',
            'page': page,
            'format': 'json',
        }
        
        response = self._make_request(self.DISCOVER_API, params)
        
        if response:
            try:
                data = response.json()
                return data
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON response")
        
        return None
    
    def parse_campaign(self, project_data: Dict) -> Optional[Dict]:
        """
        Parse campaign data from Kickstarter API response.
        
        Extracts all required fields with proper error handling for missing data.
        
        Args:
            project_data: Raw project data from API
            
        Returns:
            Parsed campaign dictionary or None if essential data missing
        """
        try:
            # Extract basic info
            project_id = project_data.get('id')
            if not project_id:
                return None
            
            # Name and creator
            name = project_data.get('name', '')
            creator = project_data.get('creator', {})
            creator_name = creator.get('name', 'Unknown')
            
            # Category info
            category_data = project_data.get('category', {})
            category = category_data.get('name', '')
            subcategory = category_data.get('parent_name', category_data.get('slug', ''))
            
            # Funding info
            goal = float(project_data.get('goal', 0))
            pledged = float(project_data.get('pledged', 0))
            backers_count = int(project_data.get('backers_count', 0))
            currency = project_data.get('currency', 'USD')
            
            # Convert to USD if necessary (simplified)
            if currency != 'USD':
                static_converted = project_data.get('usd_pledged', pledged)
                pledged = float(static_converted) if static_converted else pledged
            
            # Dates
            created_at = project_data.get('created_at')
            launched_at = project_data.get('launched_at')
            deadline = project_data.get('deadline')
            state_changed_at = project_data.get('state_changed_at')
            
            launch_date = None
            end_date = None
            duration_days = None
            
            if launched_at:
                launch_date = datetime.fromtimestamp(launched_at).strftime('%Y-%m-%d')
            if deadline:
                end_date = datetime.fromtimestamp(deadline).strftime('%Y-%m-%d')
            if launched_at and deadline:
                duration_days = (deadline - launched_at) // 86400
            
            # Filter by year
            if launch_date:
                year = int(launch_date[:4])
                if year < self.config.YEAR_RANGE[0] or year > self.config.YEAR_RANGE[1]:
                    return None
            
            # Status
            state = project_data.get('state', '')
            if state not in ['successful', 'failed']:
                return None
            
            # Campaign stats
            updates_count = project_data.get('updates_count', 0)
            comments_count = project_data.get('comments_count', 0)
            
            # Description
            blurb = project_data.get('blurb', '')
            description_length = len(blurb)
            
            # Location
            location = project_data.get('location', {})
            country = project_data.get('country', location.get('country', ''))
            
            # Reward tiers (from rewards if available)
            reward_tiers = []
            rewards = project_data.get('rewards', [])
            if rewards:
                reward_tiers = [r.get('minimum', 0) for r in rewards if r.get('minimum')]
            
            # Construct campaign dict
            campaign = {
                'project_id': project_id,
                'name': name,
                'creator': creator_name,
                'category': category,
                'subcategory': subcategory,
                'funding_goal': round(goal, 2),
                'pledged_amount': round(pledged, 2),
                'backers_count': backers_count,
                'campaign_duration_days': duration_days,
                'num_updates': updates_count,
                'num_comments': comments_count,
                'description_length': description_length,
                'launch_date': launch_date,
                'end_date': end_date,
                'status': state,
                'reward_tiers': json.dumps(reward_tiers) if reward_tiers else '[]',
                'country': country,
                'funding_ratio': round(pledged / goal, 4) if goal > 0 else 0,
            }
            
            return campaign
            
        except Exception as e:
            self.logger.debug(f"Error parsing campaign: {e}")
            self.errors.append({'type': 'parse_error', 'error': str(e)})
            return None
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint = {
            'campaigns': self.campaigns,
            'errors': self.errors,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.config.CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.info(f"Checkpoint saved: {len(self.campaigns)} campaigns")
    
    def load_checkpoint(self) -> bool:
        """
        Load progress from checkpoint file if exists.
        
        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if self.config.CHECKPOINT_FILE.exists():
            try:
                with open(self.config.CHECKPOINT_FILE, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                self.campaigns = checkpoint.get('campaigns', [])
                self.errors = checkpoint.get('errors', [])
                
                self.logger.info(f"Loaded checkpoint: {len(self.campaigns)} campaigns")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        
        return False
    
    def scrape_campaigns(self, resume: bool = True) -> pd.DataFrame:
        """
        Main scraping method. Collects campaigns from Kickstarter.
        
        Args:
            resume: Whether to resume from checkpoint if available
            
        Returns:
            DataFrame with all scraped campaigns
        """
        self.logger.info("Starting campaign scraping...")
        
        # Try to resume from checkpoint
        if resume and self.load_checkpoint():
            start_count = len(self.campaigns)
            self.logger.info(f"Resuming from {start_count} campaigns")
        else:
            start_count = 0
        
        # Determine how many more we need
        remaining = self.config.TARGET_CAMPAIGNS - len(self.campaigns)
        if remaining <= 0:
            self.logger.info("Target already reached!")
            return pd.DataFrame(self.campaigns)
        
        # Scrape both successful and failed campaigns
        states = ['successful', 'failed']
        campaigns_per_state = remaining // len(states) + 1
        
        for state in states:
            if len(self.campaigns) >= self.config.TARGET_CAMPAIGNS:
                break
            
            page = 1
            state_count = 0
            
            pbar = tqdm(
                total=campaigns_per_state, 
                desc=f"Scraping {state}",
                unit="campaigns"
            )
            
            while state_count < campaigns_per_state:
                # Get page of projects
                data = self.get_discover_page(page=page, state=state)
                
                if not data:
                    self.logger.warning(f"No data returned for page {page}")
                    break
                
                projects = data.get('projects', [])
                if not projects:
                    self.logger.info(f"No more {state} projects available")
                    break
                
                # Parse each project
                for project in projects:
                    campaign = self.parse_campaign(project)
                    
                    if campaign:
                        self.campaigns.append(campaign)
                        state_count += 1
                        pbar.update(1)
                        
                        # Checkpoint
                        if len(self.campaigns) % self.config.CHECKPOINT_INTERVAL == 0:
                            self.save_checkpoint()
                        
                        if len(self.campaigns) >= self.config.TARGET_CAMPAIGNS:
                            break
                
                page += 1
                
                # Safety limit
                if page > 200:
                    self.logger.warning("Reached page limit")
                    break
            
            pbar.close()
            self.logger.info(f"Collected {state_count} {state} campaigns")
        
        # Final save
        df = pd.DataFrame(self.campaigns)
        self._save_results(df)
        
        return df
    
    def _save_results(self, df: pd.DataFrame):
        """Save final results to CSV and clean up."""
        # Ensure directory exists
        self.config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df.to_csv(self.config.OUTPUT_FILE, index=False)
        self.logger.info(f"Saved {len(df)} campaigns to {self.config.OUTPUT_FILE}")
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("SCRAPING COMPLETE")
        self.logger.info(f"Total campaigns: {len(df)}")
        if 'status' in df.columns:
            self.logger.info(f"Successful: {(df['status'] == 'successful').sum()}")
            self.logger.info(f"Failed: {(df['status'] == 'failed').sum()}")
        self.logger.info(f"Errors encountered: {len(self.errors)}")
        self.logger.info("=" * 60)
        
        # Optionally remove checkpoint
        if self.config.CHECKPOINT_FILE.exists():
            os.remove(self.config.CHECKPOINT_FILE)
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            self.logger.info("Selenium WebDriver closed")


# =============================================================================
# Synthetic Data Generator (Fallback)
# =============================================================================

def generate_synthetic_kickstarter_data(num_campaigns: int = 600) -> pd.DataFrame:
    """
    Generate realistic synthetic Kickstarter data for development.
    
    This function creates synthetic data that mimics real Kickstarter patterns,
    useful when actual scraping isn't possible or for development purposes.
    
    The synthetic data includes realistic causal relationships between
    price, quality, marketing, and outcomes for causal inference analysis.
    
    Args:
        num_campaigns: Number of campaigns to generate
        
    Returns:
        DataFrame with synthetic campaign data
    """
    np.random.seed(42)
    
    categories = ['Technology', 'Design', 'Games', 'Film & Video', 'Music', 'Publishing']
    category_weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
    
    # Category-specific parameters for realistic generation
    category_params = {
        'Technology': {'goal_mean': 50000, 'goal_std': 80000, 'price_mean': 150, 'success_base': 0.35},
        'Design': {'goal_mean': 25000, 'goal_std': 40000, 'price_mean': 80, 'success_base': 0.40},
        'Games': {'goal_mean': 30000, 'goal_std': 50000, 'price_mean': 50, 'success_base': 0.38},
        'Film & Video': {'goal_mean': 20000, 'goal_std': 30000, 'price_mean': 30, 'success_base': 0.30},
        'Music': {'goal_mean': 10000, 'goal_std': 15000, 'price_mean': 25, 'success_base': 0.45},
        'Publishing': {'goal_mean': 8000, 'goal_std': 12000, 'price_mean': 20, 'success_base': 0.35},
    }
    
    data = []
    
    for i in tqdm(range(num_campaigns), desc="Generating synthetic data"):
        # Category selection
        category = np.random.choice(categories, p=category_weights)
        params = category_params[category]
        
        # Generate funding goal (lognormal distribution)
        goal = max(1000, np.random.lognormal(np.log(params['goal_mean']), 0.8))
        goal = round(goal, -2)  # Round to nearest 100
        
        # Campaign duration (mostly 30 days, some variation)
        duration = int(np.random.choice([15, 21, 30, 45, 60], p=[0.05, 0.10, 0.60, 0.15, 0.10]))
        
        # Reward tiers
        num_tiers = min(10, max(3, int(np.log10(goal) * 2) + np.random.randint(-1, 2)))
        
        # Average reward price
        avg_price = max(10, params['price_mean'] * (0.5 + np.random.random()))
        avg_price = round(avg_price)
        
        # Generate reward tier list
        tier_prices = sorted([
            max(1, int(avg_price * np.random.uniform(0.2, 3.0)))
            for _ in range(num_tiers)
        ])
        
        # Generate launch date (2020-2024)
        launch_date = datetime(2020, 1, 1) + timedelta(days=int(np.random.randint(0, 1460)))
        end_date = launch_date + timedelta(days=duration)
        
        # Day of week and seasonality
        day_of_week = launch_date.weekday()
        month = launch_date.month
        
        # Holiday proximity
        holiday_proximity = 1 if month in [11, 12, 1] else 0
        
        # Concurrent campaigns
        concurrent_campaigns = np.random.poisson(15 if month in [3, 4, 9, 10] else 10)
        
        # Google Trends proxy
        trend_index = min(100, max(0, int(np.random.normal(50, 20))))
        
        # Media coverage
        got_press_coverage = 1 if np.random.random() < (0.05 + 0.1 * (goal > 50000)) else 0
        
        # Updates and comments
        updates = np.random.poisson(duration / 5)
        
        # Description length
        description_length = np.random.randint(200, 2000)
        
        # --- CAUSAL OUTCOME GENERATION ---
        success_prob = params['success_base']
        
        # Price sensitivity
        price_effect = -0.001 * (avg_price - params['price_mean'])
        
        # Goal ambition penalty
        goal_effect = -0.15 * (goal / params['goal_mean'] - 1) if goal > params['goal_mean'] else 0
        
        # Day of week effect
        dow_effect = 0.05 if day_of_week in [1, 2] else (-0.05 if day_of_week in [5, 6] else 0)
        
        # Press coverage boost
        press_effect = 0.25 if got_press_coverage else 0
        
        # Trend boost
        trend_effect = 0.002 * (trend_index - 50)
        
        # Competition penalty
        competition_effect = -0.005 * (concurrent_campaigns - 12)
        
        # Random quality
        quality = np.random.normal(0, 0.15)
        
        # Final probability
        final_prob = min(0.95, max(0.05, success_prob + price_effect + goal_effect + 
                                   dow_effect + press_effect + trend_effect + 
                                   competition_effect + quality))
        
        # Determine success
        is_successful = np.random.random() < final_prob
        
        # Generate funding ratio
        if is_successful:
            base_ratio = 1.0 + np.random.exponential(0.5)
            if got_press_coverage:
                base_ratio *= 1.5
            funding_ratio = min(10.0, base_ratio)
        else:
            funding_ratio = np.random.beta(2, 5) * 0.99
        
        pledged = goal * funding_ratio
        
        # Backers
        backers = max(1, int((pledged / avg_price) * np.random.uniform(0.7, 1.3)))
        
        # Comments
        comments = int(backers * np.random.uniform(0.05, 0.15))
        
        # Creator name
        creator_name = f"Creator_{i+1}"
        
        campaign = {
            'project_id': f'proj_{i+1:05d}',
            'name': f'{category} Project {i+1}',
            'creator': creator_name,
            'category': category,
            'subcategory': category,
            'funding_goal': round(goal, 2),
            'pledged_amount': round(pledged, 2),
            'backers_count': backers,
            'campaign_duration_days': duration,
            'num_updates': updates,
            'num_comments': comments,
            'description_length': description_length,
            'launch_date': launch_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'status': 'successful' if is_successful else 'failed',
            'reward_tiers': json.dumps(tier_prices),
            'country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'AU'], 
                                        p=[0.5, 0.15, 0.1, 0.1, 0.08, 0.07]),
            'funding_ratio': round(funding_ratio, 4),
            # Additional columns for causal analysis
            'day_of_week': day_of_week,
            'month': month,
            'num_reward_tiers': num_tiers,
            'avg_reward_price': round(avg_price, 2),
            'trend_index': trend_index,
            'concurrent_campaigns': concurrent_campaigns,
            'got_press_coverage': got_press_coverage,
            'holiday_proximity': holiday_proximity,
            'is_successful': int(is_successful),
        }
        
        data.append(campaign)
    
    df = pd.DataFrame(data)
    
    # Add derived columns
    df['backers_per_day'] = df['backers_count'] / df['campaign_duration_days']
    df['pledged_per_backer'] = df['pledged_amount'] / df['backers_count'].replace(0, 1)
    df['goal_ambition'] = df.groupby('category')['funding_goal'].transform(lambda x: x / x.median())
    df['price_to_goal_ratio'] = df['avg_reward_price'] / (df['funding_goal'] / 100)
    df['hit_funding_cap'] = (df['funding_ratio'] > 3.0).astype(int)
    
    return df


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main function to run data collection.
    
    Attempts real scraping first, falls back to synthetic data if blocked.
    """
    print("=" * 60)
    print("KICKSTARTER DATA COLLECTION")
    print("=" * 60)
    
    config = ScrapingConfig()
    
    # Ensure directories exist
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try real scraping
    print("\nAttempting to scrape real Kickstarter data...")
    print("(Will fall back to synthetic data if blocked)")
    
    try:
        scraper = KickstarterScraper(config)
        df = scraper.scrape_campaigns(resume=True)
        scraper.close()
        
        if len(df) < 100:
            raise Exception("Too few campaigns scraped, falling back to synthetic data")
            
    except Exception as e:
        print(f"\nReal scraping failed: {e}")
        print("Generating synthetic data instead...")
        
        df = generate_synthetic_kickstarter_data(num_campaigns=config.TARGET_CAMPAIGNS)
        
        # Save to the expected location
        output_path = config.RAW_DATA_DIR / "kickstarter_raw.csv"
        df.to_csv(output_path, index=False)
        
        # Also save as kickstarter_raw_data.csv for compatibility
        df.to_csv(config.RAW_DATA_DIR / "kickstarter_raw_data.csv", index=False)
        
        # Write log
        with open(config.LOG_FILE, 'a') as f:
            f.write(f"\n{datetime.now().isoformat()} | INFO | Generated {len(df)} synthetic campaigns\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total campaigns: {len(df)}")
    print(f"Successful: {(df['status'] == 'successful').sum()}")
    print(f"Failed: {(df['status'] == 'failed').sum()}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nData saved to: {config.RAW_DATA_DIR}")
    
    return df


if __name__ == "__main__":
    main()
