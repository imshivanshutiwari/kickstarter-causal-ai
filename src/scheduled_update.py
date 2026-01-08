#!/usr/bin/env python
"""
Scheduled Data Update Script
============================
Can be run manually or via cron/Task Scheduler to check for new data.

Windows Task Scheduler:
    1. Open Task Scheduler
    2. Create Basic Task
    3. Set trigger (e.g., weekly)
    4. Action: Start program
    5. Program: python
    6. Arguments: "C:\path\to\src\scheduled_update.py"

Linux Cron:
    # Run every Sunday at midnight
    0 0 * * 0 python /path/to/src/scheduled_update.py
"""

import os
import sys
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduled_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Email configuration (optional - set environment variables)
EMAIL_FROM = os.environ.get('EMAIL_FROM', '')
EMAIL_TO = os.environ.get('EMAIL_TO', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))


def send_notification(subject: str, message: str):
    """Send email notification."""
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD]):
        logger.info("Email not configured. Skipping notification.")
        return
    
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


def run_update():
    """Run the scheduled update."""
    base_dir = Path(__file__).parent.parent
    
    logger.info("=" * 60)
    logger.info("SCHEDULED UPDATE STARTED")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)
    
    try:
        # Try to download new data
        result = subprocess.run(
            [sys.executable, str(base_dir / 'src' / 'kaggle_auto_downloader.py'), '--check-new'],
            capture_output=True,
            text=True,
            cwd=str(base_dir)
        )
        
        if "No new updates" in result.stdout:
            logger.info("No new data available")
            return
        
        if result.returncode == 0:
            logger.info("New data downloaded! Running pipeline...")
            
            # Run pipeline
            pipeline_result = subprocess.run(
                [sys.executable, str(base_dir / 'run_pipeline.py')],
                capture_output=True,
                text=True,
                cwd=str(base_dir)
            )
            
            if pipeline_result.returncode == 0:
                send_notification(
                    "Kickstarter Data Updated",
                    f"New Kickstarter data has been downloaded and processed.\n\n"
                    f"Time: {datetime.now()}\n"
                    f"Status: Success"
                )
                logger.info("[OK] Update complete")
            else:
                send_notification(
                    "Kickstarter Pipeline Failed",
                    f"Pipeline failed after data download.\n\n"
                    f"Error: {pipeline_result.stderr}"
                )
                logger.error("[FAIL] Pipeline failed")
        else:
            logger.info("No update needed or download failed")
            
    except Exception as e:
        logger.error(f"Update failed: {e}")
        send_notification(
            "Kickstarter Update Failed",
            f"Scheduled update failed.\n\nError: {e}"
        )


if __name__ == "__main__":
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    run_update()
