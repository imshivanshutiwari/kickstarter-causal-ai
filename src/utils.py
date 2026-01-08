"""
Utility functions for Kickstarter Counterfactual Analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path(data_type: str = "raw") -> Path:
    """Get path to data directory.
    
    Args:
        data_type: Either 'raw' or 'processed'
        
    Returns:
        Path to the data directory
    """
    return get_project_root() / "data" / data_type


def save_dataframe(df: pd.DataFrame, filename: str, data_type: str = "processed") -> str:
    """Save a DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (with or without .csv)
        data_type: Either 'raw' or 'processed'
        
    Returns:
        Path to saved file
    """
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = get_data_path(data_type) / filename
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")
    return str(filepath)


def load_dataframe(filename: str, data_type: str = "raw") -> pd.DataFrame:
    """Load a DataFrame from CSV.
    
    Args:
        filename: Name of the file (with or without .csv)
        data_type: Either 'raw' or 'processed'
        
    Returns:
        Loaded DataFrame
    """
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = get_data_path(data_type) / filename
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df


def is_holiday_proximity(date: datetime, days: int = 7) -> bool:
    """Check if date is within N days of a major US holiday.
    
    Args:
        date: Date to check
        days: Number of days proximity
        
    Returns:
        True if within proximity of a holiday
    """
    # Major US holidays (approximate dates)
    year = date.year
    holidays = [
        datetime(year, 1, 1),   # New Year's Day
        datetime(year, 7, 4),   # Independence Day
        datetime(year, 11, 24), # Thanksgiving (approximate)
        datetime(year, 12, 25), # Christmas
        datetime(year, 2, 14),  # Valentine's Day
        datetime(year, 5, 10),  # Mother's Day (approximate)
        datetime(year, 6, 15),  # Father's Day (approximate)
        datetime(year, 11, 27), # Black Friday (approximate)
    ]
    
    for holiday in holidays:
        if abs((date - holiday).days) <= days:
            return True
    return False


def calculate_funding_ratio(pledged: float, goal: float) -> float:
    """Calculate funding ratio (pledged / goal).
    
    Args:
        pledged: Amount pledged
        goal: Funding goal
        
    Returns:
        Funding ratio
    """
    if goal <= 0:
        return 0.0
    return pledged / goal


def categorize_success(funding_ratio: float) -> str:
    """Categorize campaign success level.
    
    Args:
        funding_ratio: Pledged / Goal ratio
        
    Returns:
        Category label
    """
    if funding_ratio >= 3.0:
        return "viral"
    elif funding_ratio >= 1.5:
        return "strong_success"
    elif funding_ratio >= 1.0:
        return "success"
    elif funding_ratio >= 0.5:
        return "partial"
    else:
        return "failed"


def format_currency(amount: float) -> str:
    """Format amount as USD currency."""
    return f"${amount:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"
