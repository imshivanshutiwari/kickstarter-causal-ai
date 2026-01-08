#!/usr/bin/env python
"""
FastAPI Prediction Endpoint
===========================
REST API for Kickstarter counterfactual predictions.

Usage:
    uvicorn src.api:app --reload --port 8000
    
    Or:
    python src/api.py

Endpoints:
    GET  /                   - API info
    GET  /health             - Health check
    POST /predict            - Get prediction for a campaign
    POST /counterfactual     - Get counterfactual analysis
    GET  /categories         - List available categories
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Kickstarter Counterfactual API",
    description="REST API for causal prediction of Kickstarter campaign outcomes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = None
data = None


# =============================================================================
# Pydantic Models
# =============================================================================

class CampaignInput(BaseModel):
    """Input parameters for a campaign prediction."""
    category: str = Field(default="Technology", description="Campaign category")
    funding_goal: float = Field(default=50000, ge=100, le=10000000, description="Funding goal in USD")
    campaign_duration: int = Field(default=30, ge=1, le=90, description="Duration in days")
    avg_reward_price: float = Field(default=50, ge=1, le=10000, description="Average reward price")
    num_reward_tiers: int = Field(default=5, ge=1, le=20, description="Number of reward tiers")
    trend_index: float = Field(default=50, ge=0, le=100, description="Market trend index")


class PredictionOutput(BaseModel):
    """Prediction output."""
    expected_funding: float
    funding_ratio: float
    success_probability: float
    confidence_low: float
    confidence_high: float
    recommendation: str


class CounterfactualInput(BaseModel):
    """Input for counterfactual analysis."""
    base_campaign: CampaignInput
    price_range: Optional[List[float]] = None  # If None, uses default range


class CounterfactualPoint(BaseModel):
    """Single point in counterfactual curve."""
    price: float
    predicted_funding_ratio: float


class CounterfactualOutput(BaseModel):
    """Counterfactual analysis output."""
    demand_curve: List[CounterfactualPoint]
    optimal_price: float
    current_price: float
    price_effect: str


# =============================================================================
# Helper Functions
# =============================================================================

def load_models():
    """Load trained models and data."""
    global models, data
    
    if models is not None:
        return
    
    base_dir = Path(__file__).parent.parent
    models_path = base_dir / "src" / "models" / "causal_models.pkl"
    data_path = base_dir / "data" / "processed" / "kickstarter_causal_features.csv"
    
    try:
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        logger.info("Loaded models successfully")
    except FileNotFoundError:
        logger.error(f"Models not found at {models_path}")
        models = {}
    
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} campaigns")
    except FileNotFoundError:
        logger.warning(f"Data not found at {data_path}")
        data = pd.DataFrame()


def get_prediction(campaign: CampaignInput) -> dict:
    """Generate prediction for a campaign."""
    load_models()
    
    if not models or 'ols' not in models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Prepare features
    feature_cols = models.get('feature_cols', ['avg_reward_price', 'goal_ambition', 
                                                'campaign_duration_days', 'trend_index',
                                                'concurrent_campaigns'])
    
    # Calculate derived features
    median_goal = data['funding_goal'].median() if len(data) > 0 else 50000
    goal_ambition = campaign.funding_goal / median_goal
    
    features = {
        'avg_reward_price': campaign.avg_reward_price,
        'goal_ambition': goal_ambition,
        'campaign_duration_days': campaign.campaign_duration,
        'trend_index': campaign.trend_index,
        'concurrent_campaigns': 0.5  # Default
    }
    
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    
    # Scale features
    scaler = models.get('scaler')
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Get prediction
    ols = models.get('ols')
    funding_ratio = float(ols.predict(X_scaled)[0])
    
    # Get quantile predictions for confidence interval
    quantile_models = models.get('quantile_models', {})
    if quantile_models:
        q10 = float(quantile_models.get(0.1).predict(X)[0]) if 0.1 in quantile_models else funding_ratio * 0.5
        q90 = float(quantile_models.get(0.9).predict(X)[0]) if 0.9 in quantile_models else funding_ratio * 1.5
    else:
        q10 = funding_ratio * 0.5
        q90 = funding_ratio * 1.5
    
    expected_funding = campaign.funding_goal * funding_ratio
    success_prob = min(1.0, max(0.0, funding_ratio))
    
    # Generate recommendation
    if funding_ratio >= 1.5:
        recommendation = "Strong Success - Consider higher goal"
    elif funding_ratio >= 1.0:
        recommendation = "Likely Success - On track"
    elif funding_ratio >= 0.7:
        recommendation = "At Risk - Consider adjustments"
    else:
        recommendation = "Needs Work - Review pricing and goal"
    
    return {
        'expected_funding': round(expected_funding, 2),
        'funding_ratio': round(funding_ratio, 3),
        'success_probability': round(success_prob, 3),
        'confidence_low': round(campaign.funding_goal * q10, 2),
        'confidence_high': round(campaign.funding_goal * q90, 2),
        'recommendation': recommendation
    }


# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Kickstarter Counterfactual API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "POST /predict",
            "counterfactual": "POST /counterfactual",
            "categories": "GET /categories"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    load_models()
    return {
        "status": "healthy",
        "models_loaded": models is not None and 'ols' in models,
        "data_loaded": data is not None and len(data) > 0
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(campaign: CampaignInput):
    """
    Get prediction for a Kickstarter campaign.
    
    Returns expected funding, funding ratio, and recommendations.
    """
    try:
        result = get_prediction(campaign)
        return PredictionOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/counterfactual", response_model=CounterfactualOutput)
async def counterfactual(request: CounterfactualInput):
    """
    Get counterfactual analysis: what if you changed the price?
    
    Returns a demand curve showing predicted outcomes at different price points.
    """
    load_models()
    
    base = request.base_campaign
    
    # Generate price range
    if request.price_range:
        prices = request.price_range
    else:
        prices = [base.avg_reward_price * m for m in np.linspace(0.3, 2.5, 20)]
    
    # Calculate predictions at each price
    curve = []
    max_ratio = 0
    optimal_price = base.avg_reward_price
    
    for price in prices:
        modified = CampaignInput(
            category=base.category,
            funding_goal=base.funding_goal,
            campaign_duration=base.campaign_duration,
            avg_reward_price=price,
            num_reward_tiers=base.num_reward_tiers,
            trend_index=base.trend_index
        )
        
        prediction = get_prediction(modified)
        ratio = prediction['funding_ratio']
        
        curve.append(CounterfactualPoint(price=round(price, 2), 
                                         predicted_funding_ratio=ratio))
        
        if ratio > max_ratio:
            max_ratio = ratio
            optimal_price = price
    
    # Determine price effect
    tsls_coef = models.get('tsls_coef', 0)
    if tsls_coef > 0.01:
        effect = "Higher prices slightly increase funding"
    elif tsls_coef < -0.01:
        effect = "Higher prices decrease funding"
    else:
        effect = "Price has minimal causal effect on funding"
    
    return CounterfactualOutput(
        demand_curve=curve,
        optimal_price=round(optimal_price, 2),
        current_price=base.avg_reward_price,
        price_effect=effect
    )


@app.get("/categories")
async def categories():
    """Get list of available campaign categories."""
    load_models()
    
    if data is not None and 'category' in data.columns:
        cats = data['category'].dropna().unique().tolist()
        return {"categories": sorted(cats)[:50]}  # Limit to 50
    
    return {"categories": ["Technology", "Games", "Design", "Film", "Music", "Art"]}


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
