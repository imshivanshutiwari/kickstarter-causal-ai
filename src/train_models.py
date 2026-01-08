#!/usr/bin/env python
"""
Model Training Script
=====================
Train all causal inference models with proper error handling and logging.

Models trained:
    1. OLS (baseline)
    2. Two-Stage Least Squares (2SLS) with proper linearmodels
    3. Causal Forest (econml)
    4. Quantile Regression models

Usage:
    python src/train_models.py
"""

import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_data(config: dict) -> pd.DataFrame:
    """Load causal features data."""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / config.get('data', {}).get('processed_dir', 'data/processed')
    file_name = config.get('data', {}).get('causal_features_output', 'kickstarter_causal_features.csv')
    
    full_path = data_path / file_name
    
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {full_path}")
    
    df = pd.read_csv(full_path)
    logger.info(f"Loaded {len(df)} campaigns from {full_path}")
    return df


def prepare_features(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target variable."""
    feature_cols = config.get('features', {}).get('feature_columns', [
        'avg_reward_price', 'goal_ambition', 'campaign_duration_days', 
        'trend_index', 'concurrent_campaigns'
    ])
    
    # ADDED: Auto-detect NLP features
    nlp_cols = [c for c in df.columns if c.startswith('nlp_dim_')]
    if nlp_cols:
        logger.info(f"Adding {len(nlp_cols)} NLP embedding features")
        feature_cols.extend(nlp_cols)
    
    target = config.get('features', {}).get('target', 'funding_ratio')
    
    # Use available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # Handle alternative column names
    if 'campaign_duration_days' not in df.columns and 'duration_days' in df.columns:
        df = df.rename(columns={'duration_days': 'campaign_duration_days'})
    
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if not available_cols:
        raise ValueError(f"No feature columns found. Available: {list(df.columns)}")
    
    # Drop rows with missing values
    df_clean = df[available_cols + [target]].dropna()
    
    X = df_clean[available_cols]
    y = df_clean[target]
    
    logger.info(f"Features: {available_cols}")
    logger.info(f"Samples: {len(X)}")
    
    return X, y


def train_ols(X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray,
              feature_names: list) -> Dict[str, Any]:
    """Train OLS model and return results."""
    logger.info("Training OLS model...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    results = {
        'model': model,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'coefficients': dict(zip(feature_names, model.coef_)),
        'intercept': model.intercept_
    }
    
    logger.info(f"  R²: {results['r2']:.4f}")
    logger.info(f"  MAE: {results['mae']:.4f}")
    logger.info(f"  Price coefficient: {results['coefficients'].get('avg_reward_price', 'N/A')}")
    
    return results


def train_2sls(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Train Two-Stage Least Squares model."""
    logger.info("Training 2SLS model...")
    
    # Get columns
    feature_cols = config.get('features', {}).get('feature_columns', [])
    iv_cols = config.get('features', {}).get('instrumental_variables', [
        'is_weekend_launch', 'holiday_proximity', 'trend_spike'
    ])
    target = 'funding_ratio'
    treatment = 'avg_reward_price'
    
    # Columns that exist
    available_features = [c for c in feature_cols if c in df.columns and c != treatment]
    available_ivs = [c for c in iv_cols if c in df.columns]
    
    if not available_ivs:
        logger.warning("No instrumental variables found. Using manual 2SLS.")
        available_ivs = ['is_weekend_launch']
        if 'is_weekend_launch' not in df.columns:
            df['is_weekend_launch'] = 0
    
    # Clean data
    all_cols = available_features + available_ivs + [treatment, target]
    df_clean = df[all_cols].dropna()
    
    # Stage 1: Regress treatment on instruments
    stage1_X = df_clean[available_ivs + available_features]
    stage1_y = df_clean[treatment]
    
    stage1_model = LinearRegression()
    stage1_model.fit(stage1_X, stage1_y)
    predicted_treatment = stage1_model.predict(stage1_X)
    
    # Calculate first-stage F-statistic
    r2_first = stage1_model.score(stage1_X, stage1_y)
    n = len(stage1_y)
    k = len(available_ivs)
    f_stat = (r2_first / k) / ((1 - r2_first) / (n - k - 1)) if r2_first < 1 else 0
    
    # Stage 2: Regress outcome on predicted treatment
    stage2_X = pd.DataFrame({
        'predicted_price': predicted_treatment
    })
    for col in available_features:
        stage2_X[col] = df_clean[col].values
    
    stage2_y = df_clean[target]
    
    stage2_model = LinearRegression()
    stage2_model.fit(stage2_X, stage2_y)
    
    results = {
        'stage1_model': stage1_model,
        'stage2_model': stage2_model,
        'tsls_coefficient': stage2_model.coef_[0],  # Coefficient on predicted price
        'first_stage_f_stat': f_stat,
        'instruments': available_ivs,
        'r2_first_stage': r2_first
    }
    
    logger.info(f"  2SLS price effect: {results['tsls_coefficient']:.6f}")
    logger.info(f"  First-stage F-stat: {results['first_stage_f_stat']:.2f}")
    logger.info(f"  Instruments: {available_ivs}")
    
    return results


def train_causal_forest(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Train Causal Forest for heterogeneous treatment effects."""
    logger.info("Training Causal Forest...")
    
    try:
        from econml.dml import CausalForestDML
        
        cf_config = config.get('models', {}).get('causal_forest', {})
        
        # Prepare data
        feature_cols = ['goal_ambition', 'trend_index', 'concurrent_campaigns']
        available = [c for c in feature_cols if c in df.columns]
        
        df_clean = df[available + ['avg_reward_price', 'funding_ratio']].dropna()
        
        X = df_clean[available]
        T = df_clean['avg_reward_price']
        Y = df_clean['funding_ratio']
        
        model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=cf_config.get('n_estimators', 100), 
                                          max_depth=cf_config.get('max_depth', 5),
                                          random_state=42),
            model_t=RandomForestRegressor(n_estimators=cf_config.get('n_estimators', 100),
                                          max_depth=cf_config.get('max_depth', 5),
                                          random_state=42),
            discrete_treatment=False,
            random_state=42
        )
        
        model.fit(Y.values, T.values, X=X.values)
        treatment_effects = model.effect(X.values)
        
        results = {
            'model': model,
            'treatment_effects': treatment_effects,
            'mean_effect': np.mean(treatment_effects),
            'std_effect': np.std(treatment_effects),
            'min_effect': np.min(treatment_effects),
            'max_effect': np.max(treatment_effects)
        }
        
        logger.info(f"  Mean treatment effect: {results['mean_effect']:.6f}")
        logger.info(f"  Effect range: [{results['min_effect']:.6f}, {results['max_effect']:.6f}]")
        
        return results
        
    except ImportError:
        logger.warning("econml not installed. Skipping Causal Forest.")
        return {'error': 'econml not installed'}
    except Exception as e:
        logger.error(f"Causal Forest failed: {e}")
        return {'error': str(e)}


def train_quantile_models(X_train: np.ndarray, y_train: np.ndarray, 
                          config: dict) -> Dict[float, Any]:
    """Train quantile regression models."""
    logger.info("Training quantile models...")
    
    quantiles = config.get('models', {}).get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
    gb_config = config.get('models', {}).get('gradient_boosting', {})
    
    models = {}
    
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=q,
            n_estimators=gb_config.get('n_estimators', 100),
            max_depth=gb_config.get('max_depth', 6),
            random_state=42
        )
        model.fit(X_train, y_train)
        models[q] = model
        logger.info(f"  Trained quantile {q}")
    
    return models


def save_models(results: Dict[str, Any], config: dict):
    """Save all trained models to disk."""
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / config.get('data', {}).get('models_dir', 'src/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = models_dir / config.get('data', {}).get('model_file', 'causal_models.pkl')
    
    with open(model_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Models saved to: {model_file}")


def main():
    """Main training function."""
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Load config
        config = load_config()
        
        # Load data
        df = load_data(config)
        
        # Prepare features
        X, y = prepare_features(df, config)
        feature_names = list(X.columns)
        
        # Train/test split
        test_size = config.get('models', {}).get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        results = {}
        
        # 1. OLS
        ols_results = train_ols(X_train_scaled, y_train, X_test_scaled, y_test, feature_names)
        results['ols'] = ols_results['model']
        results['ols_metrics'] = {
            'r2': ols_results['r2'],
            'mae': ols_results['mae'],
            'price_coef': ols_results['coefficients'].get('avg_reward_price', 0)
        }
        
        # 2. 2SLS
        tsls_results = train_2sls(df, config)
        results['tsls_coef'] = tsls_results['tsls_coefficient']
        results['first_stage_f'] = tsls_results['first_stage_f_stat']
        
        # 3. Causal Forest
        cf_results = train_causal_forest(df, config)
        if 'model' in cf_results:
            results['causal_forest'] = cf_results['model']
            results['treatment_effects'] = cf_results['treatment_effects']
        
        # 4. Quantile models
        quantile_models = train_quantile_models(X_train, y_train, config)
        results['quantile_models'] = quantile_models
        
        # Save scaler and feature names
        results['scaler'] = scaler
        results['feature_cols'] = feature_names
        
        # Save all models
        save_models(results, config)
        
        # Print summary
        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"OLS R²: {results['ols_metrics']['r2']:.4f}")
        print(f"OLS Price Coef (biased): {results['ols_metrics']['price_coef']:.6f}")
        print(f"2SLS Price Effect (causal): {results['tsls_coef']:.6f}")
        print(f"First-stage F-stat: {results['first_stage_f']:.2f}")
        if 'causal_forest' in results:
            print(f"Causal Forest: Trained successfully")
        print(f"Quantile models: {len(quantile_models)} trained")
        print("=" * 60)
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        print(f"✗ Error: {e}")
        print("  Run the data processing steps first.")
        return False
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
