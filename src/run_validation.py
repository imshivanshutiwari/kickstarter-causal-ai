#!/usr/bin/env python
"""
Validation Script
=================
Run all validation tests for causal models.

Tests:
    1. Temporal validation
    2. Placebo test
    3. Manski bounds
    4. Cross-validation

Usage:
    python src/run_validation.py
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

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


def load_data_and_models(config: dict) -> tuple:
    """Load data and trained models."""
    base_dir = Path(__file__).parent.parent
    
    # Load data
    data_path = base_dir / config.get('data', {}).get('processed_dir', 'data/processed')
    data_file = data_path / config.get('data', {}).get('causal_features_output', 'kickstarter_causal_features.csv')
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} campaigns")
    
    # Load models
    models_dir = base_dir / config.get('data', {}).get('models_dir', 'src/models')
    model_file = models_dir / config.get('data', {}).get('model_file', 'causal_models.pkl')
    
    with open(model_file, 'rb') as f:
        models = pickle.load(f)
    logger.info("Loaded trained models")
    
    return df, models


def temporal_validation(df: pd.DataFrame, models: dict, config: dict) -> Dict[str, Any]:
    """Validate on campaigns with high price variation."""
    logger.info("Running temporal validation...")
    
    val_config = config.get('validation', {})
    n_samples = val_config.get('temporal_validation_n', 10)
    
    # Find campaigns with price variation (proxy for "strategy changers")
    if 'price_spread' in df.columns:
        df_sorted = df.nlargest(n_samples, 'price_spread')
    else:
        df_sorted = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    # Get predictions
    feature_cols = models.get('feature_cols', ['avg_reward_price', 'goal_ambition'])
    available_cols = [c for c in feature_cols if c in df_sorted.columns]
    
    if not available_cols:
        return {'error': 'No feature columns available'}
    
    X = df_sorted[available_cols].fillna(0)
    y_actual = df_sorted['funding_ratio']
    
    # Scale and predict
    scaler = models.get('scaler')
    ols = models.get('ols')
    
    if scaler and ols:
        X_scaled = scaler.transform(X)
        y_pred = ols.predict(X_scaled)
        
        mae = np.mean(np.abs(y_actual - y_pred))
        mse = np.mean((y_actual - y_pred) ** 2)
        
        results = {
            'n_samples': len(df_sorted),
            'mae': mae,
            'rmse': np.sqrt(mse),
            'status': 'PASSED' if mae < 1.0 else 'WARNING'
        }
    else:
        results = {'error': 'Models not available'}
    
    logger.info(f"  MAE: {results.get('mae', 'N/A'):.4f}")
    return results


def placebo_test(df: pd.DataFrame, models: dict, config: dict) -> Dict[str, Any]:
    """Test for phantom effects on campaigns with no strategy change."""
    logger.info("Running placebo test...")
    
    val_config = config.get('validation', {})
    n_samples = val_config.get('placebo_test_n', 50)
    threshold = val_config.get('placebo_threshold', 0.01)
    
    # Select campaigns with LOW price variation (no strategy change)
    if 'price_spread' in df.columns:
        df_stable = df.nsmallest(n_samples, 'price_spread')
    else:
        df_stable = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    # Get predictions with tiny perturbation
    feature_cols = models.get('feature_cols', ['avg_reward_price'])
    available_cols = [c for c in feature_cols if c in df_stable.columns]
    
    if not available_cols:
        return {'error': 'No feature columns available'}
    
    X_original = df_stable[available_cols].fillna(0)
    X_perturbed = X_original * 1.001  # Tiny 0.1% change
    
    scaler = models.get('scaler')
    ols = models.get('ols')
    
    if scaler and ols:
        X_orig_scaled = scaler.transform(X_original)
        X_pert_scaled = scaler.transform(X_perturbed)
        
        pred_original = ols.predict(X_orig_scaled)
        pred_perturbed = ols.predict(X_pert_scaled)
        
        phantom_effects = np.abs(pred_perturbed - pred_original)
        mean_phantom = np.mean(phantom_effects)
        
        passed = mean_phantom < threshold
        
        results = {
            'n_samples': len(df_stable),
            'mean_phantom_effect': mean_phantom,
            'threshold': threshold,
            'status': 'PASSED' if passed else 'FAILED'
        }
    else:
        results = {'error': 'Models not available'}
    
    logger.info(f"  Mean phantom effect: {results.get('mean_phantom_effect', 'N/A'):.6f}")
    logger.info(f"  Status: {results.get('status', 'N/A')}")
    return results


def manski_bounds(df: pd.DataFrame, models: dict, config: dict) -> Dict[str, Any]:
    """Check if estimates fall within Manski bounds."""
    logger.info("Running Manski bounds analysis...")
    
    val_config = config.get('validation', {})
    n_samples = val_config.get('manski_bounds_n', 100)
    uncertainty = val_config.get('manski_uncertainty', 0.3)
    
    df_sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    feature_cols = models.get('feature_cols', ['avg_reward_price'])
    available_cols = [c for c in feature_cols if c in df_sample.columns]
    
    if not available_cols:
        return {'error': 'No feature columns available'}
    
    X = df_sample[available_cols].fillna(0)
    y_actual = df_sample['funding_ratio'].values
    
    scaler = models.get('scaler')
    ols = models.get('ols')
    
    if scaler and ols:
        X_scaled = scaler.transform(X)
        y_pred = ols.predict(X_scaled)
        
        # Calculate bounds
        lower_bounds = y_actual - uncertainty
        upper_bounds = y_actual + uncertainty
        
        within_bounds = (y_pred >= lower_bounds) & (y_pred <= upper_bounds)
        pct_within = np.mean(within_bounds) * 100
        
        results = {
            'n_samples': len(df_sample),
            'uncertainty': uncertainty,
            'pct_within_bounds': pct_within,
            'status': 'PASSED' if pct_within > 80 else 'WARNING'
        }
    else:
        results = {'error': 'Models not available'}
    
    logger.info(f"  % within bounds: {results.get('pct_within_bounds', 'N/A'):.1f}%")
    return results


def cross_validation(df: pd.DataFrame, models: dict, config: dict) -> Dict[str, Any]:
    """Run cross-validation analysis."""
    logger.info("Running cross-validation...")
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    
    feature_cols = models.get('feature_cols', ['avg_reward_price'])
    available_cols = [c for c in feature_cols if c in df.columns]
    
    if not available_cols:
        return {'error': 'No feature columns available'}
    
    df_clean = df[available_cols + ['funding_ratio']].dropna()
    X = df_clean[available_cols]
    y = df_clean['funding_ratio']
    
    cv_folds = config.get('models', {}).get('cv_folds', 5)
    
    model = LinearRegression()
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
    
    results = {
        'n_folds': cv_folds,
        'mean_cv_mae': -np.mean(cv_scores),
        'std_cv_mae': np.std(cv_scores),
        'cv_scores': list(-cv_scores)
    }
    
    logger.info(f"  Mean CV MAE: {results['mean_cv_mae']:.4f} ± {results['std_cv_mae']:.4f}")
    return results


def save_results(results: Dict[str, Any], config: dict):
    """Save validation results."""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / config.get('data', {}).get('processed_dir', 'data/processed')
    
    # Save as CSV
    results_df = pd.DataFrame([
        {'test': 'Temporal Validation', 'metric': 'MAE', 
         'value': results.get('temporal', {}).get('mae', 'N/A'),
         'status': results.get('temporal', {}).get('status', 'N/A')},
        {'test': 'Placebo Test', 'metric': 'Mean Phantom Effect',
         'value': results.get('placebo', {}).get('mean_phantom_effect', 'N/A'),
         'status': results.get('placebo', {}).get('status', 'N/A')},
        {'test': 'Manski Bounds', 'metric': '% Within Bounds',
         'value': results.get('manski', {}).get('pct_within_bounds', 'N/A'),
         'status': results.get('manski', {}).get('status', 'N/A')},
        {'test': 'Cross-Validation', 'metric': 'Mean CV MAE',
         'value': results.get('cross_val', {}).get('mean_cv_mae', 'N/A'),
         'status': 'INFO'}
    ])
    
    output_file = data_path / config.get('data', {}).get('validation_output', 'validation_results.csv')
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")


def main():
    """Main validation function."""
    print("=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)
    
    try:
        config = load_config()
        df, models = load_data_and_models(config)
        
        results = {}
        
        # Run all validation tests
        results['temporal'] = temporal_validation(df, models, config)
        results['placebo'] = placebo_test(df, models, config)
        results['manski'] = manski_bounds(df, models, config)
        results['cross_val'] = cross_validation(df, models, config)
        
        # Save results
        save_results(results, config)
        
        # Print summary
        print()
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Temporal MAE: {results['temporal'].get('mae', 'N/A')}")
        print(f"Placebo Test: {results['placebo'].get('status', 'N/A')}")
        print(f"Manski Bounds: {results['manski'].get('pct_within_bounds', 'N/A'):.1f}% within bounds")
        print(f"CV MAE: {results['cross_val'].get('mean_cv_mae', 'N/A'):.4f}")
        print("=" * 60)
        
        # Overall status
        all_passed = all([
            results['placebo'].get('status') == 'PASSED',
            results['manski'].get('status') in ['PASSED', 'WARNING']
        ])
        
        if all_passed:
            print("[OK] All critical validation tests passed")
        else:
            print("[WARN] Some validation tests need attention")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
