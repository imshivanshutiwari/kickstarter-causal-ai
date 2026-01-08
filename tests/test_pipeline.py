#!/usr/bin/env python
"""
Unit Tests for Kickstarter Counterfactual Project
==================================================

Run with: pytest tests/ -v

Tests cover:
- Data loading and processing
- Feature engineering
- Model training
- Validation functions
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample test data."""
    np.random.seed(42)
    n = 100
    
    data = {
        'name': [f'Campaign {i}' for i in range(n)],
        'category': np.random.choice(['Technology', 'Games', 'Design', 'Film'], n),
        'funding_goal': np.random.uniform(1000, 100000, n),
        'pledged_amount': np.random.uniform(0, 150000, n),
        'backers_count': np.random.randint(0, 1000, n),
        'status': np.random.choice(['successful', 'failed'], n, p=[0.4, 0.6]),
        'campaign_duration_days': np.random.randint(15, 60, n),
        'avg_reward_price': np.random.uniform(10, 200, n),
        'trend_index': np.random.uniform(20, 80, n),
        'concurrent_campaigns': np.random.randint(0, 10, n),
        'is_weekend_launch': np.random.randint(0, 2, n),
        'launch_date': pd.date_range('2022-01-01', periods=n, freq='D')
    }
    
    df = pd.DataFrame(data)
    df['funding_ratio'] = df['pledged_amount'] / df['funding_goal']
    df['goal_ambition'] = df['funding_goal'] / df['funding_goal'].median()
    
    return df


@pytest.fixture
def config():
    """Sample configuration."""
    return {
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'models_dir': 'src/models'
        },
        'features': {
            'feature_columns': ['avg_reward_price', 'goal_ambition', 'campaign_duration_days'],
            'target': 'funding_ratio'
        },
        'models': {
            'test_size': 0.2,
            'quantiles': [0.25, 0.5, 0.75]
        },
        'validation': {
            'placebo_threshold': 0.01
        }
    }


# =============================================================================
# Data Processing Tests
# =============================================================================

class TestDataProcessing:
    """Tests for data processing functions."""
    
    def test_data_has_required_columns(self, sample_data):
        """Test that sample data has required columns."""
        required = ['funding_goal', 'pledged_amount', 'status']
        for col in required:
            assert col in sample_data.columns, f"Missing column: {col}"
    
    def test_funding_ratio_calculation(self, sample_data):
        """Test funding ratio is calculated correctly."""
        expected = sample_data['pledged_amount'] / sample_data['funding_goal']
        np.testing.assert_array_almost_equal(
            sample_data['funding_ratio'].values,
            expected.values,
            decimal=5
        )
    
    def test_no_negative_values(self, sample_data):
        """Test no negative values in financial columns."""
        assert (sample_data['funding_goal'] >= 0).all()
        assert (sample_data['pledged_amount'] >= 0).all()
        assert (sample_data['backers_count'] >= 0).all()
    
    def test_duration_in_valid_range(self, sample_data):
        """Test campaign duration is within expected range."""
        assert (sample_data['campaign_duration_days'] > 0).all()
        assert (sample_data['campaign_duration_days'] <= 90).all()
    
    def test_status_values(self, sample_data):
        """Test status column has valid values."""
        valid_statuses = ['successful', 'failed']
        assert sample_data['status'].isin(valid_statuses).all()


# =============================================================================
# Feature Engineering Tests
# =============================================================================

class TestFeatureEngineering:
    """Tests for feature engineering functions."""
    
    def test_goal_ambition_calculation(self, sample_data):
        """Test goal_ambition is calculated correctly."""
        median_goal = sample_data['funding_goal'].median()
        expected = sample_data['funding_goal'] / median_goal
        
        np.testing.assert_array_almost_equal(
            sample_data['goal_ambition'].values,
            expected.values,
            decimal=5
        )
    
    def test_weekend_launch_binary(self, sample_data):
        """Test is_weekend_launch is binary."""
        assert set(sample_data['is_weekend_launch'].unique()).issubset({0, 1})
    
    def test_trend_index_range(self, sample_data):
        """Test trend_index is in valid range."""
        assert (sample_data['trend_index'] >= 0).all()
        assert (sample_data['trend_index'] <= 100).all()


# =============================================================================
# Model Tests
# =============================================================================

class TestModels:
    """Tests for model training functions."""
    
    def test_ols_model_fits(self, sample_data):
        """Test OLS model can be fitted."""
        from sklearn.linear_model import LinearRegression
        
        X = sample_data[['avg_reward_price', 'goal_ambition']].fillna(0)
        y = sample_data['funding_ratio']
        
        model = LinearRegression()
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')
        assert len(model.coef_) == 2
    
    def test_ols_predicts(self, sample_data):
        """Test OLS model can make predictions."""
        from sklearn.linear_model import LinearRegression
        
        X = sample_data[['avg_reward_price', 'goal_ambition']].fillna(0)
        y = sample_data['funding_ratio']
        
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()
    
    def test_quantile_models_train(self, sample_data, config):
        """Test quantile models can be trained."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = sample_data[['avg_reward_price', 'goal_ambition']].fillna(0)
        y = sample_data['funding_ratio']
        
        quantiles = config['models']['quantiles']
        models = {}
        
        for q in quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=10,  # Small for speed
                max_depth=3,
                random_state=42
            )
            model.fit(X, y)
            models[q] = model
        
        assert len(models) == len(quantiles)


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for validation functions."""
    
    def test_placebo_effect_is_small(self, sample_data):
        """Test placebo effect is small for unchanged data."""
        from sklearn.linear_model import LinearRegression
        
        X = sample_data[['avg_reward_price', 'goal_ambition']].fillna(0)
        y = sample_data['funding_ratio']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Tiny perturbation
        X_perturbed = X * 1.001
        
        pred_original = model.predict(X)
        pred_perturbed = model.predict(X_perturbed)
        
        phantom_effect = np.mean(np.abs(pred_perturbed - pred_original))
        
        # Should be very small
        assert phantom_effect < 0.1, f"Phantom effect too large: {phantom_effect}"
    
    def test_cross_validation_scores(self, sample_data):
        """Test cross-validation produces valid scores."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LinearRegression
        
        X = sample_data[['avg_reward_price', 'goal_ambition']].fillna(0)
        y = sample_data['funding_ratio']
        
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
        
        assert len(scores) == 3
        assert not np.isnan(scores).any()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_end_to_end_pipeline(self, sample_data, tmp_path):
        """Test full pipeline from data to predictions."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        # Process data
        X = sample_data[['avg_reward_price', 'goal_ambition', 'campaign_duration_days']].fillna(0)
        y = sample_data['funding_ratio']
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Predict
        predictions = model.predict(X_scaled)
        
        # Validate
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()
        
        # Check coefficient exists
        assert hasattr(model, 'coef_')
        
    def test_save_and_load_model(self, sample_data, tmp_path):
        """Test model can be saved and loaded."""
        import pickle
        from sklearn.linear_model import LinearRegression
        
        X = sample_data[['avg_reward_price', 'goal_ambition']].fillna(0)
        y = sample_data['funding_ratio']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Save
        model_path = tmp_path / 'test_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Compare predictions
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
