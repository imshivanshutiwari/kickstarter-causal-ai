"""
Causal Inference Models for Kickstarter Counterfactual Analysis

This module implements causal inference methods to estimate the effect of
pricing strategies on campaign outcomes, handling endogeneity through
instrumental variables and heterogeneous treatment effects.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModels:
    """Baseline prediction models that ignore causal structure."""
    
    def __init__(self):
        self.linear_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_features(self, df: pd.DataFrame, 
                        feature_cols: List[str],
                        categorical_cols: List[str] = None) -> np.ndarray:
        """Prepare features for modeling.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            categorical_cols: List of categorical columns to encode
            
        Returns:
            Feature matrix
        """
        X = df[feature_cols].copy()
        
        if categorical_cols:
            for col in categorical_cols:
                if col in X.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                    else:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        return X.values
    
    def fit_linear(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Fit linear regression baseline.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Model results dictionary
        """
        X_scaled = self.scaler.fit_transform(X)
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_scaled, y)
        
        y_pred = self.linear_model.predict(X_scaled)
        
        return {
            'model': 'Linear Regression',
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'coefficients': self.linear_model.coef_,
        }
    
    def fit_random_forest(self, X: np.ndarray, y: np.ndarray, 
                          n_estimators: int = 100) -> Dict:
        """Fit Random Forest baseline.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_estimators: Number of trees
            
        Returns:
            Model results dictionary
        """
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)
        
        y_pred = self.rf_model.predict(X)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.rf_model, X, y, cv=5, scoring='r2')
        
        return {
            'model': 'Random Forest',
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'feature_importance': self.rf_model.feature_importances_,
        }
    
    def predict(self, X: np.ndarray, model: str = 'rf') -> np.ndarray:
        """Make predictions using fitted model.
        
        Args:
            X: Feature matrix
            model: 'linear' or 'rf'
            
        Returns:
            Predictions
        """
        if model == 'linear':
            X_scaled = self.scaler.transform(X)
            return self.linear_model.predict(X_scaled)
        else:
            return self.rf_model.predict(X)


class TwoStageLeastSquares:
    """
    Two-Stage Least Squares (2SLS) estimator for causal effect estimation.
    
    This handles endogeneity of treatment (pricing) using instrumental variables
    that affect outcomes only through treatment.
    """
    
    def __init__(self):
        self.first_stage_model = None
        self.second_stage_model = None
        self.results = {}
        
    def fit(self, 
            df: pd.DataFrame,
            outcome_col: str,
            endogenous_col: str,
            instrument_cols: List[str],
            exogenous_cols: List[str]) -> Dict:
        """
        Fit 2SLS model.
        
        Args:
            df: Input DataFrame
            outcome_col: Name of outcome variable (Y)
            endogenous_col: Name of endogenous treatment variable (T)
            instrument_cols: Names of instrumental variables (Z)
            exogenous_cols: Names of exogenous control variables (X)
            
        Returns:
            Results dictionary with coefficients and diagnostics
        """
        # Prepare data
        Y = df[outcome_col].values
        T = df[endogenous_col].values
        Z = df[instrument_cols].values
        X = df[exogenous_cols].values if exogenous_cols else None
        
        # First Stage: Regress endogenous variable on instruments and controls
        if X is not None:
            first_stage_X = np.column_stack([Z, X])
            first_stage_X_with_const = sm.add_constant(first_stage_X)
        else:
            first_stage_X_with_const = sm.add_constant(Z)
        
        self.first_stage_model = sm.OLS(T, first_stage_X_with_const).fit()
        T_hat = self.first_stage_model.predict(first_stage_X_with_const)
        
        # First stage diagnostics
        first_stage_f_stat = self.first_stage_model.fvalue
        first_stage_r2 = self.first_stage_model.rsquared
        
        # Second Stage: Regress outcome on predicted treatment and controls
        if X is not None:
            second_stage_X = np.column_stack([T_hat, X])
        else:
            second_stage_X = T_hat.reshape(-1, 1)
        
        second_stage_X_with_const = sm.add_constant(second_stage_X)
        self.second_stage_model = sm.OLS(Y, second_stage_X_with_const).fit()
        
        # Treatment effect is the coefficient on T_hat
        treatment_effect = self.second_stage_model.params[1]
        treatment_se = self.second_stage_model.bse[1]
        
        self.results = {
            'treatment_effect': treatment_effect,
            'treatment_se': treatment_se,
            'treatment_t_stat': treatment_effect / treatment_se,
            'treatment_pvalue': self.second_stage_model.pvalues[1],
            'first_stage_f_stat': first_stage_f_stat,
            'first_stage_r2': first_stage_r2,
            'second_stage_r2': self.second_stage_model.rsquared,
            'instrument_strength': 'Strong' if first_stage_f_stat > 10 else 'Weak',
            'n_observations': len(Y),
        }
        
        logger.info(f"2SLS Results:")
        logger.info(f"  Treatment Effect: {treatment_effect:.4f} (SE: {treatment_se:.4f})")
        logger.info(f"  First Stage F-stat: {first_stage_f_stat:.2f}")
        logger.info(f"  Instrument Strength: {self.results['instrument_strength']}")
        
        return self.results
    
    def get_first_stage_summary(self) -> str:
        """Get first stage regression summary."""
        if self.first_stage_model:
            return self.first_stage_model.summary().as_text()
        return "Model not fitted"
    
    def get_second_stage_summary(self) -> str:
        """Get second stage regression summary."""
        if self.second_stage_model:
            return self.second_stage_model.summary().as_text()
        return "Model not fitted"


class CausalForestEstimator:
    """
    Causal Forest for Heterogeneous Treatment Effect estimation.
    
    Uses double machine learning (DML) approach to estimate how treatment
    effects vary across different subpopulations.
    """
    
    def __init__(self, n_estimators: int = 100):
        self.n_estimators = n_estimators
        self.model_y = None  # Model for outcome
        self.model_t = None  # Model for treatment
        self.effect_model = None  # Model for treatment effect
        self.treatment_effects = None
        
    def fit(self, 
            df: pd.DataFrame,
            outcome_col: str,
            treatment_col: str,
            feature_cols: List[str]) -> Dict:
        """
        Fit Causal Forest using Double Machine Learning.
        
        This is a simplified implementation. For production, use econml library.
        
        Args:
            df: Input DataFrame
            outcome_col: Outcome variable name
            treatment_col: Treatment variable name
            feature_cols: Feature column names for heterogeneity
            
        Returns:
            Results dictionary
        """
        Y = df[outcome_col].values
        T = df[treatment_col].values
        X = df[feature_cols].values
        
        # Step 1: Predict outcome from features (nuisance model)
        self.model_y = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.model_y.fit(X, Y)
        Y_hat = self.model_y.predict(X)
        Y_residual = Y - Y_hat
        
        # Step 2: Predict treatment from features (propensity-like)
        self.model_t = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.model_t.fit(X, T)
        T_hat = self.model_t.predict(X)
        T_residual = T - T_hat
        
        # Step 3: Estimate treatment effect as coefficient in residual regression
        # For each observation, effect = Y_residual / T_residual (when T_residual != 0)
        # Use local linear regression for stability
        
        # Simple approach: estimate average effect per feature bin
        n_samples = len(Y)
        self.treatment_effects = np.zeros(n_samples)
        
        # Use a model to predict effects based on features
        # Effect = correlation between residuals, weighted by treatment variation
        valid_mask = np.abs(T_residual) > 0.01
        effects = np.where(valid_mask, Y_residual / T_residual, 0)
        
        # Clip extreme effects for stability
        effects = np.clip(effects, np.percentile(effects, 5), np.percentile(effects, 95))
        
        # Smooth effects using Random Forest
        self.effect_model = RandomForestRegressor(
            n_estimators=self.n_estimators // 2,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42
        )
        self.effect_model.fit(X, effects)
        self.treatment_effects = self.effect_model.predict(X)
        
        # Average Treatment Effect (ATE)
        ate = np.mean(self.treatment_effects)
        ate_std = np.std(self.treatment_effects)
        
        results = {
            'ATE': ate,
            'ATE_std': ate_std,
            'effect_min': np.min(self.treatment_effects),
            'effect_max': np.max(self.treatment_effects),
            'effect_percentiles': {
                '10%': np.percentile(self.treatment_effects, 10),
                '25%': np.percentile(self.treatment_effects, 25),
                '50%': np.percentile(self.treatment_effects, 50),
                '75%': np.percentile(self.treatment_effects, 75),
                '90%': np.percentile(self.treatment_effects, 90),
            },
            'n_observations': n_samples,
        }
        
        logger.info(f"Causal Forest Results:")
        logger.info(f"  Average Treatment Effect: {ate:.4f} (± {ate_std:.4f})")
        logger.info(f"  Effect Range: [{results['effect_min']:.4f}, {results['effect_max']:.4f}]")
        
        return results
    
    def predict_effects(self, X: np.ndarray) -> np.ndarray:
        """Predict treatment effects for new observations."""
        if self.effect_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.effect_model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance for effect heterogeneity."""
        if self.effect_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.effect_model.feature_importances_


class QuantileRegressionModel:
    """
    Quantile regression for uncertainty quantification.
    
    Predicts the full distribution of outcomes, not just the mean.
    """
    
    def __init__(self, quantiles: List[float] = None):
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.models = {}
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit quantile regression models.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Results dictionary
        """
        X_scaled = self.scaler.fit_transform(X)
        
        results = {'quantiles': {}}
        
        for q in self.quantiles:
            # Use Gradient Boosting with quantile loss
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42
            )
            model.fit(X_scaled, y)
            self.models[q] = model
            
            y_pred = model.predict(X_scaled)
            results['quantiles'][q] = {
                'predictions': y_pred,
                'mean_pred': np.mean(y_pred),
                'std_pred': np.std(y_pred),
            }
        
        # Calculate prediction intervals
        lower = self.models[0.1].predict(X_scaled) if 0.1 in self.models else None
        upper = self.models[0.9].predict(X_scaled) if 0.9 in self.models else None
        
        if lower is not None and upper is not None:
            interval_width = np.mean(upper - lower)
            results['avg_80_interval_width'] = interval_width
        
        logger.info(f"Quantile Regression Results:")
        logger.info(f"  Quantiles fitted: {self.quantiles}")
        if 'avg_80_interval_width' in results:
            logger.info(f"  Average 80% interval width: {results['avg_80_interval_width']:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Predict quantiles for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping quantile to predictions
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X_scaled)
        
        return predictions
    
    def get_prediction_interval(self, X: np.ndarray, 
                                lower_q: float = 0.1, 
                                upper_q: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction interval bounds."""
        X_scaled = self.scaler.transform(X)
        
        lower = self.models[lower_q].predict(X_scaled)
        upper = self.models[upper_q].predict(X_scaled)
        
        return lower, upper


class CounterfactualPredictor:
    """
    Main class for counterfactual prediction combining all models.
    """
    
    def __init__(self):
        self.baseline = BaselineModels()
        self.tsls = TwoStageLeastSquares()
        self.causal_forest = CausalForestEstimator()
        self.quantile_model = QuantileRegressionModel()
        self.feature_cols = []
        self.treatment_col = None
        self.outcome_col = None
        
    def fit_all_models(self, 
                       df: pd.DataFrame,
                       outcome_col: str = 'funding_ratio',
                       treatment_col: str = 'avg_reward_price',
                       feature_cols: List[str] = None,
                       instrument_cols: List[str] = None,
                       categorical_cols: List[str] = None) -> Dict:
        """
        Fit all causal inference models.
        
        Args:
            df: Input DataFrame
            outcome_col: Outcome variable
            treatment_col: Treatment/endogenous variable
            feature_cols: Control features
            instrument_cols: Instrumental variables for 2SLS
            categorical_cols: Categorical columns to encode
            
        Returns:
            Combined results from all models
        """
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.feature_cols = feature_cols or []
        
        results = {}
        
        # Default columns if not specified
        if feature_cols is None:
            feature_cols = [
                'goal', 'duration_days', 'num_reward_tiers', 
                'trend_index', 'concurrent_campaigns', 'description_length'
            ]
        
        if instrument_cols is None:
            instrument_cols = ['day_of_week', 'holiday_proximity']
        
        if categorical_cols is None:
            categorical_cols = ['category']
        
        # Prepare feature matrix
        all_features = feature_cols + [treatment_col]
        X = self.baseline.prepare_features(df, all_features, categorical_cols)
        y = df[outcome_col].values
        
        # 1. Baseline Models
        logger.info("\n" + "="*50)
        logger.info("Fitting Baseline Models...")
        logger.info("="*50)
        
        results['linear'] = self.baseline.fit_linear(X, y)
        results['random_forest'] = self.baseline.fit_random_forest(X, y)
        
        # 2. Two-Stage Least Squares
        logger.info("\n" + "="*50)
        logger.info("Fitting 2SLS Model...")
        logger.info("="*50)
        
        # Check if instruments exist in data
        available_instruments = [col for col in instrument_cols if col in df.columns]
        exogenous = [col for col in feature_cols if col in df.columns and col != treatment_col]
        
        if available_instruments:
            results['2sls'] = self.tsls.fit(
                df, 
                outcome_col, 
                treatment_col,
                available_instruments,
                exogenous
            )
        else:
            logger.warning("No valid instruments found for 2SLS")
            results['2sls'] = {'error': 'No valid instruments'}
        
        # 3. Causal Forest
        logger.info("\n" + "="*50)
        logger.info("Fitting Causal Forest...")
        logger.info("="*50)
        
        heterogeneity_features = [col for col in feature_cols if col in df.columns]
        if categorical_cols:
            for col in categorical_cols:
                if col in df.columns:
                    heterogeneity_features.append(col)
        
        # Prepare features with encoding
        X_cf = self.baseline.prepare_features(df, heterogeneity_features, categorical_cols)
        # Convert back to DataFrame for causal forest
        df_cf = pd.DataFrame(X_cf, columns=heterogeneity_features)
        df_cf[outcome_col] = df[outcome_col].values
        df_cf[treatment_col] = df[treatment_col].values
        
        results['causal_forest'] = self.causal_forest.fit(
            df_cf,
            outcome_col,
            treatment_col,
            heterogeneity_features
        )
        
        # 4. Quantile Regression
        logger.info("\n" + "="*50)
        logger.info("Fitting Quantile Regression...")
        logger.info("="*50)
        
        results['quantile'] = self.quantile_model.fit(X, y)
        
        return results
    
    def predict_counterfactual(self, 
                               campaign: Dict,
                               price_change_pct: float = -20) -> Dict:
        """
        Predict counterfactual outcome for a campaign.
        
        Args:
            campaign: Dictionary with campaign features
            price_change_pct: Percentage change in price (e.g., -20 for 20% reduction)
            
        Returns:
            Counterfactual predictions
        """
        # Current prediction
        current_features = self._prepare_single_campaign(campaign)
        current_pred = self.baseline.predict(current_features, 'rf')[0]
        
        # Counterfactual with price change
        cf_campaign = campaign.copy()
        cf_campaign[self.treatment_col] *= (1 + price_change_pct / 100)
        cf_features = self._prepare_single_campaign(cf_campaign)
        cf_pred = self.baseline.predict(cf_features, 'rf')[0]
        
        # Get treatment effect from causal forest
        treatment_effect = self.causal_forest.predict_effects(current_features)[0]
        
        # Get prediction interval
        lower, upper = self.quantile_model.get_prediction_interval(current_features)
        
        return {
            'current_prediction': current_pred,
            'counterfactual_prediction': cf_pred,
            'naive_effect': cf_pred - current_pred,
            'causal_effect': treatment_effect * (price_change_pct / 100) * campaign.get(self.treatment_col, 100),
            'prediction_interval': (lower[0], upper[0]),
            'price_change_pct': price_change_pct,
        }
    
    def _prepare_single_campaign(self, campaign: Dict) -> np.ndarray:
        """Prepare single campaign for prediction."""
        features = []
        for col in self.feature_cols + [self.treatment_col]:
            val = campaign.get(col, 0)
            features.append(val)
        return np.array(features).reshape(1, -1)
    
    def save_models(self, path: str):
        """Save all fitted models to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'baseline': self.baseline,
                'tsls': self.tsls,
                'causal_forest': self.causal_forest,
                'quantile_model': self.quantile_model,
                'feature_cols': self.feature_cols,
                'treatment_col': self.treatment_col,
                'outcome_col': self.outcome_col,
            }, f)
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load models from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.baseline = data['baseline']
            self.tsls = data['tsls']
            self.causal_forest = data['causal_forest']
            self.quantile_model = data['quantile_model']
            self.feature_cols = data['feature_cols']
            self.treatment_col = data['treatment_col']
            self.outcome_col = data['outcome_col']
        logger.info(f"Models loaded from {path}")


def main():
    """Main function to demonstrate causal inference pipeline."""
    from pathlib import Path
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "raw" / "kickstarter_raw_data.csv"
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run scraping.py first to generate data")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} campaigns")
    
    # Initialize predictor
    predictor = CounterfactualPredictor()
    
    # Feature configuration
    feature_cols = [
        'goal', 'duration_days', 'num_reward_tiers',
        'trend_index', 'concurrent_campaigns', 'description_length'
    ]
    instrument_cols = ['day_of_week', 'holiday_proximity']
    categorical_cols = ['category']
    
    # Fit all models
    results = predictor.fit_all_models(
        df,
        outcome_col='funding_ratio',
        treatment_col='avg_reward_price',
        feature_cols=feature_cols,
        instrument_cols=instrument_cols,
        categorical_cols=categorical_cols
    )
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nBaseline Linear Regression R²: {results['linear']['r2']:.4f}")
    print(f"Baseline Random Forest R²: {results['random_forest']['r2']:.4f}")
    print(f"Random Forest CV R² (mean ± std): {results['random_forest']['cv_r2_mean']:.4f} ± {results['random_forest']['cv_r2_std']:.4f}")
    
    if '2sls' in results and 'treatment_effect' in results['2sls']:
        print(f"\n2SLS Treatment Effect: {results['2sls']['treatment_effect']:.6f}")
        print(f"2SLS p-value: {results['2sls']['treatment_pvalue']:.4f}")
        print(f"Instrument Strength: {results['2sls']['instrument_strength']}")
    
    print(f"\nCausal Forest ATE: {results['causal_forest']['ATE']:.6f}")
    print(f"Effect heterogeneity (std): {results['causal_forest']['ATE_std']:.6f}")
    
    # Save models
    model_path = Path(__file__).parent.parent / "data" / "processed" / "causal_models.pkl"
    predictor.save_models(str(model_path))
    
    return predictor, results


if __name__ == "__main__":
    main()
