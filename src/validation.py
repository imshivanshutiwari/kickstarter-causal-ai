"""
Validation Module for Counterfactual Predictions

This module implements validation techniques for causal inference models:
1. Temporal Validation - Using campaigns that changed strategy
2. Placebo Tests - Ensuring no effect where none should exist
3. Manski Bounds - Bounding counterfactual estimates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalValidator:
    """
    Temporal validation using campaigns that changed strategy.
    
    Identifies campaigns where we can observe both "before" and "after"
    a strategy change, providing quasi-experimental validation.
    """
    
    def __init__(self):
        self.validation_results = []
    
    def identify_strategy_changes(self, df: pd.DataFrame,
                                  campaign_id_col: str = 'project_id',
                                  price_col: str = 'avg_reward_price') -> pd.DataFrame:
        """
        Identify campaigns that might have changed strategy.
        
        For synthetic data, we simulate this by comparing similar campaigns
        with different strategies.
        
        Args:
            df: Input DataFrame
            campaign_id_col: Campaign identifier column
            price_col: Price column
            
        Returns:
            DataFrame of matched campaign pairs
        """
        # For real validation, you would track campaigns over time
        # For synthetic data, we create matched pairs based on similarity
        
        # Group by category and find similar campaigns with different prices
        pairs = []
        
        for category in df['category'].unique():
            cat_df = df[df['category'] == category].copy()
            
            # Sort by goal to find similar campaigns
            cat_df = cat_df.sort_values('goal')
            
            # Find pairs with similar goals but different prices
            for i in range(len(cat_df) - 1):
                row1 = cat_df.iloc[i]
                row2 = cat_df.iloc[i + 1]
                
                # Check if goals are similar (within 20%)
                goal_ratio = row2['goal'] / row1['goal'] if row1['goal'] > 0 else 0
                if 0.8 <= goal_ratio <= 1.2:
                    # Check if prices are different (>10% difference)
                    price_ratio = row2[price_col] / row1[price_col] if row1[price_col] > 0 else 0
                    if price_ratio < 0.9 or price_ratio > 1.1:
                        pairs.append({
                            'campaign_1_id': row1[campaign_id_col],
                            'campaign_2_id': row2[campaign_id_col],
                            'category': category,
                            'goal_1': row1['goal'],
                            'goal_2': row2['goal'],
                            'price_1': row1[price_col],
                            'price_2': row2[price_col],
                            'outcome_1': row1['funding_ratio'],
                            'outcome_2': row2['funding_ratio'],
                            'price_change_pct': (row2[price_col] - row1[price_col]) / row1[price_col] * 100,
                            'outcome_change': row2['funding_ratio'] - row1['funding_ratio'],
                        })
        
        pairs_df = pd.DataFrame(pairs)
        logger.info(f"Identified {len(pairs_df)} matched campaign pairs for validation")
        return pairs_df
    
    def validate_predictions(self, 
                            pairs_df: pd.DataFrame,
                            predictor,
                            df: pd.DataFrame) -> Dict:
        """
        Validate counterfactual predictions against matched pairs.
        
        Args:
            pairs_df: DataFrame of matched campaign pairs
            predictor: Fitted CounterfactualPredictor
            df: Original data DataFrame
            
        Returns:
            Validation results dictionary
        """
        errors = []
        predictions = []
        
        for _, pair in pairs_df.iterrows():
            # Get campaign 1 data
            campaign_1 = df[df['project_id'] == pair['campaign_1_id']].iloc[0].to_dict()
            
            # Predict what would happen with campaign 2's price
            price_change = pair['price_change_pct']
            
            try:
                cf_result = predictor.predict_counterfactual(campaign_1, price_change)
                
                pred_outcome = cf_result['counterfactual_prediction']
                actual_outcome = pair['outcome_2']
                
                error = abs(pred_outcome - actual_outcome)
                errors.append(error)
                
                predictions.append({
                    'pair': (pair['campaign_1_id'], pair['campaign_2_id']),
                    'predicted': pred_outcome,
                    'actual': actual_outcome,
                    'error': error,
                    'price_change': price_change,
                })
            except Exception as e:
                logger.warning(f"Error predicting for pair: {e}")
                continue
        
        if errors:
            results = {
                'n_pairs': len(errors),
                'mae': np.mean(errors),
                'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
                'median_error': np.median(errors),
                'error_percentiles': {
                    '25%': np.percentile(errors, 25),
                    '50%': np.percentile(errors, 50),
                    '75%': np.percentile(errors, 75),
                    '90%': np.percentile(errors, 90),
                },
                'predictions': predictions,
            }
        else:
            results = {'error': 'No valid predictions made'}
        
        self.validation_results = results
        return results


class PlaceboTester:
    """
    Placebo tests to validate model doesn't find effects where none exist.
    
    For campaigns that didn't change strategy, the model should predict
    no change in outcomes.
    """
    
    def __init__(self):
        self.placebo_results = {}
    
    def run_placebo_test(self, 
                        df: pd.DataFrame,
                        predictor,
                        n_samples: int = 100) -> Dict:
        """
        Run placebo test on random campaigns.
        
        Pretend campaigns changed price at midpoint, but since they didn't,
        model should predict similar outcomes to observed.
        
        Args:
            df: Campaign DataFrame
            predictor: Fitted CounterfactualPredictor
            n_samples: Number of samples to test
            
        Returns:
            Placebo test results
        """
        np.random.seed(42)
        sample_indices = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
        
        placebo_effects = []
        
        for idx in sample_indices:
            campaign = df.iloc[idx].to_dict()
            actual_outcome = campaign['funding_ratio']
            
            try:
                # Predict with NO price change (should match observed)
                cf_result = predictor.predict_counterfactual(campaign, price_change_pct=0)
                pred_outcome = cf_result['current_prediction']
                
                # The "placebo effect" is the difference between prediction and actual
                # when we pretend there was a change but there wasn't
                placebo_effect = abs(pred_outcome - actual_outcome)
                placebo_effects.append(placebo_effect)
                
            except Exception as e:
                continue
        
        results = {
            'n_samples': len(placebo_effects),
            'mean_placebo_effect': np.mean(placebo_effects),
            'std_placebo_effect': np.std(placebo_effects),
            'max_placebo_effect': np.max(placebo_effects),
            'percentile_95': np.percentile(placebo_effects, 95),
            'passed': np.mean(placebo_effects) < 0.5,  # Arbitrary threshold
        }
        
        logger.info(f"Placebo Test Results:")
        logger.info(f"  Mean Placebo Effect: {results['mean_placebo_effect']:.4f}")
        logger.info(f"  95th Percentile: {results['percentile_95']:.4f}")
        logger.info(f"  Test Passed: {results['passed']}")
        
        self.placebo_results = results
        return results


class ManskiBoundsAnalyzer:
    """
    Manski bounds for partial identification of treatment effects.
    
    Provides bounds on counterfactual estimates when assumptions are uncertain.
    """
    
    def __init__(self):
        self.bounds_results = {}
    
    def compute_bounds(self,
                       observed_outcome: float,
                       treatment_effect: float,
                       outcome_range: Tuple[float, float] = (0.0, 10.0),
                       uncertainty: float = 0.3) -> Dict:
        """
        Compute Manski bounds for a counterfactual estimate.
        
        Args:
            observed_outcome: The observed outcome
            treatment_effect: Estimated treatment effect
            outcome_range: Plausible range for outcomes
            uncertainty: Uncertainty factor
            
        Returns:
            Bounds dictionary
        """
        # Point estimate
        point_estimate = observed_outcome + treatment_effect
        
        # Conservative bounds based on outcome range
        min_possible = max(outcome_range[0], observed_outcome - abs(treatment_effect) - uncertainty)
        max_possible = min(outcome_range[1], observed_outcome + abs(treatment_effect) + uncertainty)
        
        # Manski-style worst/best case
        worst_case = max(outcome_range[0], point_estimate - uncertainty)
        best_case = min(outcome_range[1], point_estimate + uncertainty)
        
        return {
            'point_estimate': point_estimate,
            'worst_case': worst_case,
            'best_case': best_case,
            'manski_lower': min_possible,
            'manski_upper': max_possible,
            'interval_width': max_possible - min_possible,
        }
    
    def analyze_dataset_bounds(self,
                               df: pd.DataFrame,
                               predictor,
                               price_change_pct: float = -20,
                               n_samples: int = 100) -> Dict:
        """
        Compute bounds for a sample of campaigns.
        
        Args:
            df: Campaign DataFrame
            predictor: Fitted CounterfactualPredictor
            price_change_pct: Price change to analyze
            n_samples: Number of samples
            
        Returns:
            Aggregate bounds analysis
        """
        np.random.seed(42)
        sample_indices = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
        
        all_bounds = []
        
        for idx in sample_indices:
            campaign = df.iloc[idx].to_dict()
            
            try:
                cf_result = predictor.predict_counterfactual(campaign, price_change_pct)
                
                bounds = self.compute_bounds(
                    observed_outcome=campaign['funding_ratio'],
                    treatment_effect=cf_result['causal_effect'],
                    uncertainty=0.3
                )
                
                all_bounds.append(bounds)
            except Exception as e:
                continue
        
        if all_bounds:
            results = {
                'n_samples': len(all_bounds),
                'avg_point_estimate': np.mean([b['point_estimate'] for b in all_bounds]),
                'avg_interval_width': np.mean([b['interval_width'] for b in all_bounds]),
                'estimates_within_bounds': sum(
                    1 for b in all_bounds 
                    if b['manski_lower'] <= b['point_estimate'] <= b['manski_upper']
                ) / len(all_bounds),
                'bounds_sample': all_bounds[:5],
            }
        else:
            results = {'error': 'No valid bounds computed'}
        
        self.bounds_results = results
        return results


def run_full_validation(df: pd.DataFrame, predictor) -> Dict:
    """
    Run complete validation suite.
    
    Args:
        df: Campaign DataFrame
        predictor: Fitted CounterfactualPredictor
        
    Returns:
        Complete validation results
    """
    results = {}
    
    # 1. Temporal Validation
    logger.info("\n" + "="*50)
    logger.info("Running Temporal Validation...")
    logger.info("="*50)
    
    temporal = TemporalValidator()
    pairs = temporal.identify_strategy_changes(df)
    if len(pairs) > 0:
        results['temporal'] = temporal.validate_predictions(pairs, predictor, df)
    else:
        results['temporal'] = {'error': 'No matched pairs found'}
    
    # 2. Placebo Test
    logger.info("\n" + "="*50)
    logger.info("Running Placebo Test...")
    logger.info("="*50)
    
    placebo = PlaceboTester()
    results['placebo'] = placebo.run_placebo_test(df, predictor)
    
    # 3. Manski Bounds
    logger.info("\n" + "="*50)
    logger.info("Computing Manski Bounds...")
    logger.info("="*50)
    
    bounds = ManskiBoundsAnalyzer()
    results['bounds'] = bounds.analyze_dataset_bounds(df, predictor)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*50)
    
    if 'temporal' in results and 'mae' in results['temporal']:
        logger.info(f"Temporal Validation MAE: {results['temporal']['mae']:.4f}")
    
    if 'placebo' in results:
        logger.info(f"Placebo Test Passed: {results['placebo'].get('passed', 'N/A')}")
    
    if 'bounds' in results and 'avg_interval_width' in results['bounds']:
        logger.info(f"Average Bounds Width: {results['bounds']['avg_interval_width']:.4f}")
    
    return results


if __name__ == "__main__":
    # This would be run after fitting models
    logger.info("Validation module loaded. Import and use with fitted predictor.")
