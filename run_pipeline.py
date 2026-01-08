#!/usr/bin/env python
"""
Pipeline Orchestrator
=====================
Run the complete Kickstarter counterfactual analysis pipeline with a single command.

Usage:
    python run_pipeline.py           # Run full pipeline
    python run_pipeline.py --step 1  # Run specific step
    python run_pipeline.py --from 3  # Run from step 3 onwards

Steps:
    1. Process Kaggle data
    2. Enrich data
    3. Create causal features
    4. Train models
    5. Run validation
"""

import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime


def setup_logging(config: dict) -> logging.Logger:
    """Configure logging for the pipeline."""
    log_config = config.get('logging', {})
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('pipeline')
    logger.info(f"Logging to: {log_file}")
    return logger


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        print(f"Warning: config.yaml not found at {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_step(step_num: int, step_name: str, module_path: str, logger: logging.Logger) -> bool:
    """
    Run a single pipeline step.
    
    Args:
        step_num: Step number for display
        step_name: Human-readable step name
        module_path: Path to Python module to run
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"STEP {step_num}: {step_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Import and run the module
        module_file = Path(module_path)
        
        if not module_file.exists():
            logger.error(f"Module not found: {module_path}")
            return False
        
        # Execute the module
        import subprocess
        result = subprocess.run(
            [sys.executable, str(module_file)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Log output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")
        
        if result.returncode != 0:
            logger.error(f"Step failed with exit code {result.returncode}")
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    logger.error(f"  {line}")
            return False
        
        elapsed = time.time() - start_time
        logger.info(f"[OK] Step {step_num} completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Step {step_num} failed: {e}")
        return False


def run_pipeline(start_step: int = 1, end_step: int = 5, only_step: int = None):
    """
    Run the complete pipeline or specific steps.
    
    Args:
        start_step: First step to run (1-5)
        end_step: Last step to run (1-5)
        only_step: If set, run only this step
    """
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("KICKSTARTER COUNTERFACTUAL ANALYSIS PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Define pipeline steps
    steps = [
        (1, "Process Kaggle Data", "src/process_kaggle_data.py"),
        (2, "Enrich Data", "src/data_enrichment.py"),
        (3, "Create Causal Features", "src/create_causal_features.py"),
        (4, "Train Models", "src/train_models.py"),
        (5, "Run Validation", "src/run_validation.py"),
    ]
    
    # Determine which steps to run
    if only_step:
        steps_to_run = [(n, name, path) for n, name, path in steps if n == only_step]
    else:
        steps_to_run = [(n, name, path) for n, name, path in steps if start_step <= n <= end_step]
    
    if not steps_to_run:
        logger.error("No valid steps to run")
        return False
    
    logger.info(f"Running steps: {[s[0] for s in steps_to_run]}")
    
    # Run each step
    results = []
    for step_num, step_name, module_path in steps_to_run:
        success = run_step(step_num, step_name, module_path, logger)
        results.append((step_num, step_name, success))
        
        if not success:
            logger.error(f"Pipeline stopped at step {step_num}")
            break
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    
    for step_num, step_name, success in results:
        status = "[OK] PASSED" if success else "[FAIL] FAILED"
        logger.info(f"  Step {step_num}: {step_name} - {status}")
    
    all_passed = all(r[2] for r in results)
    
    if all_passed:
        logger.info("")
        logger.info("[OK] PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  - Run dashboard: python -m streamlit run app.py")
        logger.info("  - View results: data/processed/validation_results.csv")
    else:
        logger.info("")
        logger.info("[FAIL] PIPELINE FAILED")
        logger.info("  Check logs above for error details")
    
    logger.info("=" * 60)
    
    return all_passed


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run Kickstarter counterfactual analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py              # Run full pipeline
  python run_pipeline.py --step 3     # Run only step 3
  python run_pipeline.py --from 2     # Run from step 2 onwards
  python run_pipeline.py --to 3       # Run steps 1-3 only
        """
    )
    
    parser.add_argument('--step', type=int, help='Run only this step (1-5)')
    parser.add_argument('--from', dest='from_step', type=int, default=1, 
                        help='Start from this step (default: 1)')
    parser.add_argument('--to', dest='to_step', type=int, default=5,
                        help='Run up to this step (default: 5)')
    
    args = parser.parse_args()
    
    success = run_pipeline(
        start_step=args.from_step,
        end_step=args.to_step,
        only_step=args.step
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
