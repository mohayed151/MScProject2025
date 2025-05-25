"""
Main execution script for the intraday trading strategy
This script orchestrates the entire pipeline: data download, feature engineering, model training, and backtesting
"""
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Import our modules from the 'Scripts' subdirectory
from Scripts.data_preprocessing import DataDownloader, main as data_preprocessing_main
from Scripts.feature_engineering import FeatureEngineer, main as feature_engineering_main
from Scripts.model_training import ModelTrainer, main as model_training_main
from Scripts.trading_strategy import TradingStrategy # TradingStrategy no longer has a 'main' function
from Scripts.backtesting import Backtester # Backtester no longer has a 'main' function
from config import * # Import all configs, including RESULTS_DIR, INTERVAL etc.

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'trading_strategy.log'), # Log to file
        logging.StreamHandler(sys.stdout) # Also log to console
    ]
)
logger = logging.getLogger(__name__)

class TradingPipeline:
    def __init__(self):
        """
        Initialize the trading pipeline components.
        Use lazy initialization to avoid loading data/models that don't exist yet.
        """
        # Initialize components that don't require data upfront
        self.downloader = DataDownloader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        # These will be initialized lazily when needed
        self._backtester = None
        self._trading_strategy = None
        
    @property
    def backtester(self):
        """Lazy initialization of backtester."""
        if self._backtester is None:
            logger.info("Initializing Backtester...")
            self._backtester = Backtester()
        return self._backtester
    
    @property
    def trading_strategy(self):
        """Lazy initialization of trading strategy."""
        if self._trading_strategy is None:
            logger.info("Initializing TradingStrategy...")
            self._trading_strategy = TradingStrategy()
        return self._trading_strategy
        
    def run_full_pipeline(self, force_download=False, force_features=False, 
                          force_training=False, run_backtest=True):
        """
        Run the complete trading strategy pipeline.
        
        Args:
            force_download (bool): Force re-download of data.
            force_features (bool): Force re-creation of features.
            force_training (bool): Force re-training of models.
            run_backtest (bool): Whether to run backtesting.
        """
        logger.info("="*80)
        logger.info("STARTING TRADING STRATEGY PIPELINE")
        logger.info("="*80)
        
        try:
            # Step 1: Data Download
            logger.info("\n--- Step 1: Data Download ---")
            self._run_data_download(force_download)
            
            # Step 2: Feature Engineering
            logger.info("\n--- Step 2: Feature Engineering ---")
            self._run_feature_engineering(force_features)
            
            # Step 3: Model Training
            logger.info("\n--- Step 3: Model Training ---")
            self._run_model_training(force_training)
            
            # Step 4: Backtesting (includes strategy validation implicitly)
            if run_backtest:
                logger.info("\n--- Step 4: Backtesting ---")
                self._run_backtesting()
            else:
                logger.info("\n--- Step 4: Backtesting skipped as requested. ---")
                
            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True) # Log full traceback
            raise # Re-raise the exception to indicate failure to the caller
            
    def _run_data_download(self, force_download=False):
        """
        Run data download step.
        """
        existing_data = self.downloader.load_existing_data()
        
        if existing_data is not None and not force_download:
            logger.info(f"Found existing data with shape: {existing_data.shape}")
            logger.info(f"Date range: {existing_data.index[0]} to {existing_data.index[-1]}")
            
            # Check if data is recent (e.g., within the last day for intraday data)
            if existing_data.index[-1] < pd.Timestamp.now() - pd.Timedelta(days=1):
                logger.info(f"Data is older than 1 day. Updating...")
                data = self.downloader.update_data()
            else:
                logger.info("Using existing data (recent).")
                data = existing_data
        else:
            logger.info("Downloading fresh data...")
            data = self.downloader.download_all_data()
            
        if data is None:
            raise RuntimeError("Failed to obtain price data during download step.")
            
        logger.info(f"Data download completed. Shape: {data.shape}")
        
    def _run_feature_engineering(self, force_features=False):
        """
        Run feature engineering step.
        """
        # Check if combined features file exists
        combined_features_file = FEATURES_DATA_DIR / f"combined_features_{INTERVAL}_ml_strategy.csv"
        
        if combined_features_file.exists() and not force_features:
            logger.info("Found existing combined features file. Loading into FeatureEngineer's cache...")
            # Load raw data first for FeatureEngineer to process
            self.feature_engineer.load_data() 
            # Then call create_all_features with save_features=False to populate internal cache
            self.feature_engineer.create_all_features(save_features=False) 
            logger.info(f"Loaded features for {len(self.feature_engineer.features_data)} tickers from existing files.")
        else:
            logger.info("Creating features...")
            # Ensure raw data is loaded before creating features
            self.feature_engineer.load_data() 
            features_data = self.feature_engineer.create_all_features() # This will save features to files
            
            if not features_data:
                raise RuntimeError("Failed to create features during feature engineering step.")
                
            logger.info(f"Feature engineering completed for {len(features_data)} tickers.")
            
    def _run_model_training(self, force_training=False):
        """
        Run model training step.
        """
        # Check if models already exist (check for any model file)
        model_files = list(MODELS_DIR.glob("*_model.joblib"))
        
        if model_files and not force_training:
            logger.info(f"Found {len(model_files)} existing model files. Loading...")
            self.model_trainer.load_models() # This loads models, scalers, and performance metrics
            logger.info(f"Loaded {len(self.model_trainer.models)} models.")
        else:
            logger.info("Training models...")
            # The model_trainer.train_all_models will call self.model_trainer.load_features() internally
            model_types = ['ridge', 'rf'] # Train both Ridge and Random Forest
            results = self.model_trainer.train_all_models(model_types=model_types)
            
            if not results:
                raise RuntimeError("Failed to train models during model training step.")
                
            logger.info(f"Model training completed. Trained {len(results)} models.")
            
        # Always display model performance after training or loading
        self.model_trainer.evaluate_model_performance()
            
    def _run_backtesting(self):
        """
        Run backtesting.
        """
        logger.info("Running backtesting...")
        
        try:
            # Now we can safely initialize the backtester since data and models should exist
            # The property will handle lazy initialization
            self.backtester.run_backtest()
            
            # Display results and plots from the Backtester instance
            self.backtester.display_performance_summary()
            self.backtester.plot_results() # This method also saves plots if SAVE_PLOTS is True
            
            logger.info("Backtesting completed successfully.")
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}", exc_info=True)
            raise
            
    def run_individual_step(self, step):
        """
        Run individual pipeline step.
        
        Args:
            step (str): Step name ('download', 'features', 'training', 'backtest')
        """
        if step == 'download':
            data_preprocessing_main()
        elif step == 'features':
            feature_engineering_main()
        elif step == 'training':
            model_training_main()
        elif step == 'backtest':
            self._run_backtesting() # Call the pipeline's internal backtesting method
        else:
            raise ValueError(f"Unknown step: {step}. Available steps: 'download', 'features', 'training', 'backtest'.")
            
        logger.info(f"Individual step '{step}' completed.")
            
    def get_pipeline_status(self):
        """
        Check the status of each pipeline component by checking file existence.
        
        Returns:
            dict: Status of each component.
        """
        status = {}
        
        # Check data
        data_file = RAW_DATA_DIR / f"combined_ohlcv_prices_{INTERVAL}.csv"
        status['data'] = {
            'exists': data_file.exists(),
            'file': str(data_file),
            'size_mb': data_file.stat().st_size / (1024 * 1024) if data_file.exists() else 0
        }
        
        # Check features
        feature_file = FEATURES_DATA_DIR / f"combined_features_{INTERVAL}_ml_strategy.csv"
        status['features'] = {
            'exists': feature_file.exists(),
            'count': 1 if feature_file.exists() else 0,
            'file': str(feature_file) if feature_file.exists() else "N/A"
        }
        
        # Check models
        model_files = list(MODELS_DIR.glob("*_model.joblib"))
        status['models'] = {
            'exists': len(model_files) > 0,
            'count': len(model_files),
            'files': [str(f) for f in model_files[:5]]
        }
        
        # Check results
        results_data_files = list((RESULTS_DIR / 'data').glob("*.csv"))
        results_plot_files = list((RESULTS_DIR / 'plots').glob("*.png"))
        status['results'] = {
            'exists': len(results_data_files) > 0 or len(results_plot_files) > 0,
            'data_files_count': len(results_data_files),
            'plot_files_count': len(results_plot_files),
            'sample_data_files': [str(f) for f in results_data_files[:3]],
            'sample_plot_files': [str(f) for f in results_plot_files[:3]]
        }
        
        return status
        
    def print_status(self):
        """
        Print pipeline status.
        """
        status = self.get_pipeline_status()
        
        print("\n" + "="*60)
        print("TRADING STRATEGY PIPELINE STATUS")
        print("="*60)
        
        for component, info in status.items():
            print(f"\n{component.upper()}:")
            print(f"  Exists: {info['exists']}")
            
            if component == 'data':
                print(f"  File: {info['file']}")
                print(f"  Size: {info['size_mb']:.1f} MB")
            elif component == 'features':
                print(f"  File: {info['file']}")
                print(f"  Count: {info['count']}")
            elif component == 'models':
                print(f"  Count: {info['count']}")
                if info['files']:
                    print(f"  Sample files: {info['files'][:3]}")
            elif component == 'results':
                print(f"  Data files count: {info['data_files_count']}")
                print(f"  Plot files count: {info['plot_files_count']}")
                if info['sample_data_files']:
                    print(f"  Sample data files: {info['sample_data_files']}")
                if info['sample_plot_files']:
                    print(f"  Sample plot files: {info['sample_plot_files']}")

def create_argument_parser():
    """
    Create command line argument parser.
    """
    parser = argparse.ArgumentParser(description='Intraday Trading Strategy Pipeline')
    
    parser.add_argument('--step', type=str, choices=['download', 'features', 'training', 'backtest', 'full'],
                        default='full', help='Pipeline step to run')
    
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download of data')
    
    parser.add_argument('--force-features', action='store_true',
                        help='Force re-creation of features')
    
    parser.add_argument('--force-training', action='store_true',
                        help='Force re-training of models')
    
    parser.add_argument('--no-backtest', action='store_true',
                        help='Skip backtesting step')
    
    parser.add_argument('--status', action='store_true',
                        help='Show pipeline status and exit')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (DEBUG level)')
    
    return parser

def main():
    """
    Main function to parse arguments and run the trading pipeline.
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = TradingPipeline()
    
    # Handle status request
    if args.status:
        pipeline.print_status()
        return
    
    try:
        if args.step == 'full':
            # Run full pipeline
            pipeline.run_full_pipeline(
                force_download=args.force_download,
                force_features=args.force_features,
                force_training=args.force_training,
                run_backtest=not args.no_backtest
            )
        else:
            # Run individual step
            if args.step == 'download':
                pipeline._run_data_download(force_download=args.force_download)
            elif args.step == 'features':
                pipeline._run_feature_engineering(force_features=args.force_features)
            elif args.step == 'training':
                pipeline._run_model_training(force_training=args.force_training)
            elif args.step == 'backtest':
                pipeline._run_backtesting()
            else:
                raise ValueError(f"Invalid step '{args.step}' for individual execution.")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()