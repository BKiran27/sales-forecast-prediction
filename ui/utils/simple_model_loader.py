"""
Simplified and Resilient Model Loader for Streamlit UI
Supports both Local Artifacts (Streamlit Cloud / Standalone) and MLflow tracking.
"""

import os
import pickle
import joblib
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Try importing mlflow; allow running without it
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)


class SimpleModelLoader:
    """Model loader supporting pre-packaged local models with MLflow fallback"""
    
    def __init__(self):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(self.mlflow_uri)
            except Exception:
                pass
        
        self.models = {}
        self.scalers = None
        self.encoders = None
        self.feature_cols = None
        self.loaded = False
        
        # Automatically attempt to load local bundled models on initialization
        self.load_local_models()
        
    def load_local_models(self) -> bool:
        """Load models from local models/ directory for Streamlit Cloud and offline execution"""
        try:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            ui_dir = os.path.dirname(curr_dir)
            root_dir = os.path.dirname(ui_dir)
            
            candidate_dirs = [
                os.path.join(root_dir, "models"),
                os.path.join(ui_dir, "models"),
                os.path.join(os.getcwd(), "models"),
                "models"
            ]
            
            models_dir = None
            for c_dir in candidate_dirs:
                if os.path.exists(c_dir):
                    models_dir = c_dir
                    break
                    
            if not models_dir:
                logger.info("No local models/ directory found.")
                return False
                
            logger.info(f"Loading local models from: {models_dir}")
            
            # 1. Load Scalers
            scalers_path = os.path.join(models_dir, "scalers.pkl")
            if os.path.exists(scalers_path):
                try:
                    self.scalers = joblib.load(scalers_path)
                except Exception:
                    with open(scalers_path, 'rb') as f:
                        self.scalers = pickle.load(f)
                logger.info("Loaded scalers")

            # 2. Load Encoders
            encoders_path = os.path.join(models_dir, "encoders.pkl")
            if os.path.exists(encoders_path):
                try:
                    self.encoders = joblib.load(encoders_path)
                except Exception:
                    with open(encoders_path, 'rb') as f:
                        self.encoders = pickle.load(f)
                logger.info("Loaded encoders")

            # 3. Load Feature Columns
            feat_path = os.path.join(models_dir, "feature_cols.pkl")
            if os.path.exists(feat_path):
                try:
                    self.feature_cols = joblib.load(feat_path)
                except Exception:
                    with open(feat_path, 'rb') as f:
                        self.feature_cols = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_cols)} feature columns")

            # 4. Load XGBoost
            for p in [os.path.join(models_dir, "xgboost", "xgboost_model.pkl"), os.path.join(models_dir, "xgboost_model.pkl")]:
                if os.path.exists(p):
                    try:
                        self.models['xgboost'] = joblib.load(p)
                    except Exception:
                        with open(p, 'rb') as f:
                            self.models['xgboost'] = pickle.load(f)
                    logger.info("Loaded XGBoost model")
                    break

            # 5. Load LightGBM
            for p in [os.path.join(models_dir, "lightgbm", "lightgbm_model.pkl"), os.path.join(models_dir, "lightgbm_model.pkl")]:
                if os.path.exists(p):
                    try:
                        self.models['lightgbm'] = joblib.load(p)
                    except Exception:
                        with open(p, 'rb') as f:
                            self.models['lightgbm'] = pickle.load(f)
                    logger.info("Loaded LightGBM model")
                    break

            # 6. Load Ensemble
            for p in [os.path.join(models_dir, "ensemble", "ensemble_model.pkl"), os.path.join(models_dir, "ensemble_model.pkl")]:
                if os.path.exists(p):
                    try:
                        self.models['ensemble'] = joblib.load(p)
                        logger.info("Loaded Ensemble model")
                    except Exception as e:
                        logger.warning(f"Could not load saved ensemble model directly: {e}")
                    break

            # Recreate Ensemble if individual models exist
            if 'ensemble' not in self.models and ('xgboost' in self.models or 'lightgbm' in self.models):
                from .ensemble_model_standalone import EnsembleModel
                ens_models = {}
                weights = {}
                if 'xgboost' in self.models:
                    ens_models['xgboost'] = self.models['xgboost']
                    weights['xgboost'] = 0.5
                if 'lightgbm' in self.models:
                    ens_models['lightgbm'] = self.models['lightgbm']
                    weights['lightgbm'] = 0.5
                if len(ens_models) == 1:
                    k = list(ens_models.keys())[0]
                    weights[k] = 1.0
                self.models['ensemble'] = EnsembleModel(ens_models, weights)
                logger.info("Created fallback ensemble model from loaded base models")

            self.loaded = len(self.models) > 0
            return self.loaded
        except Exception as e:
            logger.error(f"Error loading local models: {e}")
            return False

    def load_models_from_run(self, run_id: str) -> bool:
        """Load models from a specific MLflow run, with local fallback"""
        if not HAS_MLFLOW:
            logger.info("MLflow not installed, loading local models")
            return self.load_local_models()
            
        try:
            logger.info(f"Attempting to load models from MLflow run: {run_id}")
            client = MlflowClient()
            local_dir = f"/tmp/mlflow_models/{run_id}"
            os.makedirs(local_dir, exist_ok=True)
            
            artifacts_path = client.download_artifacts(run_id, "", dst_path=local_dir)
            logger.info(f"Downloaded artifacts to: {artifacts_path}")
            
            # Load scalers
            scalers_path = os.path.join(artifacts_path, "scalers.pkl")
            if os.path.exists(scalers_path):
                self.scalers = joblib.load(scalers_path)
            
            # Load encoders
            encoders_path = os.path.join(artifacts_path, "encoders.pkl")
            if os.path.exists(encoders_path):
                self.encoders = joblib.load(encoders_path)
            
            # Load feature columns
            feature_cols_path = os.path.join(artifacts_path, "feature_cols.pkl")
            if os.path.exists(feature_cols_path):
                self.feature_cols = joblib.load(feature_cols_path)
            
            # Load models
            models_dir = os.path.join(artifacts_path, "models")
            if os.path.exists(models_dir):
                xgb_path = os.path.join(models_dir, "xgboost", "xgboost_model.pkl")
                if os.path.exists(xgb_path):
                    self.models['xgboost'] = joblib.load(xgb_path)
                
                lgb_path = os.path.join(models_dir, "lightgbm", "lightgbm_model.pkl")
                if os.path.exists(lgb_path):
                    self.models['lightgbm'] = joblib.load(lgb_path)
                
                ens_path = os.path.join(models_dir, "ensemble", "ensemble_model.pkl")
                if os.path.exists(ens_path):
                    try:
                        self.models['ensemble'] = joblib.load(ens_path)
                    except Exception:
                        from .ensemble_model_standalone import EnsembleModel
                        self.models['ensemble'] = EnsembleModel(
                            {'xgboost': self.models.get('xgboost'), 'lightgbm': self.models.get('lightgbm')},
                            {'xgboost': 0.5, 'lightgbm': 0.5}
                        )
            
            self.loaded = len(self.models) > 0
            if self.loaded:
                return True
        except Exception as e:
            logger.warning(f"Could not load models from MLflow: {e}. Falling back to local models.")
            
        return self.load_local_models()
    
    def get_latest_run(self) -> Optional[str]:
        """Get the latest successful run ID from MLflow if available"""
        if not HAS_MLFLOW:
            return None
        try:
            exp = mlflow.get_experiment_by_name("sales_forecasting")
            if not exp:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                return runs.iloc[0]['run_id']
            return None
        except Exception as e:
            logger.info(f"Could not reach MLflow for latest run: {e}")
            return None
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions by averaging available models"""
        predictions = []
        if 'xgboost' in self.models:
            predictions.append(self.models['xgboost'].predict(X))
        if 'lightgbm' in self.models:
            predictions.append(self.models['lightgbm'].predict(X))
        
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            raise ValueError("No models available for prediction")
    
    def predict(self, X: np.ndarray, model_type: str = 'ensemble') -> np.ndarray:
        """Make predictions with specified model"""
        if model_type == 'ensemble':
            if 'ensemble' in self.models:
                return self.models['ensemble'].predict(X)
            else:
                return self.predict_ensemble(X)
        elif model_type in self.models:
            return self.models[model_type].predict(X)
        elif 'ensemble' in self.models:
            return self.models['ensemble'].predict(X)
        elif len(self.models) > 0:
            first_model = list(self.models.values())[0]
            return first_model.predict(X)
        else:
            raise ValueError(f"Model type '{model_type}' not available and no fallback models loaded")