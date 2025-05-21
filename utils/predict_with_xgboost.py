#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict using trained XGBoost model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--model', type=str, required=True, help='XGBoost model file path')
    parser.add_argument('--preprocessing', type=str, required=True, help='Preprocessing data file path')
    parser.add_argument('--output_col', type=str, default='XGBoost_Predicted_Latency', 
                        help='Name of output column for predictions')
    return parser.parse_args()

def load_model_and_preprocessing(model_path: str, preprocessing_path: str) -> Tuple[Any, Dict[str, Any]]:
    """Load XGBoost model and preprocessing objects"""
    try:
        # Load XGBoost model
        model = xgb.Booster()
        model.load_model(model_path)
        logger.info(f"Loaded XGBoost model from {model_path}")
        
        # Load preprocessing objects
        preprocessing = joblib.load(preprocessing_path)
        logger.info(f"Loaded preprocessing objects from {preprocessing_path}")
        
        return model, preprocessing
    except Exception as e:
        logger.error(f"Error loading model or preprocessing: {e}")
        raise

def preprocess_input_data(df: pd.DataFrame, preprocessing: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data using saved preprocessing objects"""
    df = df.copy()
    
    # Extract preprocessing objects
    encoders = preprocessing['encoders']
    scaler = preprocessing['scaler']
    feature_cols = preprocessing['feature_cols']
    
    # Create copy of numerical features for scaling
    numeric_cols = [col for col in feature_cols if not col.endswith('_encoded')]
    
    # Categorical encoding
    for col, encoder in encoders.items():
        # Check if column exists in the data
        if col in df.columns:
            # Check for unknown categories
            unknown_categories = set(df[col].unique()) - set(encoder.classes_)
            if unknown_categories:
                logger.warning(f"Found {len(unknown_categories)} unknown categories in {col}. Replacing with most common class.")
                # Get most common class
                most_common_class = encoder.classes_[0]  # Fallback to first class if needed
                # Replace unknown categories
                for cat in unknown_categories:
                    df.loc[df[col] == cat, col] = most_common_class
            
            # Encode column
            df[f'{col}_encoded'] = encoder.transform(df[col])
        else:
            logger.error(f"Column {col} not found in input data")
            raise ValueError(f"Column {col} not found in input data")
    
    # Handle missing values in numeric features
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            logger.warning(f"Column {col} has {df[col].isna().sum()} missing values. Filling with median.")
            df[col] = df[col].fillna(df[col].median())
    
    # Scale numeric features
    if len(numeric_cols) > 0:
        numeric_data = df[numeric_cols].copy()
        scaled_features = pd.DataFrame(
            scaler.transform(numeric_data),
            columns=numeric_cols,
            index=df.index
        )
    else:
        scaled_features = pd.DataFrame(index=df.index)
    
    # Add encoded categorical features
    encoded_cols = [col for col in feature_cols if col.endswith('_encoded')]
    for col in encoded_cols:
        if col in df.columns:
            scaled_features[col] = df[col]
        else:
            logger.error(f"Column {col} not found in preprocessed data")
            raise ValueError(f"Column {col} not found in preprocessed data")
    
    # Ensure all feature columns exist
    missing_features = set(feature_cols) - set(scaled_features.columns)
    if missing_features:
        logger.error(f"Missing features in preprocessed data: {missing_features}")
        raise ValueError(f"Missing features in preprocessed data: {missing_features}")
    
    # Return preprocessed data with features in the correct order
    return scaled_features[feature_cols]

def predict_with_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Make predictions with the XGBoost model"""
    dmatrix = xgb.DMatrix(X)
    predictions = model.predict(dmatrix)
    return predictions

def apply_predictions(df: pd.DataFrame, predictions: np.ndarray, preprocessing: Dict[str, Any], 
                     output_col: str) -> pd.DataFrame:
    """Apply predictions to the dataframe based on model type (direct or ratio)"""
    df = df.copy()
    
    # Check if we're predicting a ratio or direct value
    ratio_col = preprocessing.get('ratio_col')
    
    if ratio_col:
        # If we predicted a ratio, multiply by the base model predictions
        if 'Predicted_Latency' in df.columns:
            logger.info(f"Applying ratio predictions to Predicted_Latency column")
            df[output_col] = predictions * df['Predicted_Latency']
        else:
            logger.warning(f"Predicted_Latency column not found. Cannot apply ratio.")
            df[output_col] = predictions
    else:
        # Direct prediction
        df[output_col] = predictions
    
    return df

def main():
    args = parse_arguments()
    
    try:
        # Load model and preprocessing objects
        model, preprocessing = load_model_and_preprocessing(args.model, args.preprocessing)
        
        # Load input data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows from {args.input}")
        
        # Preprocess input data
        X = preprocess_input_data(df, preprocessing)
        logger.info(f"Preprocessed input data, shape: {X.shape}")
        
        # Make predictions
        predictions = predict_with_model(model, X)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Apply predictions to dataframe
        result_df = apply_predictions(df, predictions, preprocessing, args.output_col)
        
        # Save results
        result_df.to_csv(args.output, index=False)
        logger.info(f"Saved predictions to {args.output}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 