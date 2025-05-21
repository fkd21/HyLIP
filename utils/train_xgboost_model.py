#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import argparse
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train XGBoost model for latency prediction')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with predictions')
    parser.add_argument('--output_dir', type=str, default='trained_models', help='Directory to save trained model')
    parser.add_argument('--model_name', type=str, default='latency_xgboost_model.json', help='Model filename')
    parser.add_argument('--target', type=str, default='measured_latency', 
                        choices=['measured_latency', 'Predicted_Latency'], 
                        help='Target column to predict')
    parser.add_argument('--ratio', type=str, default='correction_ratio',
                        help='Create ratio column: measured/predicted (override with target/Predicted_Latency)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    return parser.parse_args()

def preprocess_data(df, target_col, ratio_col=None):
    """Preprocess data for XGBoost model"""
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Check if we need to create a ratio column
    if ratio_col and 'Predicted_Latency' in df.columns and target_col != 'Predicted_Latency':
        # Create ratio of actual to predicted latency
        df[ratio_col] = df[target_col] / df['Predicted_Latency']
        logger.info(f"Created ratio column '{ratio_col}' = {target_col} / Predicted_Latency")
        
        # Handle potential infinity or NaN values
        df[ratio_col] = df[ratio_col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median to avoid dropping rows
        median_ratio = df[ratio_col].median()
        df[ratio_col] = df[ratio_col].fillna(median_ratio)
        
        # Log ratio statistics
        logger.info(f"Ratio statistics: min={df[ratio_col].min():.4f}, max={df[ratio_col].max():.4f}, "
                    f"mean={df[ratio_col].mean():.4f}, median={median_ratio:.4f}")
    
    # Convert categorical features to numeric
    categorical_cols = ['hardware', 'framework', 'model_name']
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
            logger.info(f"Encoded {col} with {len(le.classes_)} unique values")
    
    # Select features for model training
    numeric_cols = ['seq_len', 'batch_size', 'num_devices']
    encoded_cols = [f'{col}_encoded' for col in categorical_cols if col in df.columns]
    
    # If we have predicted latency as a feature but not the target, include it
    if 'Predicted_Latency' in df.columns and target_col != 'Predicted_Latency':
        numeric_cols.append('Predicted_Latency')
    
    feature_cols = numeric_cols + encoded_cols
    
    # Handle missing values
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            logger.warning(f"Column {col} has {df[col].isna().sum()} missing values. Filling with median.")
            df[col] = df[col].fillna(df[col].median())
    
    # Scale numeric features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index
    )
    
    # Combine scaled numeric features with encoded categorical features
    for col in encoded_cols:
        scaled_features[col] = df[col]
    
    return scaled_features, df[target_col if ratio_col is None else ratio_col], encoders, scaler, feature_cols

def train_model(X_train, y_train, X_val, y_val, verbose=False):
    """Train XGBoost model with validation"""
    logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
    
    # Create parameter grid
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 1 if verbose else 0
    }
    
    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Define evaluation sets
    evals = [(dtrain, 'train'), (dval, 'validation')]
    
    # Train the model with early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100 if verbose else 0
    )
    
    logger.info(f"Best iteration: {model.best_iteration}")
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)
    
    logger.info("Feature importance:")
    for i, row in importance_df.iterrows():
        logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
    
    return model, importance_df

def evaluate_model(model, X, y, dataset_name="Test"):
    """Evaluate XGBoost model on a dataset"""
    # Convert to DMatrix for prediction
    dmatrix = xgb.DMatrix(X)
    
    # Make predictions
    predictions = model.predict(dmatrix)
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    # Log metrics
    logger.info(f"{dataset_name} set evaluation:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    
    return predictions, {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def save_model_and_metadata(model, output_dir, model_name, encoders, scaler, feature_cols, 
                           metrics, importance_df, target_col, ratio_col=None):
    """Save model, preprocessing objects, and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save XGBoost model
    model_path = os.path.join(output_dir, model_name)
    model.save_model(model_path)
    logger.info(f"Saved XGBoost model to {model_path}")
    
    # Save encoders and scaler
    preprocessing_path = os.path.join(output_dir, f"{os.path.splitext(model_name)[0]}_preprocessing.joblib")
    joblib.dump({
        'encoders': encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'ratio_col': ratio_col
    }, preprocessing_path)
    logger.info(f"Saved preprocessing objects to {preprocessing_path}")
    
    # Save metrics and feature importance
    metadata_path = os.path.join(output_dir, f"{os.path.splitext(model_name)[0]}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'feature_importance': importance_df.to_dict(orient='records'),
            'target_col': target_col,
            'ratio_col': ratio_col,
            'features': feature_cols
        }, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{os.path.splitext(model_name)[0]}_importance.png")
    plt.savefig(plot_path)
    logger.info(f"Saved feature importance plot to {plot_path}")

def save_datasets(train_df, val_df, test_df, output_dir):
    """Save the train, validation, and test datasets to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_dataset.csv')
    val_path = os.path.join(output_dir, 'validation_dataset.csv')
    test_path = os.path.join(output_dir, 'test_dataset.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved train dataset to {train_path} ({len(train_df)} rows)")
    logger.info(f"Saved validation dataset to {val_path} ({len(val_df)} rows)")
    logger.info(f"Saved test dataset to {test_path} ({len(test_df)} rows)")

def main():
    args = parse_arguments()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return 1
    
    # Check if target column exists
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in dataset")
        return 1
    
    # Create train/validation/test split (80%/10%/10%)
    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    # Second split: 10% validation, 10% test (from the 20% temp)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)
    
    logger.info(f"Split data into train ({len(train_df)} rows), validation ({len(val_df)} rows), and test ({len(test_df)} rows)")
    
    # Save the raw datasets
    save_datasets(train_df, val_df, test_df, args.output_dir)
    
    # Preprocess data
    X_train, y_train, encoders, scaler, feature_cols = preprocess_data(train_df, args.target, args.ratio)
    
    # Preprocess validation and test sets with the same transformations
    X_val, y_val = preprocess_data(val_df, args.target, args.ratio)[0:2]
    X_test, y_test = preprocess_data(test_df, args.target, args.ratio)[0:2]
    
    # Train model
    model, importance_df = train_model(X_train, y_train, X_val, y_val, args.verbose)
    
    # Evaluate on all sets
    _, train_metrics = evaluate_model(model, X_train, y_train, "Training")
    _, val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_preds, test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Combine metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    # Save model, preprocessing objects, and metadata
    save_model_and_metadata(
        model, args.output_dir, args.model_name, 
        encoders, scaler, feature_cols, 
        all_metrics, importance_df,
        args.target, args.ratio
    )
    
    logger.info("Training and evaluation completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 