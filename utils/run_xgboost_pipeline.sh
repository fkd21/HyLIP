#!/bin/bash

# XGBoost model training and prediction pipeline
set -e  # Exit on error

# Default paths
INPUT_FILE="data/all_results_with_predictions.csv"
OUTPUT_DIR="trained_models"
MODEL_NAME="latency_xgboost_model.json"
TARGET="measured_latency"
RATIO="correction_ratio"
VERBOSE=false
PREDICTION_COL="XGBoost_Predicted_Latency"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --ratio)
      RATIO="$2"
      shift 2
      ;;
    --prediction-col)
      PREDICTION_COL="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Input file not found: $INPUT_FILE"
  exit 1
fi

echo "======= XGBoost Pipeline ======="
echo "Input file: $INPUT_FILE"
echo "Target column: $TARGET"
echo "Ratio column: $RATIO"
echo "===============================\n"

# Build command for training
TRAIN_CMD="python utils/train_xgboost_model.py --input \"$INPUT_FILE\" --output_dir \"$OUTPUT_DIR\" --model_name \"$MODEL_NAME\" --target \"$TARGET\" --ratio \"$RATIO\""

if [ "$VERBOSE" = true ]; then
  TRAIN_CMD="$TRAIN_CMD --verbose"
fi

echo "Training XGBoost model..."
echo "> $TRAIN_CMD"
eval "$TRAIN_CMD"

# Check if training was successful
if [ $? -ne 0 ]; then
  echo "Error: Training failed!"
  exit 1
fi

echo "\n✓ Model training completed successfully!"

# Paths for predictions
MODEL_PATH="$OUTPUT_DIR/$MODEL_NAME"
PREPROCESSING_PATH="$OUTPUT_DIR/${MODEL_NAME%.*}_preprocessing.joblib"
TRAIN_DATASET="$OUTPUT_DIR/train_dataset.csv"
VAL_DATASET="$OUTPUT_DIR/validation_dataset.csv"
TEST_DATASET="$OUTPUT_DIR/test_dataset.csv"
TRAIN_PREDICTIONS="$OUTPUT_DIR/train_with_predictions.csv"
VAL_PREDICTIONS="$OUTPUT_DIR/validation_with_predictions.csv"
TEST_PREDICTIONS="$OUTPUT_DIR/test_with_predictions.csv"

# Run predictions on the three datasets
echo "\nGenerating predictions for train/validation/test splits..."

for DS in "train" "validation" "test"; do
  IN_FILE="$OUTPUT_DIR/${DS}_dataset.csv"
  OUT_FILE="$OUTPUT_DIR/${DS}_with_predictions.csv"
  
  PREDICT_CMD="python utils/predict_with_xgboost.py --input \"$IN_FILE\" --output \"$OUT_FILE\" --model \"$MODEL_PATH\" --preprocessing \"$PREPROCESSING_PATH\" --output_col \"$PREDICTION_COL\""
  
  echo "Processing $DS dataset..."
  echo "> $PREDICT_CMD"
  eval "$PREDICT_CMD"
  
  if [ $? -ne 0 ]; then
    echo "Error: Prediction failed for $DS dataset!"
    exit 1
  fi
  
  echo "✓ Predictions saved to $OUT_FILE"
done

echo "\n✓ All predictions completed successfully!"
echo "\nResults summary:"
echo "- Trained model: $MODEL_PATH"
echo "- Preprocessing data: $PREPROCESSING_PATH"
echo "- Train predictions: $TRAIN_PREDICTIONS"
echo "- Validation predictions: $VAL_PREDICTIONS"
echo "- Test predictions: $TEST_PREDICTIONS"
echo "\nDone! 🎉" 