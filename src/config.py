# src/config.py

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH  = "data/processed/test.csv"

MODEL_PATH = "models/model.pkl"
METRICS_PATH = "reports/metrics.json"

# Final tuned hyperparameters
MODEL_PARAMS = {
    "class_weight": "balanced",
    "num_leaves": 15,
    "max_depth": 12,
    "min_child_samples": 100,
    "random_state": 42
}

THRESHOLD = 0.61