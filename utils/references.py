from typing import Final

# Metric keys
METRIC_LOSS: Final[str] = "loss"
METRIC_ACCURACY: Final[str] = "accuracy"
METRIC_F1: Final[str] = "f1"
METRIC_PRECISION: Final[str] = "precision"
METRIC_RECALL: Final[str] = "recall"

# Checkpoint keys
CKPT_MODEL_NAME: Final[str] = "model_name"
CKPT_STATE_DICT: Final[str] = "state_dict"
CKPT_BEST_VAL_F1: Final[str] = "best_val_f1"
CKPT_BEST_VAL_LOSS: Final[str] = "best_val_loss"
CKPT_BEST_VAL_ACCURACY: Final[str] = "best_val_accuracy"
CKPT_BEST_VAL_PRECISION: Final[str] = "best_val_precision"
CKPT_BEST_VAL_RECALL: Final[str] = "best_val_recall"
CKPT_BEST_EPOCH: Final[str] = "best_epoch"
CKPT_TOTAL_EPOCHS: Final[str] = "total_epochs"
CKPT_NUM_EPOCHS_LEGACY: Final[str] = "num_epochs"
CKPT_BEST_THRESHOLD: Final[str] = "best_threshold"
CKPT_THRESHOLD: Final[str] = "th_multi_label"
CKPT_BATCH_SIZE: Final[str] = "batch_size"
CKPT_LEARNING_RATE: Final[str] = "learning_rate"
CKPT_LEARNING_RATES: Final[str] = "learning_rates"
CKPT_RUN_DURATION_SECONDS: Final[str] = "run_duration_seconds"
