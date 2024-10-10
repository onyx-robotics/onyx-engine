from dataclasses import dataclass

@dataclass
class TrainingConfig:
    training_iters: int = 3000
    train_test_split_ratio: float = 0.1
    test_dataset_size: int = 500