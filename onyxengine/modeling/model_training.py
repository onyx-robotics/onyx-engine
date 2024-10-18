from pydantic import BaseModel

class TrainingConfig(BaseModel):
    training_iters: int = 3000
    train_batch_size: int = 32
    train_val_split_ratio: float = 0.9
    test_dataset_size: int = 500