from pydantic import BaseModel
from typing import Literal

class TrainingConfig(BaseModel):
    """
    Configuration for the training of a model.
    
    Args:
        training_iters (int): Number of training iterations (default is 3000).
        train_batch_size (int): Batch size for training (default is 32).
        train_val_split_ratio (float): Ratio of training data to validation data (default is 0.9).
        test_dataset_size (int): Number of samples in the test dataset (default is 500).
    """
    training_iters: int = 3000
    train_batch_size: int = 32
    train_val_split_ratio: float = 0.9
    test_dataset_size: int = 500
    checkpoint_type: Literal['single_step', 'multi_step'] = 'single_step'