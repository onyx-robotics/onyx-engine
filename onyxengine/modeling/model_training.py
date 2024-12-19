from pydantic import BaseModel
from typing import Literal, List

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
    
class OptimizationConfig(TrainingConfig):
    """
    Configuration for the optimization of a model. Inherits from TrainingConfig to specify each trial's training configuration.
    
    Args:
        optimize_model_types (List[str]): List of model types to optimize across (default is ['mlp', 'rnn', 'transformer']).
        optimize_sequence_length (bool): Whether to optimize the sequence length (default is True).
        num_trials (int): Number of trials to run in optimization (default is 10).
    """
    
    optimize_model_types: List[str] = ['mlp', 'rnn', 'transformer']
    optimize_sequence_length: bool = False
    num_trials: int = 10