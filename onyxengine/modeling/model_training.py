from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import Union, List, Literal, Optional
from onyxengine.modeling import (
    OnyxModelConfig,
    validate_param,
    MLPOptConfig,
    RNNOptConfig,
    TransformerOptConfig,
)
from onyxengine.modeling.optimizers import (
    AdamWConfig,
    AdamWOptConfig,
    SGDConfig,
    SGDOptConfig,
)
from onyxengine.modeling.lr_schedulers import (
    CosineDecayWithWarmupConfig,
    CosineDecayWithWarmupOptConfig,
    CosineAnnealingWarmRestartsConfig,
    CosineAnnealingWarmRestartsOptConfig,
)

class TrainingConfig(BaseModel):
    """
    Configuration for the training of a model.
    
    Args:
        training_iters (int): Number of training iterations (default is 3000, max is 100000).
        train_batch_size (int): Batch size for training (default is 32).
        train_val_split_ratio (float): Ratio of training data to validation data (default is 0.9).
        test_dataset_size (int): Number of samples in the test dataset (default is 500).
        checkpoint_type (Literal['single_step', 'multi_step']): Type of checkpointing (default is 'single_step').
        optimizer (Union[AdamWConfig, SGDConfig]): Optimizer configuration (default is AdamWConfig()).
        lr_scheduler (Union[None, CosineDecayWithWarmupConfig, CosineAnnealingWarmRestartsConfig]): Learning rate scheduler configuration (default is None).
    """
    type: Literal['training_config'] = Field(default='training_config', frozen=True, init=False)
    training_iters: int = 3000
    train_batch_size: int = 32
    train_val_split_ratio: float = 0.9
    test_dataset_size: int = 500
    checkpoint_type: Literal['single_step', 'multi_step'] = 'single_step'
    optimizer: Union[AdamWConfig, SGDConfig] = AdamWConfig()
    lr_scheduler: Union[None, CosineDecayWithWarmupConfig, CosineAnnealingWarmRestartsConfig] = None
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.training_iters, 'training_iters', min_val=1, max_val=100000)
        validate_param(self.train_batch_size, 'train_batch_size', min_val=1)
        validate_param(self.train_val_split_ratio, 'train_val_split_ratio', min_val=0.0, max_val=1.0)
        validate_param(self.test_dataset_size, 'test_dataset_size', min_val=1)
        return self

class OptimizationConfig(BaseModel):
    """
    Configuration for the optimization of models.
    
    Args:
        training_iters (int): Number of training iterations (default is 3000, max is 100000).
        train_batch_size (int): Batch size for training (default is 32).
        train_val_split_ratio (float): Ratio of training data to validation data (default is 0.9).
        test_dataset_size (int): Number of samples in the test dataset (default is 500).
        checkpoint_type (Literal['single_step', 'multi_step']): Type of checkpointing (default is 'single_step').
        opt_models (List[Union[MLPOptConfig, RNNOptConfig, TransformerOptConfig]]): List of model optimization configurations.
        opt_optimizers (List[Union[AdamWOptConfig, SGDOptConfig]]): List of optimizer optimization configurations.
        opt_lr_schedulers (List[Union[None, CosineDecayWithWarmupOptConfig, CosineAnnealingWarmRestartsOptConfig]]): List of learning rate scheduler optimization configurations.
        num_trials (int): Number of optimization trials (default is 10).
    """
    type: Literal['optimization_config'] = Field(default='optimization_config', frozen=True, init=False)
    training_iters: int = 3000
    train_batch_size: int = 32
    train_val_split_ratio: float = 0.9
    test_dataset_size: int = 500
    checkpoint_type: Literal['single_step', 'multi_step'] = 'single_step'
    opt_models: List[Union[MLPOptConfig, RNNOptConfig, TransformerOptConfig]] = []
    opt_optimizers: List[Union[AdamWOptConfig, SGDOptConfig]] = []
    opt_lr_schedulers: List[Union[None, CosineDecayWithWarmupOptConfig, CosineAnnealingWarmRestartsOptConfig]] = [None]
    num_trials: int = 10
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.training_iters, 'training_iters', min_val=1, max_val=100000)
        validate_param(self.train_batch_size, 'train_batch_size', min_val=1)
        validate_param(self.train_val_split_ratio, 'train_val_split_ratio', min_val=0.0, max_val=1.0)
        validate_param(self.test_dataset_size, 'test_dataset_size', min_val=1)
        validate_param(self.num_trials, 'num_trials', min_val=1)
        
        # Length of opt_models, opt_optimizers, and opt_lr_schedulers must be at least 1
        if len(self.opt_models) < 1:
            raise ValueError("Optimization config must have at least one model.")
        if len(self.opt_optimizers) < 1:
            raise ValueError("Optimization config must have at least one optimizer.")
        if len(self.opt_lr_schedulers) < 1:
            raise ValueError("Optimization config must have at least one learning rate scheduler.")
        
        # If none of model inputs are states, then optimization config cannot be multi_step
        for model in self.opt_models:
            if not any(input.relation is not None for input in model.inputs):
                if self.checkpoint_type == 'multi_step':
                    raise ValueError("Optimization config cannot be multi_step if none of the model inputs are states.")
                break
            
        return self
        
class TrainingJob(BaseModel):
    """
    Configuration for an Onyx model training job.
    
    Args:
        onyx_model_name (str): Name of the model to train.
        onyx_model_config (OnyxModelConfig): Configuration for the model to train.
        dataset_name (str): Name of the dataset to train on.
        dataset_id (Optional[str]): ID of the dataset to train on. (Default is None)
        training_config (TrainingConfig): Configuration for the training process.
    """
    type: Literal['training_job'] = Field(default='training_job', frozen=True, init=False)
    onyx_model_name: str
    onyx_model_config: OnyxModelConfig
    dataset_name: str
    dataset_id: Optional[str] = None
    training_config: TrainingConfig
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        # If none of model inputs are states, then training config cannot be multi_step
        if not any(input.relation is not None for input in self.onyx_model_config.inputs):
            if self.training_config.checkpoint_type == 'multi_step':
                raise ValueError("Training config cannot be multi_step if none of the model inputs are states.")
        return self
    
class OptimizationJob(BaseModel):
    """
    Configuration for an Onyx model optimization job.
    
    Args:
        onyx_model_name (str): Name of the model to optimize.
        dataset_name (str): Name of the dataset to optimize on.
        dataset_id (Optional[str]): ID of the dataset to optimize on. (Default is None)
        optimization_config (OptimizationConfig): Configuration for the optimization process.
    """
    type: Literal['optimization_job'] = Field(default='optimization_job', frozen=True, init=False)
    onyx_model_name: str
    dataset_name: str
    dataset_id: Optional[str] = None
    optimization_config: OptimizationConfig