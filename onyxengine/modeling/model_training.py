from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import Union, Dict, List, Literal
from onyxengine.modeling import (
    validate_param,
    validate_opt_param,
    MLPOptConfig,
    RNNOptConfig,
    TransformerOptConfig,
)

class AdamWConfig(BaseModel):
    """
    Configuration for the AdamW optimizer.
    
    Args:
        lr (float): Learning rate (default is 3e-4).
        weight_decay (float): Weight decay (default is 1e-2).
    """
    type: Literal['adamw'] = Field(default='adamw', frozen=True, init=False)
    lr: float = 3e-4
    weight_decay: float = 1e-2
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.lr, 'lr', min_val=0.0)
        validate_param(self.weight_decay, 'weight_decay', min_val=0.0)
        return self

class AdamWOptConfig(BaseModel):
    """
    Optimization config for the AdamW optimizer.
    
    Args:
        lr (Union[float, Dict[str, List[float]]): Learning rate (default is {"select": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 5e-3, 1e-2]}).
        weight_decay (Union[float, Dict[str, List[float]]): Weight decay (default is {"select": [1e-4, 1e-3, 1e-2, 1e-1]}).
    """
    type: Literal['adamw_opt'] = Field(default='adamw_opt', frozen=True, init=False)
    lr: Union[float, Dict[str, List[float]]] = {"select": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 5e-3, 1e-2]}
    weight_decay: Union[float, Dict[str, List[float]]] = {"select": [1e-4, 1e-3, 1e-2, 1e-1]}
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_opt_param(self.lr, 'lr', options=['select', 'range'], min_val=0.0)
        validate_opt_param(self.weight_decay, 'weight_decay', options=['select', 'range'], min_val=0.0)
        return self

class SGDConfig(BaseModel):
    """
    Configuration for the SGD optimizer.
    
    Args:
        lr (float): Learning rate (default is 3e-4).
        weight_decay (float): Weight decay (default is 1e-2).
        momentum (float): Momentum (default is 0.9).
    """
    type: Literal['sgd'] = Field(default='sgd', frozen=True, init=False)
    lr: float = 3e-4
    weight_decay: float = 1e-2
    momentum: float = 0.9
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.lr, 'lr', min_val=0.0)
        validate_param(self.weight_decay, 'weight_decay', min_val=0.0)
        validate_param(self.momentum, 'momentum', min_val=0.0, max_val=1.0)
        return self

class SGDOptConfig(BaseModel):
    """
    Optimization config for the SGD optimizer.
    
    Args:
        lr (Union[float, Dict[str, List[float]]): Learning rate (default is {"select": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 5e-3, 1e-2]}).
        weight_decay (Union[float, Dict[str, List[float]]): Weight decay (default is {"select": [1e-4, 1e-3, 1e-2, 1e-1]}).
        momentum (Union[float, Dict[str, List[float]]): Momentum (default is {"select": [0.0, 0.8, 0.9, 0.95, 0.99]}).
    """
    type: Literal['sgd_opt'] = Field(default='sgd_opt', frozen=True, init=False)
    lr: Union[float, Dict[str, List[float]]] = {"select": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 5e-3, 1e-2]}
    weight_decay: Union[float, Dict[str, List[float]]] = {"select": [1e-4, 1e-3, 1e-2, 1e-1]}
    momentum: Union[float, Dict[str, List[float]]] = {"select": [0.0, 0.8, 0.9, 0.95, 0.99]}
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_opt_param(self.lr, 'lr', options=['select', 'range'], min_val=0.0)
        validate_opt_param(self.weight_decay, 'weight_decay', options=['select', 'range'], min_val=0.0)
        validate_opt_param(self.momentum, 'momentum', options=['select', 'range'], min_val=0.0, max_val=1.0)
        return self

class OptimizerConfig(BaseModel):
    config: Union[AdamWConfig, SGDConfig] = Field(..., discriminator='type')

class OptimizerOptConfig(BaseModel):
    config: Union[AdamWOptConfig, SGDOptConfig] = Field(..., discriminator='type')

class CosineDecayWithWarmupConfig(BaseModel):
    """
    Configuration for learning rate scheduler with cosine decay and linear warmup.
    
    Args:
        max_lr (float): Maximum learning rate (default is 3e-4).
        min_lr (float): Minimum learning rate (default is 3e-5).
        warmup_iters (int): Number of warmup iterations (default is 200).
        decay_iters (int): Number of decay iterations (default is 1000).
    """
    type: Literal['cosine_decay_with_warmup'] = Field(default='cosine_decay_with_warmup', frozen=True, init=False)
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    decay_iters: int = 1000
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.max_lr, 'max_lr', min_val=0.0)
        validate_param(self.min_lr, 'min_lr', min_val=0.0)
        validate_param(self.warmup_iters, 'warmup_iters', min_val=0)
        validate_param(self.decay_iters, 'decay_iters', min_val=0)
        return self

class CosineDecayWithWarmupOptConfig(BaseModel):
    """
    Optimization config for learning rate scheduler with cosine decay and linear warmup.
    
    Args:
        max_lr (Union[float, Dict[str, List[float]]): Maximum learning rate (default is {"select": [1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 3e-3, 5e-3]}).
        min_lr (Union[float, Dict[str, List[float]]): Minimum learning rate (default is {"select": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 8e-5, 1e-4]}).
        warmup_iters (Union[int, Dict[str, List[int]]): Number of warmup iterations (default is {"select": [50, 100, 200, 400, 800]}).
        decay_iters (Union[int, Dict[str, List[int]]): Number of decay iterations (default is {"select": [500, 1000, 2000, 4000, 8000]}).
    
    """
    type: Literal['cosine_decay_with_warmup_opt'] = Field(default='cosine_decay_with_warmup_opt', frozen=True, init=False)
    max_lr: Union[float, Dict[str, List[float]]] = {"select": [1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 3e-3, 5e-3]}
    min_lr: Union[float, Dict[str, List[float]]] = {"select": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 8e-5, 1e-4]}
    warmup_iters: Union[int, Dict[str, List[int]]] = {"select": [50, 100, 200, 400, 800]}
    decay_iters: Union[int, Dict[str, List[int]]] = {"select": [500, 1000, 2000, 4000, 8000]}
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_opt_param(self.max_lr, 'max_lr', options=['select', 'range'], min_val=0.0)
        validate_opt_param(self.min_lr, 'min_lr', options=['select', 'range'], min_val=0.0)
        validate_opt_param(self.warmup_iters, 'warmup_iters', options=['select', 'range'], min_val=0)
        validate_opt_param(self.decay_iters, 'decay_iters', options=['select', 'range'], min_val=0)
        return self

class CosineAnnealingWarmRestartsConfig(BaseModel):
    """
    Configuration for learning rate scheduler with cosine annealing and warm restarts.
    
    Args:
        T_0 (int): Initial period of learning rate decay (default is 2000).
        T_mult (int): Multiplicative factor for the period of learning rate decay (default is 1).
        eta_min (float): Minimum learning rate (default is 3e-5).
    """
    type: Literal['cosine_annealing_warm_restarts'] = Field(default='cosine_annealing_warm_restarts', frozen=True, init=False)
    T_0: int = 2000
    T_mult: int = 1
    eta_min: float = 3e-5
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.T_0, 'T_0', min_val=0)
        validate_param(self.T_mult, 'T_mult', min_val=0)
        validate_param(self.eta_min, 'eta_min', min_val=0.0)
        return self

class CosineAnnealingWarmRestartsOptConfig(BaseModel):
    """
    Optimization config for learning rate scheduler with cosine annealing and warm restarts.
    
    Args:
        T_0 (Union[int, Dict[str, List[int]]]): Initial period of learning rate decay (default is {"select": [200, 500, 1000, 2000, 5000, 10000]}).
        T_mult (Union[int, Dict[str, List[int]]]): Multiplicative factor for the period of learning rate decay (default is {"select": [1, 2, 3]}).
        eta_min (Union[float, Dict[str, List[float]]]): Minimum learning rate (default is {"select": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 3e-4]}).
    """
    type: Literal['cosine_annealing_warm_restarts_opt'] = Field(default='cosine_annealing_warm_restarts_opt', frozen=True, init=False)
    T_0: Union[int, Dict[str, List[int]]] = {"select": [200, 500, 1000, 2000, 5000, 10000]}
    T_mult: Union[int, Dict[str, List[int]]] = {"select": [1, 2, 3]}
    eta_min: Union[float, Dict[str, List[float]]] = {"select": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 3e-4]}
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_opt_param(self.T_0, 'T_0', options=['select', 'range'], min_val=0)
        validate_opt_param(self.T_mult, 'T_mult', options=['select', 'range'], min_val=0)
        validate_opt_param(self.eta_min, 'eta_min', options=['select', 'range'], min_val=0.0)
        return self

class SchedulerConfig(BaseModel):
    config: Union[CosineDecayWithWarmupConfig, CosineAnnealingWarmRestartsConfig] = Field(..., discriminator='type')

class SchedulerOptConfig(BaseModel):
    config: Union[CosineDecayWithWarmupOptConfig, CosineAnnealingWarmRestartsOptConfig] = Field(..., discriminator='type')

class TrainingConfig(BaseModel):
    """
    Configuration for the training of a model.
    
    Args:
        training_iters (int): Number of training iterations (default is 3000).
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

class OptimizationConfig(BaseModel):
    """
    Configuration for the optimization of models.
    
    Args:
        training_iters (int): Number of training iterations (default is 3000).
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