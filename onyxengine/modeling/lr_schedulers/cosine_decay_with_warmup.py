from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import Union, Dict, List, Literal
from onyxengine.modeling import validate_param, validate_opt_param

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

