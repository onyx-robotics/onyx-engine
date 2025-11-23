from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import Union, Dict, List, Literal
from onyxengine.modeling import validate_param, validate_opt_param

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

