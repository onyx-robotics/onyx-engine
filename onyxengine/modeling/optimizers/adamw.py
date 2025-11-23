from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import Union, Dict, List, Literal
from onyxengine.modeling import validate_param, validate_opt_param

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

