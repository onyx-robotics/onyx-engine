from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import Union, Dict, List, Literal
from onyxengine.modeling import validate_param, validate_opt_param

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

