from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

class BaseFeature(BaseModel):
    type: Literal['base_feature'] = Field(default='base_feature', frozen=True, init=False)
    name: str
    scale: Literal['mean'] | List[float] = 'mean'
    train_mean: Optional[float] = Field(default=None, init=False)
    train_std: Optional[float] = Field(default=None, init=False)
    train_min: Optional[float] = Field(default=None, init=False)
    train_max: Optional[float] = Field(default=None, init=False)
    
    @model_validator(mode='after')
    def validate_scale(self) -> Self:
        if isinstance(self.scale, list):
            if len(self.scale) != 2:
                raise ValueError("Scale list must have 2 values representing the range of real-world values for this feature as: [min, max]")
            if self.scale[0] >= self.scale[1]:
                raise ValueError("Scale must be in the form [min, max] where min < max")
            
        return self

class Output(BaseFeature):
    """
    A standard output feature to be used by the model.
    
    Args:
        name (str): Name of the output feature.
        scale (Literal['mean'] | List[float]): Scale for the output feature:
            
            - 'mean': Feature is normalized to the mean of the training set. (Default).
            - List[float]: Feature is normalized to a specified range of real-world values
    """
    
    type: Literal['output'] = Field(default='output', frozen=True, init=False)
    
class Input(BaseFeature):
    """
    A standard input feature to be used by the model.
    
    Args:
        name (str): Name of the input feature.
        scale (Literal['mean'] | List[float]): Scale for the output feature:
            
            - 'mean': Feature is normalized to the mean of the training set. (Default).
            - List[float]: Feature is normalized to a specified range of real-world values    """
    
    type: Literal['input'] = Field(default='input', frozen=True, init=False)
    
class State(BaseFeature):
    """
    A state feature that can be derived from a parent feature through different relationships (output, delta, or derivative).

    Args:
        name (str): Name of the state feature.
        relation (Literal['output', 'delta', 'derivative']): Method to solve for the feature:
        
            - 'output': Feature is the direct output of the model
            - 'delta': Feature is the change/delta of the parent value
            - 'derivative': Feature is the derivative of the parent value
        parent (str): Name of the parent feature from which this state is derived
        scale (Literal['mean'] | List[float]): Scale for the output feature:
            
            - 'mean': Feature is normalized to the mean of the training set. (Default).
            - List[float]: Feature is normalized to a specified range of real-world values
    """
    type: Literal['state'] = Field(default='state', frozen=True, init=False)
    relation: Literal['output', 'delta', 'derivative'] # Method to solve for the feature: the output of the model, parent is the delta of the value, or derivative of parent value
    parent: str # Parent feature to derive from
    
class Feature(BaseModel):
    config: Union[Input, Output, State] = Field(..., discriminator='type')