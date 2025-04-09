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
    type: Literal['output'] = Field(default='output', frozen=True, init=False)
    
class Input(BaseFeature):
    type: Literal['input'] = Field(default='input', frozen=True, init=False)
    
class State(BaseFeature):
    type: Literal['state'] = Field(default='state', frozen=True, init=False)
    relation: Literal['output', 'delta', 'derivative'] # Method to solve for the variable: the output of the model, parent is the delta of the value, or derivative of parent value
    parent: str # Parent variable to derive from
    
class Feature(BaseModel):
    config: Union[Input, Output, State] = Field(..., discriminator='type')