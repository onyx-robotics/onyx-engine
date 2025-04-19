from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import List, Union, Dict, Literal
from onyxengine.modeling import Input, Output, State, validate_param, validate_opt_param

def validate_inputs_and_outputs(inputs: List[Union[Input, State]], outputs: List[Output]) -> None:
    # Check that there is at least one input and one output
    if not inputs:
        raise ValueError("At least one input is required.")
    if not outputs:
        raise ValueError("At least one output is required.")
    
    # Check that all outputs are of type Output
    for output in outputs:
        if not isinstance(output, Output):
            raise ValueError(f"Output {output} is not of type Output.")
        
    # Check that all inputs are of type Input or State
    for input in inputs:
        if not isinstance(input, (Input, State)):
            raise ValueError(f"Input {input} is not of type Input or State.")
    
    # Check that all input and output names are unique
    all_names = [input.name for input in inputs] + [output.name for output in outputs]
    if len(set(all_names)) != len(all_names):
        raise ValueError("All input and output names must be unique.")

class OnyxModelBaseConfig(BaseModel):
    type: Literal['onyx_model'] = Field(default='onyx_model', frozen=True, init=False) # Override in child classes
    outputs: List[Output]
    inputs: List[Union[Input, State]]
    dt: float = 0.0
    sequence_length: int = 1
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_inputs_and_outputs(self.inputs, self.outputs)
        validate_param(self.dt, 'dt', min_val=0.0)
        validate_param(self.sequence_length, 'sequence_length', min_val=1, max_val=50)
        return self
    
    @property
    def num_outputs(self) -> int:
        return len(self.outputs)
    @property
    def num_states(self) -> int:
        return len([x for x in self.inputs if isinstance(x, State)])
    @property
    def num_inputs(self) -> int:
        return len([x for x in self.inputs if isinstance(x, Input)])
    
class OnyxModelOptBaseConfig(BaseModel):
    type: Literal['onyx_model_opt'] = Field(default='onyx_model_opt', frozen=True, init=False) # Override in child classes
    outputs: List[Output]
    inputs: List[Union[Input, State]]
    dt: Union[float, Dict[str, List[float]]] = {"range": [0.001, 0.1, 0.005]}
    sequence_length: Union[int, Dict[str, List[int]]] = {"range": [1, 10, 1]}
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_inputs_and_outputs(self.inputs, self.outputs)
        validate_opt_param(self.dt, 'dt', options=['select', 'range'], min_val=0.0001, max_val=10.0)
        validate_opt_param(self.sequence_length, 'sequence_length', options=['select', 'range'], min_val=1, max_val=50)
        return self