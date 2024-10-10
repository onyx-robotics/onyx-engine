from typing import Literal
from dataclasses import dataclass

@dataclass
class SimVariable:
    name: str # Variable name
    solver: Literal['next', 'delta', 'derivative'] # Method to derive the variable: the next value of parent, add a delta of parent value, or derivative of parent value
    parent_type: Literal['model_output', 'state_x'] # Parent variable to derive from
    parent_ind: int = 0 # Index of parent variable list [model_outputs] or [x]