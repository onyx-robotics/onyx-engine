import pandas as pd
from pydantic import BaseModel

class OnyxDatasetConfig(BaseModel):
    num_outputs: int
    num_state: int
    num_control: int
    dt: float

class OnyxDataset:
    """
    Onyx dataset class for storing dataframe and metadata for the dataset.
    
    Args:
        dataframe (pd.DataFrame): Dataframe containing the dataset.
        num_outputs (int): Number of output features.
        num_state (int): Number of state variables.
        num_control (int): Number of control variables.
        dt (float): Delta time between samples in seconds.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame = None,
        num_outputs: int = None,
        num_state: int = None,
        num_control: int = None,
        dt: float = None,
    ):
        self.dataframe = dataframe
        self.num_outputs = num_outputs
        self.num_state = num_state
        self.num_control = num_control
        self.num_inputs = self.num_state + self.num_control if self.num_state and self.num_control else None
        self.dt = dt

    @property
    def config(self):
        return OnyxDatasetConfig(
            num_outputs=self.num_outputs,
            num_state=self.num_state,
            num_control=self.num_control,
            dt=self.dt,
        )
        
    def from_config(self, config: OnyxDatasetConfig):
        self.num_outputs = config.num_outputs
        self.num_state = config.num_state
        self.num_control = config.num_control
        self.num_inputs = self.num_state + self.num_control
        self.dt = config.dt