import pandas as pd
from pydantic import BaseModel, model_validator
from typing_extensions import Self
from typing import List, Optional

class OnyxDatasetConfig(BaseModel):
    features: List[str] = []
    num_outputs: int = 0
    num_state: int = 0
    num_control: int = 0
    dt: float = 0

    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        # Check that there's at least one output
        assert self.num_outputs > 0, "num_outputs must be greater than 0."
        # Check that there's at least one input
        assert self.num_state + self.num_control > 0, "At least one state or control variable must be defined."
        # Check that dt is greater than 0
        assert self.dt > 0, "dt must be greater than 0."
        # Check that the number of features matches the sum of num_outputs, num_state, and num_control
        assert (
            len(self.features) == self.num_outputs + self.num_state + self.num_control
        ), "Number of features does not match sum of num_outputs, num_state, and num_control."
        return self

class OnyxDataset:
    """
    Onyx dataset class for storing dataframe and metadata for the dataset. Can be initialized with a configuration object or by parameter.
    
    Args:
        features (List[str]): List of feature names.
        dataframe (pd.DataFrame): Dataframe containing the dataset.
        num_outputs (int): Number of output variables.
        num_state (int): Number of state variables.
        num_control (int): Number of control variables.
        dt (float): Time step of the dataset.
        config (OnyxDatasetConfig): Configuration object for the dataset. (Optional if other parameters are provided)
    """
    def __init__(
        self,
        features: Optional[List[str]] = [],
        dataframe: pd.DataFrame = pd.DataFrame(),
        num_outputs: int = 0,
        num_state: int = 0,
        num_control: int = 0,
        dt: float = 0,
        config: OnyxDatasetConfig = None
    ):
        if config is not None:
            self.config = config
            self.dataframe = dataframe
            self.validate_dataframe()
        else:
            self.config = OnyxDatasetConfig(
                features=features,
                num_outputs=num_outputs,
                num_state=num_state,
                num_control=num_control,
                dt=dt
            )
            self.dataframe = dataframe
            self.validate_dataframe()
            
    def validate_dataframe(self):
        # Make sure number of features matches number of columns
        assert len(self.config.features) == len(
            self.dataframe.columns
        ), "Number of features does not match number of columns in dataframe."
        # Ensure column names match features
        self.dataframe.columns = self.config.features