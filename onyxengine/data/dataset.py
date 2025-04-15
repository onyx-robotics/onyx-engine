import pandas as pd
from pydantic import BaseModel, model_validator, Field
from typing_extensions import Self
from typing import List, Optional, Literal

class OnyxDatasetConfig(BaseModel):
    type: Literal['dataset'] = Field(default='dataset', frozen=True, init=False)
    outputs: List[str] = []
    inputs: List[str] = []
    dt: float = 0

    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        # Check that there's at least one output
        assert len(self.outputs) > 0, "At least one output feature must be defined."
        # Check that there's at least one input
        assert len(self.inputs) > 0, "At least one input feature must be defined."
        # Check that dt is greater than 0
        assert self.dt > 0, "dt must be greater than 0."
        # Check that the number of features matches the sum of num_outputs, num_state, and num_control
        return self

class OnyxDataset:
    """
    Onyx dataset class for storing dataframe and metadata for the dataset. Can be initialized with a configuration object or by parameter.
    
    Args:
        dataframe (pd.DataFrame): Dataframe containing the dataset.
        outputs (List[str]): List of output feature names.
        inputs (List[str]): List of input feature names.
        dt (float): Time step of the dataset.
        config (OnyxDatasetConfig): Configuration object for the dataset. (Optional if other parameters are provided)
    """
    def __init__(
        self,
        dataframe: pd.DataFrame = pd.DataFrame(),
        outputs: Optional[List[str]] = [],
        inputs: Optional[List[str]] = [],
        dt: float = 0,
        config: OnyxDatasetConfig = None
    ):
        if config is not None:
            self.config = config
            self.dataframe = dataframe
            self.validate_dataframe()
        else:
            self.config = OnyxDatasetConfig(
                outputs=outputs,
                inputs=inputs,
                dt=dt
            )
            self.dataframe = dataframe
            self.validate_dataframe()
            
    def validate_dataframe(self):
        features = self.config.outputs + self.config.inputs
        # Make sure number of features matches number of columns
        assert len(features) == len(
            self.dataframe.columns
        ), "Number of outputs and inputs does not match number of columns in dataframe."
        # Ensure column names match features
        self.dataframe.columns = features
