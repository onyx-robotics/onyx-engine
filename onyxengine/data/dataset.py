import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class OnyxDatasetConfig(BaseModel):
    type: Literal['dataset'] = Field(default='dataset', frozen=True, init=False)
    features: List[str] = []
    dt: float = 0

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
        features: Optional[List[str]] = [],
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
                dt=dt
            )
            self.dataframe = dataframe
            self.validate_dataframe()
            
    def validate_dataframe(self):
        features = self.config.features
        # Make sure number of features matches number of columns
        assert len(features) == len(
            self.dataframe.columns
        ), "Number of features does not match number of columns in dataframe."
        # Ensure column names match features
        self.dataframe.columns = features
