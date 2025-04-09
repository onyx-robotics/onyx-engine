import json
from typing import Union
from pydantic import BaseModel, Field
from onyxengine.modeling.models import *

class ModelConfig(BaseModel):
    config: Union[MLPConfig, RNNConfig, TransformerConfig] = Field(..., discriminator='type')
    
class ModelOptConfig(BaseModel):
    config: Union[MLPOptConfig, RNNOptConfig, TransformerOptConfig] = Field(..., discriminator='type')

def model_from_config(model_config: Union[str, dict]):
    config_dict = json.loads(model_config) if isinstance(model_config, str) else model_config
    config = ModelConfig(config=config_dict).config
    type = config.type
    
    if type == 'mlp':
        model = MLP(config)
    elif type == 'rnn':
        model = RNN(config)
    elif type == 'transformer':
        model = Transformer(config)
    else:
        raise ValueError(f"Could not find model type {type}")

    return model