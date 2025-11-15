import json
from typing import Union
from onyxengine.modeling import OnyxModelConfigClass, MLP, RNN, Transformer

def model_from_config(model_config: Union[str, dict]):
    config_dict = json.loads(model_config) if isinstance(model_config, str) else model_config
    config = OnyxModelConfigClass(config=config_dict).config
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