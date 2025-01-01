import json
from typing import Union
from onyxengine.modeling.models import *

def model_from_config(model_config: Union[str, dict]):
    config_dict = json.loads(model_config) if isinstance(model_config, str) else model_config
    model_type = config_dict['onyx_model_type']
    config_dict.pop('onyx_model_type')

    if model_type == 'mlp':
        mlp_config = MLPConfig.model_validate(config_dict)
        model = MLP(mlp_config)
    elif model_type == 'rnn':
        rnn_config = RNNConfig.model_validate(config_dict)
        model = RNN(rnn_config)
    elif model_type == 'transformer':
        transformer_config = TransformerConfig.model_validate(config_dict)
        model = Transformer(transformer_config)
    else:
        raise ValueError(f"Could not find model type {model_type}")

    return model

def opt_model_config_from_config(opt_model_config: Union[str, dict]):
    config_dict = json.loads(opt_model_config) if isinstance(opt_model_config, str) else opt_model_config
    opt_model_type = config_dict['onyx_model_type']
    config_dict.pop('onyx_model_type')
    
    if opt_model_type == 'mlp_opt':
        opt_model = MLPOptConfig.model_validate(config_dict)
    elif opt_model_type == 'rnn_opt':
        opt_model = RNNOptConfig.model_validate(config_dict)
    elif opt_model_type == 'transformer_opt':
        opt_model = TransformerOptConfig.model_validate(config_dict)
    else:
        raise ValueError(f"Could not find optimization model type {opt_model_type}")
    
    return opt_model