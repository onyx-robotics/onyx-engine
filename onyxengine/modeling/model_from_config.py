from onyxengine.modeling.models import *
import json

def model_from_config_json(config_json):
    config_dict = json.loads(config_json)
    model_type = config_dict['model_type']
    # Remove model_type from config_dict
    config_dict.pop('model_type')
    
    if model_type == 'mlp':
        mlp_config = MLPConfig(**config_dict)
        model = MLP(mlp_config)
    elif model_type == 'rnn':
        rnn_config = RNNConfig(**config_dict)
        model = RNN(rnn_config)
    elif model_type == 'transformer':
        transformer_config = GPTConfig(**config_dict)
        model = GPT(transformer_config)
    else:
        raise ValueError(f"Could not find model type {model_type}")
    
    return model