from onyxengine.modeling.models import *
import json

def model_from_config_json(config_json):
    config_dict = json.loads(config_json)
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