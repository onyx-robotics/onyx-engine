import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from typing import Literal, List, Union, Dict
from onyxengine.modeling import ModelSimulatorConfig, ModelSimulator, validate_param, validate_opt_param
    
class RNNConfig(BaseModel):
    """
    Configuration class for the RNN model.
    
    Args:
        onyx_model_type (str): Model type = 'rnn', immutable.
        sim_config (ModelSimulatorConfig): Configuration for the model's simulator.
        num_inputs (int): Number of input features (default is 1).
        num_outputs (int): Number of output features (default is 1).
        sequence_length (int): Length of input sequences (default is 1).
        rnn_type (Literal['RNN', 'LSTM', 'GRU']): Type of RNN to use (default is 'LSTM').
        hidden_layers (int): Number of hidden layers in the RNN (default is 2).
        hidden_size (int): Number of hidden units in each layer (default is 32).
        dropout (float): Dropout rate (default is 0.0).
        bias (bool): Whether or not to include bias in the RNN (default is True).
    """
    onyx_model_type: str = Field(default='rnn', frozen=True, init=False)
    sim_config: ModelSimulatorConfig = ModelSimulatorConfig()
    num_inputs: int = 1
    num_outputs: int = 1
    rnn_type: Literal['RNN', 'LSTM', 'GRU'] = 'LSTM'
    sequence_length: int = 1
    hidden_layers: int = 2
    hidden_size: int = 32
    dropout: float = 0.0
    bias: bool = True
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.num_inputs, 'num_inputs', min_val=1)
        validate_param(self.num_outputs, 'num_outputs', min_val=1)
        validate_param(self.sequence_length, 'sequence_length', min_val=1, max_val=50)
        validate_param(self.hidden_layers, 'hidden_layers', min_val=1, max_val=10)
        validate_param(self.hidden_size, 'hidden_size', min_val=1, max_val=1024)
        validate_param(self.dropout, 'dropout', min_val=0.0, max_val=1.0)
        return self
    
class RNNOptConfig(BaseModel):
    """
    Optimization config for the RNN model.
    
    Args:
        onyx_model_type (str): Model type = 'rnn_opt', immutable.
        sim_config (ModelSimulatorConfig): Configuration for the model's simulator.
        num_inputs (int): Number of input features (default is 1).
        num_outputs (int): Number of output features (default is 1).
        rnn_type (Union[Literal['RNN', 'LSTM', 'GRU'], Dict[str, List[str]]): Type of RNN to use (default is {"select": ['RNN', 'LSTM', 'GRU']}).
        sequence_length (Union[int, Dict[str, List[int]]): Length of input sequences (default is {"select": [1, 2, 4, 5, 6, 8, 10, 12, 14, 15]}).
        hidden_layers (Union[int, Dict[str, List[int]]): Number of hidden layers in the RNN (default is {"range": [2, 5, 1]}).
        hidden_size (Union[int, Dict[str, List[int]]): Number of hidden units in each layer (default is {"select": [12, 24, 32, 64, 128]}).
        dropout (Union[float, Dict[str, List[float]]): Dropout rate (default is {"range": [0.0, 0.4, 0.1]}).
        bias (Union[bool, Dict[str, List[bool]]): Whether or not to include bias in the RNN (default is True).
    """
    onyx_model_type: str = Field(default='rnn_opt', frozen=True, init=False)
    sim_config: ModelSimulatorConfig = ModelSimulatorConfig()
    num_inputs: int = 1
    num_outputs: int = 1
    rnn_type: Union[Literal['RNN', 'LSTM', 'GRU'], Dict[str, List[str]]] = {"select": ['RNN', 'LSTM', 'GRU']}
    sequence_length: Union[int, Dict[str, List[int]]] = {"select": [1, 2, 4, 5, 6, 8, 10, 12, 14, 15]}
    hidden_layers: Union[int, Dict[str, List[int]]] = {"range": [2, 5, 1]}
    hidden_size: Union[int, Dict[str, List[int]]] = {"select": [12, 24, 32, 64, 128]}
    dropout: Union[float, Dict[str, List[float]]] = {"range": [0.0, 0.4, 0.1]}
    bias: Union[bool, Dict[str, List[bool]]] = True
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.num_inputs, 'num_inputs', min_val=1)
        validate_param(self.num_outputs, 'num_outputs', min_val=1)
        validate_opt_param(self.rnn_type, 'rnn_type', options=['select'], select_from=['RNN', 'LSTM', 'GRU'])
        validate_opt_param(self.sequence_length, 'sequence_length', options=['select', 'range'], min_val=1, max_val=50)
        validate_opt_param(self.hidden_layers, 'hidden_layers', options=['select', 'range'], min_val=1, max_val=10)
        validate_opt_param(self.hidden_size, 'hidden_size', options=['select', 'range'], min_val=1, max_val=1024)
        validate_opt_param(self.dropout, 'dropout', options=['select', 'range'], min_val=0.0, max_val=1.0)
        validate_opt_param(self.bias, 'bias', options=['select'], select_from=[True, False])
        return self

class RNN(nn.Module, ModelSimulator):
    def __init__(self, config: RNNConfig):
        nn.Module.__init__(self)
        ModelSimulator.__init__(self, config.sim_config)
        self.config = config
        self.rnn_type = config.rnn_type
        self.sequence_length = config.sequence_length
        self.hidden_layers = config.hidden_layers
        self.hidden_size = config.hidden_size        
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(config.num_inputs, self.hidden_size, self.hidden_layers, dropout=config.dropout, bias=config.bias, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(config.num_inputs, self.hidden_size, self.hidden_layers, dropout=config.dropout, bias=config.bias, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(config.num_inputs, self.hidden_size, self.hidden_layers, dropout=config.dropout, bias=config.bias, batch_first=True)
        else:
            raise ValueError("Invalid RNN type. Choose from 'RNN', 'LSTM', or 'GRU'.")
        self.output_layer = nn.Linear(self.hidden_size, config.num_outputs)
        self.hidden_state = None
    
    def reset_hidden_state(self, batch_size, device):
        if self.rnn_type == 'LSTM':
            self.hidden_state = (torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device),
                                    torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device))
        else:
            self.hidden_state = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=device)
    
    def forward(self, x):
        # Reset hidden state if this is the first use or batch size has changed
        if self.hidden_state is None:
            self.reset_hidden_state(x.size(0), x.device)
        elif self.rnn_type == 'LSTM':
            if self.hidden_state[0].size(1) != x.size(0):
                self.reset_hidden_state(x.size(0), x.device)
        else:
            if self.hidden_state.size(1) != x.size(0):
                self.reset_hidden_state(x.size(0), x.device)
                    
        rnn_output, next_hidden_state = self.rnn(x, self.hidden_state)
        
        network_output = self.output_layer(rnn_output[:, -1, :])
        return network_output