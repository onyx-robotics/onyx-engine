import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from onyxengine.modeling import ModelSimulatorConfig, ModelSimulator

class RNNConfig(BaseModel):
    """
    Configuration class for the RNN model.
    
    Args:
        onyx_model_type (str): Model type = 'rnn', immutable.
        sim_config (ModelSimulatorConfig): Configuration for the model's simulator.
        rnn_type (str): Type of RNN to use (default is 'RNN').
        num_inputs (int): Number of input features (default is 1).
        num_outputs (int): Number of output features (default is 1).
        sequence_length (int): Length of the input sequence (default is 1).
        hidden_layers (int): Number of hidden layers (default is 2).
        hidden_size (int): Size of each hidden layer (default is 32).
        dropout (float): Dropout rate (default is 0.0).
        streaming_mode (bool): Whether to use streaming mode (default is False).
    """
    onyx_model_type: str = Field(default='rnn', frozen=True, init=False)
    sim_config: ModelSimulatorConfig = ModelSimulatorConfig()
    rnn_type: str = 'RNN'
    num_inputs: int = 1
    num_outputs: int = 1
    sequence_length: int = 1 # Not needed by pytorch rnn's but useful for model tracking
    hidden_layers: int = 2
    hidden_size: int = 32
    dropout: float = 0.0
    streaming_mode: bool = False

class RNN(nn.Module, ModelSimulator):
    def __init__(self, config: RNNConfig):
        nn.Module.__init__(self)
        ModelSimulator.__init__(self, config.sim_config)
        self.config = config
        self.rnn_type = config.rnn_type
        self.sequence_length = config.sequence_length
        num_inputs = config.num_inputs
        num_outputs = config.num_outputs
        self.hidden_layers = config.hidden_layers
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.streaming_mode = config.streaming_mode
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(num_inputs, self.hidden_size, self.hidden_layers, dropout=self.dropout, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(num_inputs, self.hidden_size, self.hidden_layers, dropout=self.dropout, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(num_inputs, self.hidden_size, self.hidden_layers, dropout=self.dropout, batch_first=True)
        else:
            raise ValueError("Invalid RNN type. Choose from 'RNN', 'LSTM', or 'GRU'.")
        self.output_layer = nn.Linear(self.hidden_size, num_outputs)
        self.hidden_state = None
        
    def set_streaming_mode(self, mode_setting: bool):
        self.streaming_mode = mode_setting
    
    def reset_hidden_state(self):
        self.hidden_state = None
    
    def forward(self, x):
        # Init hidden state if not streaming or if this is the first pass
        if self.streaming_mode == False or self.hidden_state == None:
            if self.rnn_type == 'LSTM':
                self.hidden_state = (torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=x.device),
                                     torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=x.device))
            else:
                self.hidden_state = torch.zeros(self.hidden_layers, x.size(0), self.hidden_size, device=x.device)
                    
        rnn_output, next_hidden_state = self.rnn(x, self.hidden_state)
        
        if self.streaming_mode == True:
            self.hidden_state = next_hidden_state
        
        network_output = self.output_layer(rnn_output[:, -1, :])
        return network_output