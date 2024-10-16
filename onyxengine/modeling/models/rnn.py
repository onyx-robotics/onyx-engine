import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from onyxengine.modeling import ModelSimulatorConfig, ModelSimulator

class RNNConfig(BaseModel):
    onyx_model_type: str = Field(default='rnn', frozen=True, init=False)
    sim_config: ModelSimulatorConfig = ModelSimulatorConfig()
    num_inputs: int = 1
    num_outputs: int = 1
    sequence_length: int = 1 # Not needed by pytorch rnn's but useful for model tracking
    hidden_layers: int = 2
    hidden_size: int = 32
    rnn_type: str = 'RNN'
    streaming_mode: bool = False

class RNN(nn.Module, ModelSimulator):
    def __init__(self, config: RNNConfig):
        nn.Module.__init__(self)
        ModelSimulator.__init__(self, config.sim_config)
        self.config = config
        self.sequence_length = config.sequence_length
        num_inputs = config.num_inputs
        num_outputs = config.num_outputs
        self.hidden_layers = config.hidden_layers
        self.hidden_size = config.hidden_size
        self.rnn_type = config.rnn_type
        self.streaming_mode = config.streaming_mode
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(num_inputs, self.hidden_size, self.hidden_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(num_inputs, self.hidden_size, self.hidden_layers, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(num_inputs, self.hidden_size, self.hidden_layers, batch_first=True)
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