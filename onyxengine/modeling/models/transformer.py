# Transformer with GPT decoder-only architecture
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from pydantic import Field, model_validator
from typing_extensions import Self
from typing import List, Union, Dict, Literal
from onyxengine.modeling import (
    OnyxModelBaseConfig,
    OnyxModelOptBaseConfig,
    validate_param,
    validate_opt_param,
    ModelSimulator,
    FeatureScaler,
)

class TransformerConfig(OnyxModelBaseConfig):
    """
    Configuration class for the Transformer model.
    
    Args:
        type (str): Model type = 'transformer', immutable.
        outputs (List[Output]): List of output variables.
        inputs (List[Input | State]): List of input variables.
        dt (float): Time step for the model.
        sequence_length (int): Length of the input sequence (default is 1).
        n_layer (int): Number of transformer layers (default is 1).
        n_head (int): Number of attention heads (default is 4).
        n_embd (int): Size of the embedding dimension (default is 32).
        dropout (float): Dropout rate for layers (default is 0.0).
        bias (bool): Whether to use bias in layers (default is True).
    """
    type: Literal['transformer'] = Field(default='transformer', frozen=True, init=False)
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 32
    dropout: float = 0.0
    bias: bool = True # Bias in Linears and LayerNorms

    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.n_layer, 'n_layer', min_val=1, max_val=10)
        validate_param(self.n_head, 'n_head', min_val=1, max_val=12)
        validate_param(self.n_embd, 'n_embd', min_val=1, max_val=1024)
        validate_param(self.dropout, 'dropout', min_val=0.0, max_val=1.0)
        # n_embd must be divisible by n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        return self

class TransformerOptConfig(OnyxModelOptBaseConfig):
    """
    Optimization config class for the Transformer model.
    
    Args:
        type (str): Model type = 'transformer_opt', immutable.
        outputs (List[Output]): List of output variables.
        inputs (List[Input | State]): List of input variables.
        dt (float): Time step for the model.
        sequence_length (Union[int, Dict[str, List[int]]): Length of the input sequence (default is {"select": [1, 2, 4, 5, 6, 8, 10, 12, 14, 15]}).
        n_layer (Union[int, Dict[str, List[int]]): Number of transformer layers (default is {"range": [2, 5, 1]}).
        n_head (Union[int, Dict[str, List[int]]): Number of attention heads (default is {"range": [2, 10, 2]}).
        n_embd (Union[int, Dict[str, List[int]]): Size of the embedding dimension (default is {"select": [12, 24, 32, 64, 128]}).
        dropout (Union[float, Dict[str, List[float]]): Dropout rate for layers (default is {"range": [0.0, 0.4, 0.1]}).
        bias (Union[bool, Dict[str, List[bool]]): Whether to use bias in layers (default is True).
    """
    type: Literal['transformer_opt'] = Field(default='transformer_opt', frozen=True, init=False)
    n_layer: Union[int, Dict[str, List[int]]] = {"range": [2, 5, 1]}
    n_head: Union[int, Dict[str, List[int]]] = {"range": [2, 10, 2]}
    n_embd: Union[int, Dict[str, List[int]]] = {"select": [12, 24, 32, 64, 128]}
    dropout: Union[float, Dict[str, List[float]]] = {"range": [0.0, 0.4, 0.1]}
    bias: Union[bool, Dict[str, List[bool]]] = True

    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_opt_param(self.n_layer, 'n_layer', options=['select', 'range'], min_val=1, max_val=10)
        validate_opt_param(self.n_head, 'n_head', options=['select', 'range'], min_val=1, max_val=12)
        validate_opt_param(self.n_embd, 'n_embd', options=['select', 'range'], min_val=1, max_val=1024)
        validate_opt_param(self.dropout, 'dropout', options=['select', 'range'], min_val=0.0, max_val=1.0)
        validate_opt_param(self.bias, 'bias', options=['select'], select_from=[True, False])
        return self

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                        .view(1, 1, config.sequence_length, config.sequence_length))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class GPT_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPT_MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module, ModelSimulator):
    def __init__(self, config: TransformerConfig):
        nn.Module.__init__(self)
        ModelSimulator.__init__(
            self,
            outputs=config.outputs,
            inputs=config.inputs,
            sequence_length=config.sequence_length,
            dt=config.dt,
        )        
        self.feature_scaler = FeatureScaler(outputs=config.outputs, inputs=config.inputs)
        self.config = config
        num_inputs = len(config.inputs)
        num_outputs = len(config.outputs)
        
        # Continuous states embedded with linear layer instead of token-level nn.Embedding
        self.transformer = nn.ModuleDict(dict(
            state_embedding = nn.Linear(num_inputs, config.n_embd),
            pos_embedding = nn.Embedding(config.sequence_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, num_outputs, bias=False)
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        device = x.device
        batch_size, seq_len, num_input_dim = x.size()
        assert seq_len <= self.config.sequence_length, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.sequence_length}"
        
        # Scale inputs
        x = self.feature_scaler.scale_inputs(x)
        
        # Positional embedding
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_embd = self.transformer.pos_embedding(pos) # Shape (seq_len, num_embd)
        x_embd = self.transformer.state_embedding(x) # Shape (batch, seq_len, num_embd)
        
        # Transformer decoder forward
        x = self.transformer.drop(x_embd + pos_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Project back out to embedded outputs
        output = self.lm_head(x[:, [-1], :]).squeeze(1)
        return self.feature_scaler.unscale_outputs(output)
