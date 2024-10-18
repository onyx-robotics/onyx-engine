# Transformer with GPT decoder-only architecture
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from pydantic import BaseModel, Field
from onyxengine.modeling import ModelSimulatorConfig, ModelSimulator

class TransformerConfig(BaseModel):
    onyx_model_type: str = Field(default='transformer', frozen=True, init=False)
    sim_config: ModelSimulatorConfig = ModelSimulatorConfig()
    num_inputs: int = 0
    num_outputs: int = 0
    sequence_length: int = 1
    n_layer: int = 1
    n_head: int = 12
    n_embd: int = 24
    dropout: float = 0.0
    bias: bool = True # Bias in Linears and LayerNorms

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
        ModelSimulator.__init__(self, config.sim_config)
        self.config = config
        
        # Continuous states embedded with linear layer instead of token-level nn.Embedding
        self.transformer = nn.ModuleDict(dict(
            state_embedding = nn.Linear(config.num_inputs, config.n_embd),
            pos_embedding = nn.Embedding(config.sequence_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.num_outputs, bias=False)
        
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
        return output