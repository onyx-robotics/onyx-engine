# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Onyx Engine is a Python SDK for optimizing AI models for hardware simulation and control. It provides a client library that communicates with a remote Onyx Engine server to train, optimize, and manage machine learning models. The library supports automatic tracing and provides a platform for turning data into performant, deployable models.

## Build and Development Commands

```bash
# Install dependencies
pip install -e .
```

## Environment Configuration

Create a `.env` file in the project root:
```
ONYX_API_KEY=your_api_key_here
ONYX_ENGINE_DEV_MODE=False  # Set to True for localhost:7000
```

The SDK uses:
- Production server: `api.onyx-robotics.com`
- Dev server: `localhost:7000`

## Architecture

### Package Structure

```
onyxengine/
├── __init__.py          # Config constants (SERVER_URL, API_KEY, paths)
├── api.py               # Public API functions (save/load datasets/models, train, optimize)
├── api_utils.py         # HTTP/WebSocket utilities, file upload/download
├── data/
│   └── dataset.py       # OnyxDataset and OnyxDatasetConfig classes
└── modeling/
    ├── model_features.py    # Input/Output feature definitions with scaling
    ├── model_base_config.py # Base Pydantic configs for models
    ├── model_simulator.py   # Base class for all models (simulation interface)
    ├── model_from_config.py # Factory: config dict → model instance
    ├── model_training.py    # TrainingConfig, OptimizationConfig
    ├── models/              # MLP, RNN, Transformer (PyTorch nn.Module)
    ├── optimizers/          # AdamW, SGD configs
    └── lr_schedulers/       # Cosine decay schedulers
```

### Key Patterns

**Discriminated Unions**: Model/optimizer/scheduler configs use Pydantic's `Annotated[Union[...], Field(discriminator='type')]` for polymorphic serialization.

**Config vs OptConfig**: Each component has two config variants:
- `*Config`: Concrete values for training (e.g., `MLPConfig`)
- `*OptConfig`: Search spaces for optimization with `{"select": [...]}` or `{"range": [min, max, step]}`

**Feature System**: Models are built around `Input` and `Output` features that can have derivation relationships (delta, derivative) and automatic scaling.

**Client-Server Model**: The SDK sends configs to the remote Engine server for training/optimization. Local caching of datasets and models in `./onyx/datasets/` and `./onyx/models/`.

### Public API (onyxengine.api)

- `save_dataset()` / `load_dataset()` - Dataset management
- `save_model()` / `load_model()` - Model management
- `train_model()` - Train a single model configuration
- `optimize_model()` - Hyperparameter search across model configs
- `get_object_metadata()` - Retrieve versioned metadata

### Model Types

All models inherit from `nn.Module` and `ModelSimulator`:
- **MLP**: Feedforward network with configurable hidden layers, activation, dropout
- **RNN**: Supports RNN/LSTM/GRU variants
- **Transformer**: GPT-style decoder-only architecture

## Documentation

Documentation lives in a separate repository at `~/onyx/docs` (Mintlify-based, deployed to docs.onyx-robotics.com).

When making changes to the SDK, update corresponding documentation:
- **API changes**: Update `~/onyx/docs/api-reference/` (e.g., `train-model.mdx`, `mlp-config.mdx`)
- **New features**: Add to tutorials in `~/onyx/docs/tutorials/`
- **Concept changes**: Update `~/onyx/docs/concepts/`

Key documentation files that mirror SDK components:
- `api-reference/train-model.mdx` ↔ `onyxengine/api.py:train_model()`
- `api-reference/mlp-config.mdx` ↔ `onyxengine/modeling/models/mlp.py`
- `api-reference/input.mdx`, `output.mdx` ↔ `onyxengine/modeling/model_features.py`
- `tutorials/training-models.mdx` ↔ Training workflow
