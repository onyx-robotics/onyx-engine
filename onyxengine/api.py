import os
import json
from typing import List
import torch
import pandas as pd
from pydantic import BaseModel
from onyxengine import DATASETS_PATH, MODELS_PATH
from onyxengine.data import OnyxDataset, OnyxDatasetConfig
from onyxengine.modeling import model_from_config_json, TrainingConfig
from onyxengine.modeling.models import *
from .api_utils import handle_post_request, upload_object, download_object, set_object_metadata

def get_object_metadata(object_name: str) -> dict:
    """
    Get the metadata for an object in the Engine.
    
    Args:
        object_name (str): The name of the object to get metadata for
    
    Returns:
        dict: The metadata for the object, or None if the object does not exist.
        
    Example:
        >>> metadata = onyx.get_object_metadata('example_data')
        >>> print(metadata)
    """
    assert isinstance(object_name, str), "object_name must be a string."
    
    # Get metadata for the object in onyx engine
    response = handle_post_request("/get_object_metadata", {"object_name": object_name})
    if response is None:
        return None
    
    return json.loads(response)

def save_dataset(name: str, dataset: OnyxDataset, source_dataset_names: List[str]=[]):
    """
    Save a dataset to the Engine.
    
    Args:
        name (str): The name for the new dataset
        dataset (OnyxDataset): The OnyxDataset object to save
        source_dataset_names (List[str]): The names of the source datasets used to create this dataset
        
    Example:
        >>> # Create a training dataset from raw data
        >>> raw_data = onyx.load_dataset('example_data')
        >>> train_data = pd.DataFrame()
        >>> train_data['acceleration'] = raw_data.dataframe['acceleration']
        >>> train_data['velocity'] = raw_data.dataframe['velocity']
        >>> train_data['position'] = raw_data.dataframe['position']
        >>> train_data['control_input'] = raw_data.dataframe['control_input']
        >>> train_data = train_data.dropna()
        >>>
        >>> # Create and save the training dataset
        >>> train_dataset = OnyxDataset(
        ...     dataframe=train_data,
        ...     num_outputs=1,
        ...     num_state=2,
        ...     num_control=1,
        ...     dt=0.0025
        ... )
        >>> onyx.save_dataset("example_train_data", dataset=train_dataset, 
        ...             source_dataset_names=['example_data'])
    """
    assert isinstance(name, str), "name must be a string."
    assert isinstance(dataset, OnyxDataset), "dataset must be an OnyxDataset."
    
    # Check that source datasets exist
    for source in source_dataset_names:
        if get_object_metadata(source) is None:
            raise SystemExit(f"Onyx Engine API error: Source dataset [{source}] not found in the Engine.")
    
    # Save a local copy of the dataset and its config
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    dataset_filename = name + '.csv'
    config_filename = name + '.json'
    config_json = dataset.config.model_dump_json(indent=2)
    dataset.dataframe.to_csv(os.path.join(DATASETS_PATH, dataset_filename), index=False)
    with open(os.path.join(DATASETS_PATH, config_filename), 'w') as f:
        f.write(config_json)
    
    # Upload the dataset and config to the cloud
    upload_object(config_filename, 'dataset')
    upload_object(dataset_filename, 'dataset')
    set_object_metadata(name, 'dataset', config_json, source_dataset_names)
    print(f'Dataset [{name}] saved to the Engine.')

def load_dataset(name: str, use_cache=True) -> OnyxDataset:
    """
    Load a dataset from the Engine, either from a local cached copy or by downloading from the Engine.
    
    Args:
        name (str): The name of the dataset to load.
        use_cache (bool, optional): Whether to use the cached local version of the dataset. Defaults to True.
    
    Returns:
        OnyxDataset: The loaded dataset.
        
    Example:
        >>> # Load a dataset
        >>> dataset = onyx.load_dataset('example_data')
        >>> print(dataset.dataframe.head())
    """
    assert isinstance(name, str), "name must be a string."
    
    # Check that the dataset exists
    if get_object_metadata(name) is None:
        raise SystemExit(f"Onyx Engine API error: Dataset [{name}] not found in the Engine.")
    
    # Get dataset and config filenames
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    dataset_filename = name + '.csv'
    config_filename = name + '.json'
    dataset_path = os.path.join(DATASETS_PATH, dataset_filename)
    config_path = os.path.join(DATASETS_PATH, config_filename)
    
    # Download the dataset from the cloud if it doesn't exist locally
    if not os.path.exists(dataset_path) or not use_cache:
        print(f'Downloading [{name}] from the Engine...')
        download_object(dataset_filename, 'dataset')
        download_object(config_filename, 'dataset')
    else:
        print(f'Using local copy of [{name}], set use_cache=False to re-download.')
            
    # Load the dataset and config from local storage
    dataset_dataframe = pd.read_csv(dataset_path)
    with open(config_path, 'r') as f:
        config_json = f.read()
    dataset_config = OnyxDatasetConfig.model_validate_json(config_json)
    dataset = OnyxDataset(dataframe=dataset_dataframe)
    dataset.from_config(dataset_config)
    
    return dataset

def save_model(name: str, model: torch.nn.Module, source_dataset_names: List[str]=[]):
    """
    Save a model to the Engine. Generally you won't need to use this function as the Engine will save models it trains automatically.
    
    Args:
        name (str): The name for the new model.
        model (torch.nn.Module): The PyTorch model to save.
        source_dataset_names (List[str]): The names of the source datasets used to train the model.
        
    Example:
        >>> # Create model configuration
        >>> sim_config = ModelSimulatorConfig(
        ...     outputs=['acceleration'],
        ...     states=[
        ...         State(name='velocity', relation='derivative', parent='acceleration'),
        ...         State(name='position', relation='derivative', parent='velocity'),
        ...     ],
        ...     controls=['control_input'],
        ...     dt=0.0025
        ... )
        >>> mlp_config = MLPConfig(
        ...     sim_config=sim_config,
        ...     num_inputs=sim_config.num_inputs,
        ...     num_outputs=sim_config.num_outputs,
        ...     hidden_layers=2,
        ...     hidden_size=32,
        ...     activation='relu',
        ...     dropout=0.2,
        ...     bias=True
        ... )
        >>> # Create and save model
        >>> model = MLP(mlp_config)
        >>> onyx.save_model("example_model", model, 
        ...           source_dataset_names=['example_train_data'])
    """
    assert isinstance(name, str), "name must be a string."
    assert isinstance(model, torch.nn.Module), "model must be an Onyx model."
    
    # Check that source datasets exist
    for source in source_dataset_names:
        if get_object_metadata(source) is None:
            raise SystemExit(f"Onyx Engine API error: Source dataset [{source}] not found in the Engine.")
    
    # Save model to local storage
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    model_filename = name + '.pt'
    config_filename = name + '.json'
    config_json = model.config.model_dump_json(indent=2)
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, model_filename))
    with open(os.path.join(MODELS_PATH, config_filename), 'w') as f:
        f.write(config_json)
    
    # Upload the dataset and config to the cloud
    upload_object(config_filename, 'model')
    upload_object(model_filename, 'model')
    set_object_metadata(name, 'model', config_json, source_dataset_names)
    print(f'Model [{name}] saved to the Engine.')

def load_model(name: str, use_cache=True) -> torch.nn.Module:
    """
    Load a model from the Engine, either from a local cached copy or by downloading from the Engine.
    
    Args:
        name (str): The name of the model to load.
        use_cache (bool, optional): Whether to use the cached local version of the model. Defaults to True.
    
    Returns:
        torch.nn.Module: The loaded Onyx model.
        
    Example:
        >>> # Load our model
        >>> model = load_model('example_model')
        >>> 
        >>> # Run basic inference with our model
        >>> test_input = torch.ones(1, 1, 3)
        >>> with torch.no_grad():
        ...     test_output = model(test_input)
        >>> print(test_output)
        >>> 
        >>> # Simulate a trajectory with our model
        >>> # Model will fill in the x_trajectory tensor with the simulated trajectory
        >>> batch_size = 1
        >>> seq_length = 1
        >>> sim_steps = 100
        >>> x0 = torch.ones(batch_size, seq_length, 2)
        >>> u = torch.ones(batch_size, sim_steps, 1)
        >>> 
        >>> x_traj = torch.zeros(1, sim_steps, 3)
        >>> model.simulate(x_traj, x0, u)
        >>> print(x_traj)
    """
    assert isinstance(name, str), "name must be a string."
    
    # Check that the model exists
    if get_object_metadata(name) is None:
        raise SystemExit(f"Onyx Engine API error: Model [{name}] not found in the Engine.")
    
    # Get model and config filenames
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    model_filename = name + '.pt'
    config_filename = name + '.json'
    model_path = os.path.join(MODELS_PATH, model_filename)
    config_path = os.path.join(MODELS_PATH, config_filename)
    
    # Download the model from the cloud if it doesn't exist locally
    if not os.path.exists(model_path) or not use_cache:
        print(f'Downloading [{name}] from the Engine...')
        download_object(model_filename, 'model')
        download_object(config_filename, 'model')
    else:
        print(f'Using local copy of [{name}], set use_cache=False to re-download.')
            
    # Load the model using config and state_dict from local storage
    with open(config_path, 'r') as f:
        config_json = f.read()
    model = model_from_config_json(config_json)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def train_model(dataset_name: str, model_name: str, model_config: BaseModel, training_config: TrainingConfig):
    """
    Train a model on the Engine using a specified dataset, model config, and training config.
    
    Args:
        dataset_name (str): The name of the dataset to train on
        model_name (str): The name of the model to train
        model_config (BaseModel): The configuration for the model
        training_config (TrainingConfig): The configuration for the training process
        
    Example:
        >>> # Create model configuration
        >>> sim_config = ModelSimulatorConfig(
        ...     outputs=['acceleration'],
        ...     states=[
        ...         State(name='velocity', relation='derivative', parent='acceleration'),
        ...         State(name='position', relation='derivative', parent='velocity'),
        ...     ],
        ...     controls=['control_input'],
        ...     dt=0.0025
        ... )
        >>> model_config = MLPConfig(
        ...     sim_config=sim_config,
        ...     num_inputs=sim_config.num_inputs,
        ...     num_outputs=sim_config.num_outputs,
        ...     hidden_layers=2,
        ...     hidden_size=32,
        ...     activation='relu',
        ...     dropout=0.2,
        ...     bias=True
        ... )
        >>> 
        >>> # Create training configuration
        >>> training_config = TrainingConfig(
        ...     training_iters=3000,
        ... )
        >>> 
        >>> # Execute training
        >>> onyx.train_model(
        ...     dataset_name='example_train_data',
        ...     model_name='example_model',
        ...     model_config=model_config,
        ...     training_config=training_config,
        ... )
    """
    assert isinstance(dataset_name, str), "dataset_name must be a string."
    assert isinstance(model_name, str), "model_name must be a string."
    assert isinstance(model_config, BaseModel), "model_config must be an Onyx model config."
    assert isinstance(training_config, TrainingConfig), "training_config must be a TrainingConfig."
    
    # Check that the dataset exists
    if get_object_metadata(dataset_name) is None:
        raise SystemExit(f"Onyx Engine API error: Dataset [{dataset_name}] not found in the Engine.")
    
    # Prepare to make a request to the onyx server
    model_config_json = model_config.model_dump_json(indent=2)
    training_config_json = training_config.model_dump_json(indent=2)

    # Request the onyx server to train the model
    response = handle_post_request("/train_model", {
        "dataset_name": dataset_name,
        "onyx_model_name": model_name,
        "onyx_model_config": model_config_json,
        "training_config": training_config_json,
    })
    
    print(f'Started training for model [{model_name}] using dataset [{dataset_name}].')