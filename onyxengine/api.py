import os
import json
from typing import List, Optional, Dict, Union
import torch
import pandas as pd
from onyxengine import DATASETS_PATH, MODELS_PATH
from onyxengine.data import OnyxDataset, OnyxDatasetConfig
from onyxengine.modeling.models import *
from onyxengine.modeling import model_from_config, TrainingConfig, OptimizationConfig, ModelSimulatorConfig
from .api_utils import handle_post_request, upload_object, download_object, set_object_metadata, monitor_training_job, SourceObject
import asyncio

def get_object_metadata(object_name: str, object_version: int=None) -> dict:
    """
    Get the metadata for an object in the Engine.
    
    Args:
        object_name (str): The name of the object to get metadata for
        object_version (int, optional): The version of the object to get metadata for. (Default is None)
    
    Returns:
        dict: The metadata for the object, or None if the object does not exist.
        
    Example:
    
    .. code-block:: python
    
        metadata = onyx.get_object_metadata('example_data')
        print(metadata)
        
    """
    assert isinstance(object_name, str), "object_name must be a string."
    assert object_version is None or isinstance(object_version, int), "object_version must be an integer."
    
    # Get metadata for the object in onyx engine
    response = handle_post_request("/get_object_metadata", {"object_name": object_name, "object_version": object_version})
    if response is None:
        return None
    
    metadata = json.loads(response)
    if isinstance(metadata['object_config'], str):
        metadata['object_config'] = json.loads(metadata['object_config'])
    
    return metadata

def save_dataset(name: str, dataset: OnyxDataset, source_datasets: List[Dict[str, Optional[str]]]=[]):
    """
    Save a dataset to the Engine.
    
    Args:
        name (str): The name for the new dataset
        dataset (OnyxDataset): The OnyxDataset object to save
        source_datasets (List[Dict[str, Optional[str]]]): The source datasets used as a list of dictionaries like [{'name': 'dataset_name', 'version': 'dataset_version'}]
        
    Example:
    
    .. code-block:: python

        # Load data
        raw_data = onyx.load_dataset('example_data')

        # Pull out features for model training
        train_data = pd.DataFrame()
        train_data['acceleration_predicted'] = raw_data.dataframe['acceleration']
        train_data['velocity'] = raw_data.dataframe['velocity']
        train_data['position'] = raw_data.dataframe['position']
        train_data['brake_input'] = raw_data.dataframe['brake_input']
        train_data = train_data.dropna()

        # Save training dataset
        train_dataset = OnyxDataset(
            dataframe=train_data,
            num_outputs=1,
            num_state=2,
            num_control=1,
            dt=0.0025
        )
        onyx.save_dataset("example_train_data", dataset=train_dataset, source_dataset_names=['example_data'])
    
    """
    assert isinstance(name, str), "name must be a string."
    assert isinstance(dataset, OnyxDataset), "dataset must be an OnyxDataset."
    source_datasets = [SourceObject.model_validate(source) for source in source_datasets]
    
    # Validate the dataset dataframe, name, and source datasets
    if dataset.dataframe.empty:
        raise SystemExit("Onyx Engine API error: Dataset dataframe is empty.")
    if name == '':
        raise SystemExit("Onyx Engine API error: Dataset name must be a non-empty string.")
    for source in source_datasets:
        if get_object_metadata(source.name, source.version) is None:
            raise SystemExit(f"Onyx Engine API error: Source dataset [{source}] not found in the Engine.")
    
    # Save a local copy of the dataset
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    dataset_filename = name + '.csv'
    dataset.dataframe.to_csv(os.path.join(DATASETS_PATH, dataset_filename), index=False)
    
    # Upload the dataset and config to the cloud
    upload_object(dataset_filename, 'dataset')
    set_object_metadata(name, 'dataset', dataset.config.model_dump_json(), source_datasets)
    
    # Get the object metadata and save locally
    metadata = get_object_metadata(name)
    with open(os.path.join(DATASETS_PATH, name + '.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=2))
        
    print(f'Dataset [{name}] saved to the Engine.')

def load_dataset(name: str, version: int=None) -> OnyxDataset:
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
    assert version is None or isinstance(version, int), "version must be an integer."
    config_filename = name + '.json'
    dataset_filename = name + '.csv'
    dataset_path = os.path.join(DATASETS_PATH, dataset_filename)
    config_path = os.path.join(DATASETS_PATH, config_filename)

    # Get dataset metadata
    metadata = get_object_metadata(name, version)
    if metadata is None:
        raise SystemExit(f"Onyx Engine API error: Dataset [{name}:v{version}] not found in the Engine.")

    def download_dataset():
        download_object(dataset_filename, 'dataset', version)
        with open(os.path.join(config_path), 'w') as f:
            f.write(json.dumps(metadata, indent=2))

    # If the dataset doesn't exist locally, download it
    if not os.path.exists(dataset_path):
        if not os.path.exists(DATASETS_PATH):
            os.makedirs(DATASETS_PATH)
        download_dataset()
    else:
        # Else check if local version is outdated or does not match requested version
        with open(os.path.join(config_path), 'r') as f:
            local_metadata = json.load(f)
        if version is None and metadata['version'] != local_metadata['version']:
            download_dataset()
        elif version is not None and version != local_metadata['version']:
            download_dataset()

    # Load the dataset from local storage
    dataset_dataframe = pd.read_csv(dataset_path)
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())
    dataset_config = OnyxDatasetConfig.model_validate(config_json["object_config"])
    dataset = OnyxDataset(config=dataset_config, dataframe=dataset_dataframe)

    return dataset

def save_model(name: str, model: torch.nn.Module, source_datasets: List[Dict[str, Optional[str]]]=[]):
    """
    Save a model to the Engine. Generally you won't need to use this function as the Engine will save models it trains automatically.
    
    Args:
        name (str): The name for the new model.
        model (torch.nn.Module): The PyTorch model to save.
        source_datasets (List[Dict[str, Optional[str]]]): The source datasets used as a list of dictionaries like [{'name': 'dataset_name', 'version': 'dataset_version'}]
        
    Example:
    
    .. code-block:: python
    
        # Create model configuration
        sim_config = ModelSimulatorConfig(
             outputs=['acceleration'],
             states=[
                 State(name='velocity', relation='derivative', parent='acceleration'),
                 State(name='position', relation='derivative', parent='velocity'),
             ],
             controls=['control_input'],
             dt=0.0025
         )
        mlp_config = MLPConfig(
             sim_config=sim_config,
             num_inputs=sim_config.num_inputs,
             num_outputs=sim_config.num_outputs,
             hidden_layers=2,
             hidden_size=32,
             activation='relu',
             dropout=0.2,
             bias=True
         )
        # Create and save model
        model = MLP(mlp_config)
        onyx.save_model("example_model", model, 
                   source_dataset_names=['example_train_data'])
                   
    """
    assert isinstance(name, str), "name must be a string."
    assert isinstance(model, torch.nn.Module), "model must be an Onyx model."
    source_datasets = [SourceObject.model_validate(source) for source in source_datasets]
    
    # Validate the model name and source datasets
    if name == '':
        raise SystemExit("Onyx Engine API error: Model name must be a non-empty string.")
    for source in source_datasets:
        if get_object_metadata(source.name, source.version) is None:
            raise SystemExit(f"Onyx Engine API error: Source dataset [{source}] not found in the Engine.")
    
    # Save model to local storage
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    model_filename = name + '.pt'
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, model_filename))
    
    # Upload the model and config to the cloud
    upload_object(model_filename, 'model')
    set_object_metadata(name, 'model', model.config.model_dump_json(), source_datasets)
    
    # Get the object metadata and save locally
    metadata = get_object_metadata(name)
    with open(os.path.join(MODELS_PATH, name + '.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=2))
    
    print(f'Model [{name}] saved to the Engine.')

def load_model(name: str, version: int=None) -> torch.nn.Module:
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
    assert version is None or isinstance(version, int), "version must be an integer."
    model_filename = name + '.pt'
    config_filename = name + '.json'
    model_path = os.path.join(MODELS_PATH, model_filename)
    config_path = os.path.join(MODELS_PATH, config_filename)

    # Get model metadata
    metadata = get_object_metadata(name, version)
    if metadata is None:
        raise SystemExit(f"Onyx Engine API error: Model [{name}:v{version}] not found in the Engine.")

    def download_model():
        download_object(model_filename, 'model', version)
        with open(os.path.join(config_path), 'w') as f:
            f.write(json.dumps(metadata, indent=2))

    # If the model doesn't exist locally, download it
    if not os.path.exists(model_path):
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        download_model()
    else:
        # Else check if local version is outdated or does not match requested version
        with open(os.path.join(config_path), 'r') as f:
            local_metadata = json.load(f)
        if version is None and metadata['version'] != local_metadata['version']:
            download_model()
        elif version is not None and version != local_metadata['version']:
            download_model()

    # Load the model from local storage
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())
    model = model_from_config(config_json['object_config'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model

def train_model(
    model_name: str = "",
    model_config: Union[MLPConfig, RNNConfig, TransformerConfig] = None,
    dataset_name: str = "",
    dataset_version: Optional[int] = None,
    training_config: TrainingConfig = TrainingConfig(),
    monitor_training: bool = True,
):
    """
    Train a model on the Engine using a specified dataset, model config, and training config.
    
    Args:
        model_name (str): The name of the model to train. (Required)
        model_config (Union[MLPConfig, RNNConfig, TransformerConfig]): The configuration for the model to train. (Required)
        dataset_name (str): The name of the dataset to train on. (Required)
        dataset_version (int, optional): The version of the dataset to train on, None = current_version. (Default is None)
        training_config (TrainingConfig): The configuration for the training process. (Default is TrainingConfig())
        monitor_training (bool, optional): Whether to monitor the training job. (Default is True)
    """
    assert isinstance(model_name, str), "model_name must be a string."
    assert isinstance(model_config, (MLPConfig, RNNConfig, TransformerConfig)), "model_config must be a model config."
    assert isinstance(dataset_name, str), "dataset_name must be a string."
    assert dataset_version is None or isinstance(dataset_version, int), "dataset_version must be an integer."
    assert isinstance(training_config, TrainingConfig), "training_config must be a TrainingConfig."
    assert isinstance(monitor_training, bool), "monitor_training must be a boolean."

    # Check that model/dataset names are not empty
    if model_name == '':
        raise SystemExit("Onyx Engine API error: Model name must be a non-empty string.")
    if dataset_name == '':
        raise SystemExit("Onyx Engine API error: Dataset name must be a non-empty string.")

    # Check that the dataset exists
    if get_object_metadata(dataset_name, dataset_version) is None:
        raise SystemExit(f"Onyx Engine API error: Dataset [{dataset_name}:v{dataset_version}] not found in the Engine.")

    # Request the onyx server to train the model
    response = handle_post_request("/train_model", {
        "onyx_model_name": model_name,
        "onyx_model_config": model_config.model_dump_json(),
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "training_config": training_config.model_dump_json(),
    })

    print(f'Preparing to train model [{model_name}] using dataset [{dataset_name}].')    
    if monitor_training:
        try:
            asyncio.run(monitor_training_job(response['job_id'], training_config))
        except KeyboardInterrupt:
            print('Training job monitoring stopped.')


def optimize_model(
    model_name: str = "",
    model_sim_config: ModelSimulatorConfig = None,
    dataset_name: str = "",
    dataset_version: Optional[int] = None,
    optimization_config: OptimizationConfig = None,
):
    """
    Optimize a model on the Engine using a specified dataset, model simulator config, and optimization config.
    
    Args:
        model_name (str): The name of the model to optimize. (Required)
        model_sim_config (ModelSimulatorConfig): The configuration for the model simulator. (Required)
        dataset_name (str): The name of the dataset to optimize on. (Required)
        dataset_version (int, optional): The version of the dataset to optimize on, None = current_version. (Default is None)
        optimization_config (OptimizationConfig): The configuration for the optimization process. (Required)

    
    Example:
    
    .. code-block:: python
    

    """
    assert isinstance(model_name, str), "model_name must be a string."
    assert isinstance(model_sim_config, ModelSimulatorConfig), "model_sim_config is required and must be a ModelSimulatorConfig."
    assert isinstance(dataset_name, str), "dataset_name must be a string."
    assert dataset_version is None or isinstance(dataset_version, int), "dataset_version must be an integer."
    assert isinstance(optimization_config, OptimizationConfig), "optimization_config is required and must be an OptimizationConfig."

    # Check that model/dataset names are not empty
    if model_name == '':
        raise SystemExit("Onyx Engine API error: Model name must be a non-empty string.")
    if dataset_name == '':
        raise SystemExit("Onyx Engine API error: Dataset name must be a non-empty string.")

    # Check that the dataset exists
    if get_object_metadata(dataset_name) is None:
        raise SystemExit(f"Onyx Engine API error: Dataset [{dataset_name}] not found in the Engine.")

    # Request the onyx server to train the model
    response = handle_post_request("/optimize_model", {
        "onyx_model_name": model_name,
        "onyx_model_sim_config": model_sim_config.model_dump_json(),
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "optimization_config": optimization_config.model_dump_json(),
    })

    print(f'Preparing to optimize model [{model_name}] using dataset [{dataset_name}].')
