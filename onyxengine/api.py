import os
import json
from typing import List, Optional, Dict, Union, Literal
import torch
import pandas as pd
from onyxengine import DATASETS_PATH, MODELS_PATH
from onyxengine.data import OnyxDataset, OnyxDatasetConfig
from onyxengine.modeling.models import *
from onyxengine.modeling import model_from_config, TrainingConfig, OptimizationConfig
from .api_utils import handle_post_request, upload_object, download_object, set_object_metadata, monitor_training_job, SourceObject
import asyncio

def get_object_metadata(name: str, version_id: str=None) -> dict:
    """
    Get the metadata for an object in the Engine.
    
    Args:
        name (str): The name of the object to get metadata for
        version_id (str, optional): The version id of the object to get metadata for, None = latest_version. (Default is None)
    
    Returns:
        dict: The metadata for the object, or None if the object does not exist.
        
    Example:
    
    .. code-block:: python
    
        # Get metadata for an Onyx object (dataset, model)
        metadata = onyx.get_object_metadata('example_data')
        print(metadata)
        
        # Get metadata for a specific version
        metadata = onyx.get_object_metadata('example_data', version_id='a05fb872-0a7d-4a68-b189-aeece143c7e4')
        print(metadata)
        
    """
    assert isinstance(name, str), "name must be a string."
    assert version_id is None or isinstance(version_id, str), "version must be an string."
    
    # Get metadata for the object in onyx engine
    metadata = handle_post_request("/get_object_metadata", {"object_name": name, "object_id": version_id})
    if metadata is None:
        return None
    if isinstance(metadata['config'], str):
        metadata['config'] = json.loads(metadata['config'])
            
    return metadata

def save_dataset(name: str, dataset: OnyxDataset, source_datasets: List[Dict[str, Optional[str]]]=[]):
    """
    Save a dataset to the Engine.
    
    Args:
        name (str): The name for the new dataset
        dataset (OnyxDataset): The OnyxDataset object to save
        source_datasets (List[Dict[str, Optional[str]]]): The source datasets used as a list of dictionaries, eg. [{'name': 'dataset_name', 'version_id': 'dataset_version'}]. If no version is provided, the latest version will be used.
        
    Example:
    
    .. code-block:: python

        # Load data
        raw_data = onyx.load_dataset('example_data')

        # Pull out features for model training
        train_data = pd.DataFrame()
        train_data['acceleration_predicted'] = raw_data.dataframe['acceleration']
        train_data['velocity'] = raw_data.dataframe['velocity']
        train_data['position'] = raw_data.dataframe['position']
        train_data['control_input'] = raw_data.dataframe['control_input']
        train_data = train_data.dropna()

        # Save training dataset
        train_dataset = OnyxDataset(
            dataframe=train_data,
            outputs=['acceleration_predicted'],
            inputs=['velocity', 'position', 'control_input'],
            dt=0.0025
        )
        onyx.save_dataset(name='example_train_data', dataset=train_dataset, source_datasets=[{'name': 'example_data'}])
    
    """
    assert isinstance(name, str), "name must be a string."
    assert isinstance(dataset, OnyxDataset), "dataset must be an OnyxDataset."
    sources = [SourceObject.model_validate({"name": source["name"], "id": source.get("version_id")}) for source in source_datasets]
    
    # Validate the dataset dataframe, name, and source datasets
    if dataset.dataframe.empty:
        raise SystemExit("Onyx Engine API error: Dataset dataframe is empty.")
    if name == '':
        raise SystemExit("Onyx Engine API error: Dataset name must be a non-empty string.")
    for source in sources:
        if get_object_metadata(source.name, source.id) is None:
            raise SystemExit(f"Onyx Engine API error: Source dataset [{source}] not found in the Engine.")
    
    # Save a local copy of the dataset
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    dataset_filename = name + '.csv'
    dataset.dataframe.to_csv(os.path.join(DATASETS_PATH, dataset_filename), index=False)
    
    # Upload the dataset and config to the cloud
    metadata = set_object_metadata(name, 'dataset', dataset.config.model_dump_json(), sources)
    upload_object(dataset_filename, 'dataset', metadata['id'])
    handle_post_request("/notify_dataset_uploaded", {"dataset_name": name, "dataset_id": metadata['id']})
    
    # Save local copy of metadata
    full_metadata = get_object_metadata(name, metadata['id'])
    with open(os.path.join(DATASETS_PATH, name + '.json'), 'w') as f:
        f.write(json.dumps(full_metadata, indent=2))
        
    print(f'Dataset [{name}] saved to the Engine.')

def load_dataset(name: str, version_id: str=None) -> OnyxDataset:
    """
    Load a dataset from the Engine, either from a local cached copy or by downloading from the Engine.
    
    Args:
        name (str): The name of the dataset to load.
        version_id (str, optional): The version id of the dataset to load, None = latest_version. (Default is None)
    
    Returns:
        OnyxDataset: The loaded dataset.
        
    Example:
    
    .. code-block:: python

        # Load the training dataset
        train_dataset = onyx.load_dataset('example_train_data')
        print(train_dataset.dataframe.head())
        
    """
    assert isinstance(name, str), "name must be a string."
    assert version_id is None or isinstance(version_id, str), "version_id must be an string."
    config_filename = name + '.json'
    dataset_filename = name + '.csv'
    dataset_path = os.path.join(DATASETS_PATH, dataset_filename)
    config_path = os.path.join(DATASETS_PATH, config_filename)

    # Get dataset metadata
    metadata = get_object_metadata(name, version_id)
    if metadata is None:
        raise SystemExit(f"Onyx Engine API error: Dataset [{name}: {version_id}] not found in the Engine.")

    def download_dataset():
        download_object(dataset_filename, 'dataset', version_id)
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
        if version_id is None and metadata['id'] != local_metadata['id']:
            download_dataset()
        elif version_id is not None and version_id != local_metadata['id']:
            download_dataset()

    # Load the dataset from local storage
    dataset_dataframe = pd.read_csv(dataset_path)
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())
    dataset_config = OnyxDatasetConfig.model_validate(config_json["config"])
    dataset = OnyxDataset(config=dataset_config, dataframe=dataset_dataframe)

    return dataset

def save_model(name: str, model: MLP | RNN | Transformer, source_datasets: List[Dict[str, Optional[str]]]=[]):
    """
    Save a model to the Engine. Generally you won't need to use this function as the Engine will save models it trains automatically.
    
    Args:
        name (str): The name for the new model.
        model (MLP | RNN | Transformer): The Onyx model to save.
        source_datasets (List[Dict[str, Optional[str]]]): The source datasets used as a list of dictionaries, eg. [{'name': 'dataset_name', 'version_id': 'dataset_version'}]. If no version is provided, the latest version will be used.
        
    Example:
    
    .. code-block:: python
    
        # Create model configuration
        outputs = [
            Output(name='acceleration_prediction'),
        ]
        inputs = [
            State(name='velocity', relation='derivative', parent='acceleration_prediction'),
            State(name='position', relation='derivative', parent='velocity'),
            Input(name='control_input'),
        ]
        mlp_config = MLPConfig(
            outputs=outputs,
            inputs=inputs,
            dt=0.0025,
            sequence_length=8,
            hidden_layers=3,
            hidden_size=64,
            activation='relu',
            dropout=0.2,
            bias=True
        )
         
        # Create and save model
        model = MLP(mlp_config)
        onyx.save_model(name='example_model', model=model, source_datasets=[{'name': 'example_train_data'}])
                   
    """
    assert isinstance(name, str), "name must be a string."
    assert isinstance(model, (MLP, RNN, Transformer)), "model must be an MLP, RNN, or Transformer model."
    sources = [SourceObject.model_validate({"name": source["name"], "id": source.get("version_id")}) for source in source_datasets]
    
    # Validate the model name and source datasets
    if name == '':
        raise SystemExit("Onyx Engine API error: Model name must be a non-empty string.")
    for source in sources:
        if get_object_metadata(source.name, source.id) is None:
            raise SystemExit(f"Onyx Engine API error: Source dataset [{source}] not found in the Engine.")
    
    # Save model to local storage
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    model_filename = name + '.pt'
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, model_filename))
    
    # Upload the model and config to the cloud
    metadata = set_object_metadata(name, 'model', model.config.model_dump_json(), sources)
    upload_object(model_filename, 'model', metadata['id'])
    
    # Save local copy of metadata
    full_metadata = get_object_metadata(name, metadata['id'])
    with open(os.path.join(MODELS_PATH, name + '.json'), 'w') as f:
        f.write(json.dumps(full_metadata, indent=2))
    
    print(f'Model [{name}] saved to the Engine.')

def load_model(name: str, version_id: str=None, mode: Literal["online", "offline"]="online") -> MLP | RNN | Transformer:
    """
    Load a model from the Engine, either from a local cached copy or by downloading from the Engine.
    
    Args:
        name (str): The name of the model to load.
        version_id (str, optional): The version of the model to load, None = latest_version. (Default is None)
        mode (Literal["online", "offline"]): Whether to use the Engine or restrict model loading to offline storage (Default is "online")
    
    Returns:
        MLP | RNN | Transformer: The loaded Onyx model.
        
    Example:
    
    .. code-block:: python
    
        # Load our model
        model = onyx.load_model('example_model')
        print(model.config)
        
        # Load a specific version of the model
        model = onyx.load_model('example_model', version_id='a05fb872-0a7d-4a68-b189-aeece143c7e4')
        print(model.config)
        
    """
    assert isinstance(name, str), "name must be a string."
    assert version_id is None or isinstance(version_id, str), "version_id must be an string."
    assert mode in ["online", "offline"], "mode must be either 'online' or 'offline'"
    
    model_filename = name + '.pt'
    config_filename = name + '.json'
    model_path = os.path.join(MODELS_PATH, model_filename)
    config_path = os.path.join(MODELS_PATH, config_filename)

    if mode == "offline":
        # In offline mode, try to directly load from local storage
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise SystemExit(f"Onyx Engine API error: Model [{name}] not found in local storage")
        
        with open(config_path, 'r') as f:
            config_json = json.load(f)
        model = model_from_config(config_json['config'])
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    # Online mode - check remote metadata
    metadata = get_object_metadata(name, version_id)
    if metadata is None:
        raise SystemExit(f"Onyx Engine API error: Model [{name}: {version_id}] not found in the Engine.")

    def download_model():
        download_object(model_filename, 'model', version_id)
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
        if version_id is None and metadata['id'] != local_metadata['id']:
            download_model()
        elif version_id is not None and version_id != local_metadata['id']:
            download_model()

    # Load the model from local storage
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())
    model = model_from_config(config_json['config'])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model

def train_model(
    model_name: str = "",
    model_config: Union[MLPConfig, RNNConfig, TransformerConfig] = None,
    dataset_name: str = "",
    dataset_version_id: Optional[str] = None,
    training_config: TrainingConfig = TrainingConfig(),
    monitor_training: bool = True,
):
    """
    Train a model on the Engine using a specified dataset, model config, and training config.
    
    Args:
        model_name (str): The name of the model to train. (Required)
        model_config (Union[MLPConfig, RNNConfig, TransformerConfig]): The configuration for the model to train. (Required)
        dataset_name (str): The name of the dataset to train on. (Required)
        dataset_version_id (str, optional): The version of the dataset to train on, None = latest_version. (Default is None)
        training_config (TrainingConfig): The configuration for the training process. (Default is TrainingConfig())
        monitor_training (bool, optional): Whether to monitor the training job. (Default is True)
        
    Example:
    
    .. code-block:: python
    
        # Model config
        outputs = [
            Output(name='acceleration_prediction'),
        ]
        inputs = [
            State(name='velocity', relation='derivative', parent='acceleration_prediction'),
            State(name='position', relation='derivative', parent='velocity'),
            Input(name='control_input'),
        ]

        model_config = MLPConfig(
            outputs=outputs,
            inputs=inputs,
            dt=0.0025,
            sequence_length=8,
            hidden_layers=3,
            hidden_size=64,
            activation='relu',
            dropout=0.2,
            bias=True
        )
        
        # Training config
        training_config = TrainingConfig(
            training_iters=2000,
            train_batch_size=32,
            test_dataset_size=500,
            checkpoint_type='single_step',
            optimizer=AdamWConfig(lr=3e-4, weight_decay=1e-2),
            lr_scheduler=CosineDecayWithWarmupConfig(max_lr=3e-4, min_lr=3e-5, warmup_iters=200, decay_iters=1000)
        )

        # Execute training
        onyx.train_model(
            model_name='example_model',
            model_config=model_config,
            dataset_name='example_train_data',
            training_config=training_config,
            monitor_training=True
        )
        
    """
    assert isinstance(model_name, str), "model_name must be a string."
    assert isinstance(model_config, (MLPConfig, RNNConfig, TransformerConfig)), "model_config must be a model config."
    assert isinstance(dataset_name, str), "dataset_name must be a string."
    assert dataset_version_id is None or isinstance(dataset_version_id, str), "dataset_version_id must be an string."
    assert isinstance(training_config, TrainingConfig), "training_config must be a TrainingConfig."
    assert isinstance(monitor_training, bool), "monitor_training must be a boolean."

    # Check that model/dataset names are not empty
    if model_name == '':
        raise SystemExit("Onyx Engine API error: Model name must be a non-empty string.")
    if dataset_name == '':
        raise SystemExit("Onyx Engine API error: Dataset name must be a non-empty string.")

    # Check that the dataset exists
    data_metadata = get_object_metadata(dataset_name, dataset_version_id)
    if data_metadata is None:
        raise SystemExit(f"Onyx Engine API error: Dataset [{dataset_name}: {dataset_version_id}] not found in the Engine.")
    data_config = OnyxDatasetConfig.model_validate(data_metadata['config'])
    # Check that config of model matches features of dataset
    if len(model_config.inputs) != len(data_config.inputs):
        raise SystemExit(f"Onyx Engine API error: Number of inputs in model config does not match dataset.")
    if len(model_config.outputs) != len(data_config.outputs):
        raise SystemExit(f"Onyx Engine API error: Number of outputs in model config does not match dataset.")
    # Check that config dt is an integer multiple of dataset dt
    if abs((model_config.dt % data_config.dt)) > 1e-9:
        raise SystemExit(f"Onyx Engine API error: Model config dt must be an integer multiple of dataset dt.")

    # Request the onyx server to train the model
    response = handle_post_request("/train_model", {
        "onyx_model_name": model_name,
        "onyx_model_config": model_config.model_dump_json(),
        "dataset_name": dataset_name,
        "dataset_id": dataset_version_id,
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
    dataset_name: str = "",
    dataset_version_id: Optional[str] = None,
    optimization_config: OptimizationConfig = None,
):
    """
    Optimize a model on the Engine using a specified dataset, model simulator config, and optimization configs. Optimization configs define the search space for hyperparameters.
    
    Args:
        model_name (str): The name of the model to optimize. (Required)
        model_sim_config (ModelSimulatorConfig): The configuration for the model simulator. (Required)
        dataset_name (str): The name of the dataset to optimize on. (Required)
        dataset_version_id (str, optional): The version of the dataset to optimize on, None = latest_version. (Default is None)
        optimization_config (OptimizationConfig): The configuration for the optimization process. (Required)

    Example:
    
    .. code-block:: python
    
        # Model inputs/outputs
        outputs = [
            Output(name='acceleration_prediction'),
        ]
        inputs = [
            State(name='velocity', relation='derivative', parent='acceleration_prediction'),
            State(name='position', relation='derivative', parent='velocity'),
            Input(name='control_input'),
        ]
        
        # Model optimization configs
        mlp_opt = MLPOptConfig(
            outputs=outputs,
            inputs=inputs,
            dt=0.0025,
            sequence_length={"select": [1, 2, 4, 5, 6, 8, 10]},
            hidden_layers={"range": [2, 4, 1]},
            hidden_size={"select": [12, 24, 32, 64, 128]},
            activation={"select": ['relu', 'tanh']},
            dropout={"range": [0.0, 0.4, 0.1]},
            bias=True
        )
        rnn_opt = RNNOptConfig(
            outputs=outputs,
            inputs=inputs,
            dt=0.0025,
            rnn_type={"select": ['RNN', 'LSTM', 'GRU']},
            sequence_length={"select": [1, 2, 4, 5, 6, 8, 10, 12, 14, 15]},
            hidden_layers={"range": [2, 4, 1]},
            hidden_size={"select": [12, 24, 32, 64, 128]},
            dropout={"range": [0.0, 0.4, 0.1]},
            bias=True
        )
        transformer_opt = TransformerOptConfig(
            outputs=outputs,
            inputs=inputs,
            dt=0.0025,
            sequence_length={"select": [1, 2, 4, 5, 6, 8, 10, 12, 14, 15]},
            n_layer={"range": [2, 4, 1]},
            n_head={"range": [2, 10, 2]},
            n_embd={"select": [12, 24, 32, 64, 128]},
            dropout={"range": [0.0, 0.4, 0.1]},
            bias=True
        )
            
        # Optimizer configs
        adamw_opt = AdamWOptConfig(
            lr={"select": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 5e-3, 1e-2]},
            weight_decay={"select": [1e-4, 1e-3, 1e-2, 1e-1]}
        )
        sgd_opt = SGDOptConfig(
            lr={"select": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 5e-3, 1e-2]},
            weight_decay={"select": [1e-4, 1e-3, 1e-2, 1e-1]},
            momentum={"select": [0, 0.8, 0.9, 0.95, 0.99]}
        )
        
        # Learning rate scheduler configs
        cos_decay_opt = CosineDecayWithWarmupOptConfig(
            max_lr={"select": [1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 3e-3, 5e-3]},
            min_lr={"select": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 8e-5, 1e-4]},
            warmup_iters={"select": [50, 100, 200, 400, 800]},
            decay_iters={"select": [500, 1000, 2000, 4000, 8000]}
        )
        cos_anneal_opt = CosineAnnealingWarmRestartsOptConfig(
            T_0={"select": [200, 500, 1000, 2000, 5000, 10000]},
            T_mult={"select": [1, 2, 3]},
            eta_min={"select": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 3e-4]}
        )
        
        # Optimization config
        opt_config = OptimizationConfig(
            training_iters=2000,
            train_batch_size=512,
            test_dataset_size=500,
            checkpoint_type='single_step',
            opt_models=[mlp_opt, rnn_opt, transformer_opt],
            opt_optimizers=[adamw_opt, sgd_opt],
            opt_lr_schedulers=[None, cos_decay_opt, cos_anneal_opt],
            num_trials=5
        )
        
        # Execute model optimization
        onyx.optimize_model(
            model_name='example_model_optimized',
            model_sim_config=sim_config,
            dataset_name='example_train_data',
            optimization_config=opt_config,
        )
        
    """
    assert isinstance(model_name, str), "model_name must be a string."
    assert isinstance(dataset_name, str), "dataset_name must be a string."
    assert dataset_version_id is None or isinstance(dataset_version_id, str), "dataset_version_id must be an string."
    assert isinstance(optimization_config, OptimizationConfig), "optimization_config is required and must be an OptimizationConfig."

    # Check that model/dataset names are not empty
    if model_name == '':
        raise SystemExit("Onyx Engine API error: Model name must be a non-empty string.")
    if dataset_name == '':
        raise SystemExit("Onyx Engine API error: Dataset name must be a non-empty string.")

    # Check that the dataset exists
    data_metadata = get_object_metadata(dataset_name, dataset_version_id)
    if data_metadata is None:
        raise SystemExit(f"Onyx Engine API error: Dataset [{dataset_name}: {dataset_version_id}] not found in the Engine.")
    data_config = OnyxDatasetConfig.model_validate(data_metadata['config'])
    # Check that config of model matches features of dataset
    for opt_model in optimization_config.opt_models:
        if len(opt_model.inputs) != len(data_config.inputs):
            raise SystemExit(f"Onyx Engine API error: Number of inputs in model config does not match dataset.")
        if len(opt_model.outputs) != len(data_config.outputs):
            raise SystemExit(f"Onyx Engine API error: Number of outputs in model config does not match dataset.")

    # Request the onyx server to train the model
    response = handle_post_request("/optimize_model", {
        "onyx_model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_id": dataset_version_id,
        "optimization_config": optimization_config.model_dump_json(),
    })

    print(f'Preparing to optimize model [{model_name}] using dataset [{dataset_name}].')
