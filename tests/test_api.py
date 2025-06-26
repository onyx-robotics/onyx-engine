import torch
import pandas as pd
import onyxengine as onyx
from onyxengine.data import OnyxDataset
from onyxengine.modeling import ( 
    Output,
    Input,
    State,
    MLPConfig,
    MLP,
    RNNConfig,
    TransformerConfig,
    TrainingConfig, 
    OptimizationConfig,
    AdamWConfig,
    SGDConfig,
    CosineDecayWithWarmupConfig,
    CosineAnnealingWarmRestartsConfig,
    MLPOptConfig,
    RNNOptConfig,
    TransformerOptConfig,
    AdamWOptConfig,
    SGDOptConfig,
    CosineDecayWithWarmupOptConfig,
    CosineAnnealingWarmRestartsOptConfig
)

def test_metadata_get():
    # Get metadata for an Onyx object (dataset, model)
    metadata = onyx.get_object_metadata('raw_data')
    print(metadata)
    
    # Get metadata for a specific version
    metadata = onyx.get_object_metadata('raw_data', version_id='dcfec841-1748-47e2-b6c7-3c821cc69b4a')
    print(metadata)

def test_data_download():
    # Load the training dataset
    train_dataset = onyx.load_dataset('raw_data')
    print(train_dataset.dataframe.head())

def test_data_upload():
    # Load data
    # raw_data = onyx.load_dataset('brake_data')
    raw_data = pd.read_csv('onyx/datasets/brake_data.csv')

    # Pull out features for model training
    train_data = pd.DataFrame()
    train_data['acceleration'] = raw_data['acceleration']
    train_data['velocity'] = raw_data['velocity']
    train_data['position'] = raw_data['position']
    train_data['brake_input'] = raw_data['brake_input']
    train_data = train_data.dropna()

    # Save training dataset
    train_dataset = OnyxDataset(
        dataframe=train_data,
        outputs=['acceleration'],
        inputs=['velocity', 'position', 'brake_input'],
        dt=0.0025
    )
    onyx.save_dataset(name='raw_data', dataset=train_dataset)#, source_datasets=[{'name': 'brake_data'}])

def test_model_upload():
    # Create model configuration
    outputs = [
        Output(name='acceleration_prediction'),
    ]
    inputs = [
        State(name='velocity', relation='derivative', parent='acceleration_prediction'),
        State(name='position', relation='derivative', parent='velocity'),
        Input(name='brake_input'),
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
    
    model = MLP(mlp_config)
    onyx.save_model(name='small_embedded_model5', model=model, source_datasets=[{'name': 'training_data'}])

def test_model_download():
    model = onyx.load_model('small_embedded_model')
    print(model.config)

def test_train_model():
    # Model config
    outputs = [
        Output(name='acceleration_predicted', scale='mean'),
    ]
    inputs = [
        State(name='velocity', relation='derivative', parent='acceleration_predicted', scale='mean'),
        State(name='position', relation='derivative', parent='velocity', scale='mean'),
        Input(name='brake_input', scale='mean'),
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
        training_iters=3000,
        train_batch_size=1024,
        test_dataset_size=500,
        checkpoint_type='multi_step',
        optimizer=AdamWConfig(lr=3e-4, weight_decay=1e-2),
        # lr_scheduler=CosineDecayWithWarmupConfig(max_lr=1e-3, min_lr=3e-5, warmup_iters=200, decay_iters=1000)
    )

    # Execute training
    onyx.train_model(
        model_name='small_embedded_model',
        model_config=model_config,
        dataset_name='training_data',
        training_config=training_config,
        monitor_training=False
    )

def test_optimize_model():
    # Model inputs/outputs
    outputs = [
        Output(name='acceleration_predicted', scale='mean'),
    ]
    inputs = [
        State(name='velocity', relation='derivative', parent='acceleration_predicted', scale='mean'),
        State(name='position', relation='derivative', parent='velocity', scale='mean'),
        Input(name='brake_input', scale='mean'),
    ]
    
    # Model optimization configs
    mlp_opt = MLPOptConfig(
        outputs=outputs,
        inputs=inputs,
        dt=0.0025,
        sequence_length={"select": [1, 2, 4, 5, 6, 8, 10]},
        hidden_layers={"range": [2, 5, 1]},
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
        training_iters=3000,
        train_batch_size=1024,
        test_dataset_size=500,
        checkpoint_type='multi_step',
        opt_models=[mlp_opt],
        opt_optimizers=[adamw_opt],
        opt_lr_schedulers=[None],
        num_trials=20
    )
    
    # Execute model optimization
    onyx.optimize_model(
        model_name='small_embedded_model',
        dataset_name='training_data',
        optimization_config=opt_config,
    )

def test_use_model():    
    # Load our model
    model = onyx.load_model('small_embedded_model')
    model.eval()
    total_inputs = len(model.config.inputs)
    num_states = len([s for s in model.config.inputs if isinstance(s, State)])
    num_inputs = total_inputs - num_states

    # Example 1: Run inference with our model (using normal pytorch model prediction)
    batch_size = 1
    seq_length = model.config.sequence_length
    test_input = torch.ones(batch_size, seq_length, total_inputs)
    with torch.no_grad():
        test_output = model(test_input)
    print(test_output)
    
    # Example 2: Simulate a trajectory with our model
    # Model will fill in the x_traj tensor with the simulated trajectory
    sim_steps = 10
    state_x0 = torch.ones(batch_size, seq_length, num_states)
    inputs = torch.ones(batch_size, seq_length+sim_steps, num_inputs)
    state_traj = torch.zeros(1, seq_length+sim_steps, total_inputs)
    model.simulate(state_traj, state_x0, inputs)
    print(state_traj)

if __name__ == '__main__':
    test_metadata_get()
    # test_data_download()
    # test_data_upload()
    # test_model_upload()
    # test_model_download()
    # test_train_model()
    # test_optimize_model()
    # test_use_model()