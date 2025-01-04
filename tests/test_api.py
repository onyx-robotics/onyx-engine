import torch
import pandas as pd
import onyxengine as onyx
from onyxengine.data import OnyxDataset
from onyxengine.modeling import (
    ModelSimulatorConfig, 
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
    data = onyx.get_object_metadata('brake_model_test_test')
    print(data)

def test_data_download():
    # Load the training dataset
    train_dataset = onyx.load_dataset('brake_train_data')
    print(train_dataset.dataframe.head())

def test_data_upload():
    # Load data
    raw_data = onyx.load_dataset('brake_data')

    # Pull out features for model training
    train_data = pd.DataFrame()
    train_data['acceleration_predicted'] = raw_data.dataframe['acceleration']
    train_data['velocity'] = raw_data.dataframe['velocity']
    train_data['position'] = raw_data.dataframe['position']
    train_data['brake_input'] = raw_data.dataframe['brake_input']
    train_data = train_data.dropna()

    # Save training dataset
    train_dataset = OnyxDataset(
        features=train_data.columns,
        dataframe=train_data,
        num_outputs=1,
        num_state=2,
        num_control=1,
        dt=0.0025
    )
    onyx.save_dataset(name='brake_train_data', dataset=train_dataset, source_datasets=[{'name': 'brake_data'}])
    
def test_model_upload():
    sim_config = ModelSimulatorConfig(
        outputs=['acceleration'],
        states=[
            State(name='velocity', relation='derivative', parent='acceleration'),
            State(name='position', relation='derivative', parent='velocity'),
        ],
        controls=['brake_input'],
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
    
    model = MLP(mlp_config)
    onyx.save_model(name='brake_model_test', model=model, source_datasets=[{'name': 'brake_train_data'}])
    
def test_model_download():
    model = onyx.load_model('brake_model_test_test')
    print(model.config)
    
def test_train_model():
    # Model config
    sim_config = ModelSimulatorConfig(
        outputs=['acceleration'],
        states=[
            State(name='velocity', relation='derivative', parent='acceleration'),
            State(name='position', relation='derivative', parent='velocity'),
        ],
        controls=['brake_input'],
        dt=0.0025
    )
    model_config = MLPConfig(
        sim_config=sim_config,
        num_inputs=sim_config.num_inputs,
        num_outputs=sim_config.num_outputs,
        hidden_layers=2,
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
        model_name='brake_model_test',
        model_config=model_config,
        dataset_name='brake_train_data',
        training_config=training_config,
        monitor_training=False
    )
    
def test_optimize_model():
    # Model sim config (used across all trials)
    sim_config = ModelSimulatorConfig(
        outputs=['acceleration'],
        states=[
            State(name='velocity', relation='derivative', parent='acceleration'),
            State(name='position', relation='derivative', parent='velocity'),
        ],
        controls=['brake_input'],
        dt=0.0025
    )
    
    # Model optimization configs
    mlp_opt = MLPOptConfig(
        sim_config=sim_config,
        num_inputs=sim_config.num_inputs,
        num_outputs=sim_config.num_outputs,
        sequence_length={"select": [1, 2, 4, 5, 6, 8, 10]},
        hidden_layers={"range": [2, 4, 1]},
        hidden_size={"select": [12, 24, 32, 64, 128]},
        activation={"select": ['relu', 'tanh']},
        dropout={"range": [0.0, 0.4, 0.1]},
        bias=True
    )
    rnn_opt = RNNOptConfig(
        sim_config=sim_config,
        num_inputs=sim_config.num_inputs,
        num_outputs=sim_config.num_outputs,
        rnn_type={"select": ['RNN', 'LSTM', 'GRU']},
        sequence_length={"select": [1, 2, 4, 5, 6, 8, 10, 12, 14, 15]},
        hidden_layers={"range": [2, 4, 1]},
        hidden_size={"select": [12, 24, 32, 64, 128]},
        dropout={"range": [0.0, 0.4, 0.1]},
        bias=True
    )
    transformer_opt = TransformerOptConfig(
        sim_config=sim_config,
        num_inputs=sim_config.num_inputs,
        num_outputs=sim_config.num_outputs,
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
        model_name='brake_model_optimized',
        model_sim_config=sim_config,
        dataset_name='brake_train_data',
        optimization_config=opt_config,
    )
    
def test_use_model():    
    # Load our model
    model = onyx.load_model('brake_model_test')
    num_inputs = model.config.sim_config.num_inputs
    num_states = model.config.sim_config.num_states
    num_controls = model.config.sim_config.num_controls
    seq_length = model.config.sequence_length

    # Run inference with our model
    batch_size = 1
    test_input = torch.ones(batch_size, seq_length, num_inputs)
    with torch.no_grad():
        test_output = model(test_input)
    print(test_output)
    
    # Simulate a trajectory with our model
    # Model will fill in the x_traj tensor with the simulated trajectory
    sim_steps = 10
    x0 = torch.ones(batch_size, seq_length, num_states)
    u = torch.ones(batch_size, sim_steps, num_controls)
    x_traj = torch.zeros(1, sim_steps, num_inputs)
    model.simulate(x_traj, x0, u)
    print(x_traj)
    
if __name__ == '__main__':
    test_metadata_get()
    # test_data_download()
    # test_data_upload()
    # test_model_upload()
    # test_model_download()
    # test_train_model()
    # test_optimize_model()
    # test_use_model()
