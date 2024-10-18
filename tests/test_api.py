import torch
import pandas as pd
import onyxengine as onyx
from onyxengine.data import OnyxDataset
from onyxengine.modeling import ModelSimulatorConfig, State, MLPConfig, MLP, TrainingConfig

def test_metadata_get():
    data = onyx.get_object_metadata('brake_train_data')
    print(data)
    #print(data['data'][0]['object_config'])

def test_data_download():
    # Load the training dataset
    train_dataset = onyx.load_dataset('brake_train_data')
    print(train_dataset.dataframe.head())

def test_data_upload():
    # Load data
    raw_data = onyx.load_dataset('brake_train_data')

    # Pull out features for model training
    train_data = pd.DataFrame()
    train_data['acceleration_predicted'] = raw_data.dataframe['acceleration_predicted']
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
    onyx.save_dataset("brake_train_data_test", dataset=train_dataset, source_dataset_ids=['brake_train_data'])
    
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
    onyx.save_model("vehicle_brake_model", model, mlp_config, source_dataset_ids=['brake_train_data_test'])
    
def test_model_download():
    model = onyx.load_model('vehicle_brake_model')
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
        hidden_size=32,
        activation='relu',
        dropout=0.2,
        bias=True
    )
    
    # Training config
    training_config = TrainingConfig(
        training_iters=3000,
    )

    # Execute training
    onyx.train_model(
        dataset_id='vehicle_braking_data',
        model_id='brake_model_test',
        model_config=model_config,
        training_config=training_config,
    )
    
def test_use_model():    
    # Load our model
    model = onyx.load_model('brake_model_test')

    # Run inference with our model
    test_input = torch.ones(1, 1, 3)
    with torch.no_grad():
        test_output = model(test_input)
    print(test_output)
    
if __name__ == '__main__':
    test_metadata_get()
    #test_data_download()
    #test_data_upload()
    #test_model_upload()
    #test_model_download()
    #test_train_model()
    #test_use_model()