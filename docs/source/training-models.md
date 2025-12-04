(training-models)=
# Training Models

To train a model in the Engine, we just need to configure the training run.

## Training Script

We will train our first model on the dataset "example_train_data" that is provided for you in the Engine. We'll use the following training code, you can paste this into your editor and then follow the explanation of each part below:

```python
from onyxengine.modeling import (
    Output,
    Input,
    MLPConfig,
    TrainingConfig,
    AdamWConfig,
    CosineDecayWithWarmupConfig
)
import onyxengine as onyx

# Model config
outputs = [
    Output(name='acceleration_prediction'),
]
inputs = [
    Input(name='velocity', parent='acceleration_prediction', relation='derivative'),
    Input(name='position', parent='velocity', relation='derivative'),
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
    train_batch_size=1024,
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
```

## Model Configuration

The first thing you'll notice is the definition of our model's outputs/inputs. 

By default, AI models typically predict one step at a time. But for simulation and controls, we need to predict more than a single step and instead simulate trajectories over multiple steps. Managing state relations, numerical integration, and recursive model calls is tedious, so our models provide a built-in `simulate()` function to handle this for you efficiently.

All you need to do is define the model's outputs, inputs, and states:

```python
outputs = [
    Output(name='acceleration_prediction'),
]
inputs = [
    Input(name='velocity', parent='acceleration_prediction', relation='derivative'),
    Input(name='position', parent='velocity', relation='derivative'),
    Input(name='control_input'),
]
```

States are similar to dynamical system state-space models, where $\dot{x} = f(x, u)$. In this example:

- Model Outputs
    - **acceleration_predicted** ($\dot{x}$) is the output of the model.
- Model Inputs
    - **velocity** ($x_1$) is the first state, whose parent **acceleration** is its derivative.
    - **position** ($x_2$) is the second state, whose parent **velocity** is its derivative.
    - **control_input** ($u$) is a control input, whose value is known at each time step.

We will show how to simulate models in {ref}`simulating-models`, but for training we just need to provide the configuration.

Now, we can pass the outputs and inputs to any Onyx model architecture. We'll use a simple Multi-Layer Perceptron (MLP) model:

```python
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
```

And that's it! The model is ready to be trained.

## Training Configuration

The training configuration specifies how the model will be trained:

```python
training_config = TrainingConfig(
    training_iters=2000,
    train_batch_size=1024,
    test_dataset_size=500,
    checkpoint_type='single_step',
    optimizer=AdamWConfig(lr=3e-4, weight_decay=1e-2),
    lr_scheduler=CosineDecayWithWarmupConfig(max_lr=3e-4, min_lr=3e-5, warmup_iters=200, decay_iters=1000)
)
```

This training is set to run for 2000 iterations, where each iteration processes a batch of 32 data samples.

During training, the dataset will be split into three parts:

- **Train**: The bulk of the dataset, used to train the model's weights.
- **Validation**: A smaller split (~10% of the data), used to validate that we are not overfitting to the training data.
- **Test**: A final set of data that is used to visualize the model's predictions in the Engine.

The **checkpoint_type** specifies whether the Engine should train the model for the best single-step or multi-step prediction.

- **single_step**: Saves best model weights for predicting a **single step** into the future, this is the conventional AI model evaluation. Even if your goal is multi-step simulation, single-step checkpointing is useful for seeing how "modelable" a dataset is, or what dynamics the model is capturing.
- **multi_step**: Saves best model weights for predicting **multiple steps** into the future (ie. simulating trajectories). This is the metric is often what we care most about for simulation and control.

## train_model()

Now that we have the model and training configurations, we can train the model:

```python
onyx.train_model(
    model_name='example_model',
    model_config=model_config,
    dataset_name='example_train_data',
    training_config=training_config,
    monitor_training=True
)
```

This will initiate training in the Engine. The progress of the training will be displayed in the python console (exiting the console will not stop the training).

We recommend monitoring the training process via the Engine Platform, where more details/visualizations are available to help you get the best training results.

Congratulations! You've trained your first model in the Engine. Here are some quick things to try with this script:

- Our model used the current value of the inputs to make predictions, but often times hardware AI models benefit from using a history sequence of inputs. To increase the sequence length, we can simply change the **sequence_length** parameter of Onyx models and re-run the training script:

    ```python
    model_config = MLPConfig(
        outputs=outputs,
        inputs=inputs,
        dt=0.0025,
        sequence_length=12, # Increased sequence length to 12
        hidden_layers=3,
        hidden_size=64,
        activation='relu',
        dropout=0.2,
        bias=True
    )
    ```

- If you want to see how easy it is to swap model architectures, try replacing the model_config with a **Transformer** model:

    ```python
    from onyxengine.modeling import TransformerConfig

    model_config = TransformerConfig(
        outputs=outputs,
        inputs=inputs,
        dt=0.0025,
        sequence_length=10,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.2,
        bias=True
    )
    ```

- Or, if you want to let the Engine optimize a model for you, check out {ref}`optimizing-models`.

