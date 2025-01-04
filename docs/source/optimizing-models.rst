.. _optimizing-models:

Optimizing Models
=================

Optimizing models is similar in code to :ref:`training-models`. Now, however, we can let the Engine experiment with different training/model configurations to find the best model for our system.

Optimization Script
-------------------

We will optimize a model using the following code, you can paste this into your editor and then follow the explanation below:

.. code-block:: python

    from onyxengine.modeling import (
        ModelSimulatorConfig,
        State,
        MLPOptConfig,
        RNNOptConfig,
        TransformerOptConfig,
        AdamWOptConfig,
        SGDOptConfig,
        CosineDecayWithWarmupOptConfig,
        CosineAnnealingWarmRestartsOptConfig,
        OptimizationConfig
    )
    import onyxengine as onyx

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
        opt_lr_schedulers=[None, cos_decay_opt, cos_anneal_opt], # None means no scheduler, ie. constant learning rate
        num_trials=5
    )

    # Execute training
    onyx.optimize_model(
        model_name='brake_model_optimized',
        model_sim_config=sim_config,
        dataset_name='brake_train_data',
        optimization_config=opt_config,
    )

OptConfigs
----------

Every Onyx model, model weight optimizer, and learning rate scheduler has its own "OptConfig" which allows you to specify the search spaces for hyperparameters.

Search spaces can be defined with the following:

- **A value** - Constrain the Engine to one value for all optimization trials.

    .. code-block:: python

        mlp_opt = MLPOptConfig(
            hidden_layers=2,
            hidden_size=32,
            activation='relu',
            dropout=0.2,
            bias=True
        )

- **"select"** - A selection of values the Engine can choose from for each trial.

    .. code-block:: python

        mlp_opt = MLPOptConfig(
            hidden_layers={"select": [2, 3, 4]},
            hidden_size={"select": [12, 24, 32, 64, 128]},
            activation={"select": ['relu', 'tanh']},
            dropout={"select": [0.0, 0.1, 0.2, 0.3, 0.4]}
            bias={"select": [True, False]}
        )

- **"range"** - A range of values the Engine can choose from for each trial. Ranges must be defined as a list of three values: [start, end, step].

    .. code-block:: python

        mlp_opt = MLPOptConfig(
            hidden_layers={"range": [2, 4, 1]},
            hidden_size={"range": [12, 128, 16]},
            activation='relu', # Ranges not supported for strings
            dropout={"range": [0.0, 0.4, 0.1]}
            bias=True, # Ranges not supported for booleans
        )

- **All of the above** - You can mix these methods to define the search space for each hyperparameter.

    .. code-block:: python

        mlp_opt = MLPOptConfig(
            hidden_layers={"range": [2, 5, 1]},
            hidden_size={"select": [12, 24, 32, 64, 128]},
            activation={"select": ['relu', 'tanh']},
            dropout={"range": [0.0, 0.4, 0.1]},
            bias=True
        )

Once you've defined your OptConfigs, you can pass them to the complete OptimizationConfig

.. code-block:: python

    # Optimization config
    opt_config = OptimizationConfig(
        training_iters=2000,
        train_batch_size=512,
        test_dataset_size=500,
        checkpoint_type='single_step',
        opt_models=[mlp_opt, rnn_opt, transformer_opt],
        opt_optimizers=[adamw_opt, sgd_opt],
        opt_lr_schedulers=[None, cos_decay_opt, cos_anneal_opt], # None means no scheduler, ie. constant learning rate
        num_trials=5
    )

The `TrainingConfig` parameters like **training_iters**, **checkpoint_type**, **etc.** define the training process for each trial.

The **num_trials** parameter specifies how many trials the Engine will run.

In general, the default values our team has selected for OptConfigs tend to be good starting points for model optimization.

optimize_model()
----------------

We are now ready to optimize our model in the Engine:

.. code-block:: python

    # Execute model optimization
    onyx.optimize_model(
        model_name='brake_model_optimized',
        model_sim_config=sim_config,
        dataset_name='brake_train_data',
        optimization_config=opt_config,
    )

You can monitor the optimization process via the Engine Platform. The different trials of the model optimization will be stored as separate versions of the model in the Engine. To download a specific model version, you can use the **version** parameter of the :meth:`onyx.load_model` function.

.. code-block:: python

    onyx.load_model('brake_model_optimized', version=1)

Now that we have optimized our model, we can deploy it for simulation in :ref:`simulating-models`.