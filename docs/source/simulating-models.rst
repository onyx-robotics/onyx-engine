.. _simulating-models:

Simulating Models
=================

Onyx models come with built-in simulation capabilites to quickly deploy trained models.

Model Simulator
---------------

To recap from :ref:`training-models`, AI models predict one step at a time by default. But for simulation and controls, we need to predict more than a single step and instead simulate trajectories over multiple steps. Managing state relations, numerical integration, and recursive model calls is tedious, so we provide a ModelSimulator to handle this for you efficiently.

All Onyx models inherit from a `ModelSimulator` class that adds a :meth:`simulate` method to the model. To use this method, we just need to configure the model's simulator. 

Once the model is trained/saved, the simulator config will be saved with it, so you will not need to set the simulator when loading the model.

.. code-block:: python

    from onyxengine.modeling import (
        ModelSimulatorConfig,
        State
    )
    import onyxengine as onyx

    # Configure a model simulator for a new model
    sim_config = ModelSimulatorConfig(
        outputs=['acceleration'],
        states=[
            State(name='velocity', relation='derivative', parent='acceleration'),
            State(name='position', relation='derivative', parent='velocity'),
        ],
        controls=['control_input'],
        dt=0.0025
    )

Sim configs are similar to dynamical system state-space models, where :math:`\dot{x} = f(x, u)`. In this example:

- Model Outputs
    - **acceleration** (:math:`\dot{x}`) is the output of the model.
- Model Inputs
    - **velocity** (:math:`x_1`) is the first state, whose parent **acceleration** is its derivative.
    - **position** (:math:`x_2`) is the second state, whose parent **velocity** is its derivative.
    - **control_input** (:math:`u`) is a control input and provided at each time step.

The control inputs are given at each time step, and outputs are predicted by the model, so they do not need any special attention.

The **states**, however, must be calculated from the outputs (or other states) and fed back into the model as inputs to rollout a trajectory. The available relations between states and their parents are:

- **"output"**: The state is equal to an output of the model.
    
    - :math:`\text{state}_{t+1} = \text{parent}_t`


- **"delta"**: The parent is the delta of the state.

    - :math:`\text{state}_{t+1} = \text{state}_t + \text{parent}_t`

- **"derivative"**: The parent is the derivative of the state.

    - :math:`\text{state}_{t+1} = \text{state}_t + \text{parent}_t \cdot \text{dt}`

.. Note::
    
    Often it can be easier for a hardware AI model to predict the delta or derivative of a feature than the feature itself.

Lastly, the **dt** parameter is the model's time step, this parameter is used both in simulation and training.

simulate()
----------

We will now use the :meth:`simulate` method on a model. Here is the example code:

.. code-block:: python

    import torch
    import onyxengine as onyx

    # Load our model
    model = onyx.load_model('example_model')
    num_states = model.config.sim_config.num_states
    num_controls = model.config.sim_config.num_controls
    num_inputs = model.config.sim_config.num_inputs
    seq_length = model.config.sequence_length

    # Run inference with our model (using normal pytorch model prediction)
    batch_size = 1
    test_input = torch.ones(batch_size, seq_length, num_inputs)
    with torch.no_grad():
        test_output = model(test_input)
    print(test_output)

    # Simulate a trajectory with our model
    # Model will fill in the traj_solution tensor with the simulated trajectory
    sim_steps = 10
    x0 = torch.ones(batch_size, seq_length, num_states)
    u = torch.ones(batch_size, sim_steps, num_controls)
    traj_solution = torch.zeros(1, sim_steps, num_inputs)
    model.simulate(traj_solution, x0, u)
    print(traj_solution)

To make the :meth:`simulate` method performant, we avoid using dynamic memory allocation by passing in the pre-allocated tensors needed:

.. figure:: _static/simulate_traj_solution.png
    :alt: Simulated trajectory solution
    :align: center
    :width: 40%

.. raw:: html

    <br><br>

- **traj_solution** - The solution tensor whose states will be "filled in" with the simulated trajectory. 

    - Should be of shape (batch_size, sim_steps, num_inputs):
    
        - **batch_size** is the number of parallel trajectories to simulate
        - **sim_steps** is the number of time steps to simulate
        - **num_inputs** is the number of inputs to the model (num_states + num_controls).

- **x0** - The initial state for the simulation, and the first `sequence_length` state values of the traj_solution tensor. 

    - Should be of shape (batch_size, seq_length, num_states)

- **u** - The control inputs for the simulation, and the complete array of control values for the traj_solution. 

    - Should be of shape (batch_size, sim_steps, num_controls)

- **(Optional) output_traj** - Optionally, you can pass a tensor to store the model outputs 

    - Should be of shape (batch_size, sim_steps, num_outputs)

The batch dimension allows for parallel simulation of multiple trajectories, which is where GPU acceleration becomes useful.

If you need to run some code (such as a controller) at each time step, you can just use the simulate method in a loop and simulate one step at a time.
