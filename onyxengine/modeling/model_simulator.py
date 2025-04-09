import torch
from typing import List, Literal, Union, Tuple
from contextlib import nullcontext
from .model_features import Input, Output, State

class ModelSimulator():
    def __init__(self, outputs: List[Output], inputs: List[Union[Input, State]], sequence_length: int, dt: float):
        # Separate states by dependency then reorder to ensure parents are computed first
        output_dep_states, state_dep_states = self._separate_states(outputs, inputs)
        self.output_dep_states = self._resolve_state_order(output_dep_states)
        self.state_dep_states = self._resolve_state_order(state_dep_states)
        self.sequence_length = sequence_length
        self.dt = dt
        self.amp_context = nullcontext()
        self.n_state = len([state for state in inputs if isinstance(state, State)])
        self.n_inputs = len(inputs) - self.n_state
    
    def simulator_mixed_precision(self, setting: bool):
        if setting == False:
            self.amp_context = nullcontext()
            return
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        torch.backends.cuda.matmul.allow_tf32 = True # Allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # Allow tf32 on cudnn
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.amp_context = torch.amp.autocast(device_type=device, dtype=amp_dtype)
    
    def _separate_states(self, outputs: List[Output], inputs: List[Input]):
        # Separate states by output dependent and state dependent
        output_names = [output.name for output in outputs]
        input_names = [input.name for input in inputs]
        states = [(idx, state) for idx, state in enumerate(inputs) if isinstance(state, State)]
        output_dep_states = []
        state_dep_states = []
        for idx, state in states:
            if state.parent in output_names:
                output_dep_states.append((state, idx, output_names.index(state.parent)))
            else:
                state_dep_states.append((state, idx, input_names.index(state.parent)))
                
        return output_dep_states, state_dep_states
    
    def _resolve_state_order(self, state_tuple_list) -> List[Tuple[State, int, int]]:
        # Sort states based on parent-child dependencies
        state_map = {state.name: state for state, _, _ in state_tuple_list}        
        def sort_key(state_tuple):
            state, idx, parent_idx = state_tuple
            # Output states get highest priority, then resolve parent-child dependencies
            priority = 0 if state.relation == 'output' else 1
            parent_depth = 0
            parent = state.parent
            while parent:
                parent_depth += 1
                parent = state_map[parent].parent if parent in state_map else None
            
            return (priority, parent_depth)

        return sorted(state_tuple_list, key=sort_key)
        
    def _step(self, x, model_pred=None):
        # Do a single forward step of the model and update the states
        dx = self.forward(x[:, :-1, :])
        if model_pred is not None:
            model_pred = dx
            
        # Update states with dependencies on model output dx
        for state, idx, parent_idx in self.output_dep_states:
            if state.relation == 'output':
                x[:, -1, idx] = dx[:, parent_idx]
            elif state.relation == 'delta':
                x[:, -1, idx] = x[:, -2, idx] + dx[:, parent_idx]
            elif state.relation == 'derivative':
                x[:, -1, idx] = x[:, -2, idx] + dx[:, parent_idx]*self.dt
            
        # Update state with dependencies on state x
        for state, idx, parent_idx in self.state_dep_states:
            if state.relation == 'delta':
                x[:, -1, idx] = x[:, -2, idx] + x[:, -1, parent_idx]
            elif state.relation == 'derivative':
                x[:, -1, idx] = x[:, -2, idx] + x[:, -1, parent_idx]*self.dt

    def simulate(self, state_traj, state_x0, inputs=None, model_outputs=None):
        # Fills in the values of the state variables in state_traj,
        # state_traj (batch_size, sequence_length+sim_steps, num_states+num_inputs)
        # state_x0 - initial state (batch_size, sequence_length, num_states)
        # inputs - non-state inputs (batch_size, sim_steps+sequence_length, num_inputs)
        # model_outputs - model outputs to fill in too (batch_size, sim_steps, num_outputs))
        
        # Initialize simulation data
        seq_length = self.sequence_length
        sim_steps = state_traj.size(1) - seq_length
        state_traj[:, :seq_length, :self.n_state] = state_x0
        if inputs is not None:
            state_traj[:, :, -self.n_inputs:] = inputs

        with self.amp_context and torch.no_grad():
            for i in range(sim_steps):
                # Pass in the trajectory up to the t+1 step
                if model_outputs is not None:
                    self._step(state_traj[:, i:i+seq_length+1, :], model_outputs[:, i, :])
                else:
                    self._step(state_traj[:, i:i+seq_length+1, :])

class NumpyModelSimulator():
    def __init__(self, model, dt=None, ode_solver: Literal['euler', 'rk4']='euler'):
        self.model = model
        self.dt = dt
        self.ode_solver = ode_solver
    
    def simulate(self, x, u):
        if self.ode_solver == 'euler':
            x_next = self.euler_step(x, u, self.dt, self.model)
        elif self.ode_solver == 'rk4':
            x_next = self.rk4_step(x, u, self.dt, self.model)
        return x_next

    def euler_step(x, u, dt, model):
        x_next = x + model(x, u) * dt
        return x_next

    def rk4_step(x, u, dt, model):
        k1 = model(x, u)    
        k2 = model(x + dt / 2 * k1, u)
        k3 = model(x + dt / 2 * k2, u)
        k4 = model(x + dt * k3, u)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next