import torch
from typing import List, Literal
from pydantic import BaseModel
from contextlib import nullcontext

class State(BaseModel):
    """
    State variable used in ModelSimulatorConfig.
    
    Args:
        name (str): Variable name.
        relation (Literal['output', 'delta', 'derivative']): Method to solve for the variable: variable is an output of the model, parent is the delta of the variable, or parent is the derivative of the variable.
        parent (str): Parent variable to derive from.
    """
    
    name: str
    relation: Literal['output', 'delta', 'derivative']
    parent: str
    
class ModelSimulatorConfig(BaseModel):
    """
    Configuration class for the model simulator.
    
    Args:
        outputs (List[str]): List of output variables.
        states (List[State]): List of state variables.
        controls (List[str]): List of control variables.
        dt (float): Time step for simulation.
    """
    
    outputs: List[str] = []
    states: List[State] = []
    controls: List[str] = []
    dt: float = 0
    
    @property
    def num_outputs(self):
        return len(self.outputs)
    @property
    def num_states(self):
        return len(self.states)
    @property
    def num_controls(self):
        return len(self.controls)
    @property
    def num_inputs(self):
        return self.num_states + self.num_controls

class ModelSimulator():
    def __init__(self, sim_config: ModelSimulatorConfig):
        # Separate variables by dependency the reorder to ensure parents are computed first
        output_dep_vars, state_dep_vars = self._separate_variables(sim_config)
        self.output_dep_vars = self._resolve_variable_order(output_dep_vars)
        self.state_dep_vars = self._resolve_variable_order(state_dep_vars)
        self.dt = sim_config.dt
        self.amp_context = nullcontext()
        self.n_state = len(sim_config.states)
        self.n_control = len(sim_config.controls)
    
    def simulator_mixed_precision(self, setting: bool):
        if setting == False:
            self.amp_context = nullcontext()
            return
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        torch.backends.cuda.matmul.allow_tf32 = True # Allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # Allow tf32 on cudnn
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.amp_context = torch.amp.autocast(device_type=device, dtype=amp_dtype)
    
    def _separate_variables(self, sim_config: ModelSimulatorConfig):
        # Separate variables by output dependent and state dependent
        output_names = sim_config.outputs
        state_names = [state.name for state in sim_config.states]
        output_dep_vars = []
        state_dep_vars = []
        for idx, var in enumerate(sim_config.states):
            if var.parent in output_names:
                output_dep_vars.append((var, idx, output_names.index(var.parent)))
            else:
                state_dep_vars.append((var, idx, state_names.index(var.parent)))
                
        return output_dep_vars, state_dep_vars
    
    def _resolve_variable_order(self, var_tuple_list):
        # Sort variables based on parent-child dependencies
        var_map = {var.name: var for var, _, _ in var_tuple_list}        
        def sort_key(sim_variable_tuple):
            var, idx, parent_idx = sim_variable_tuple
            # Output variables get highest priority, then resolve parent-child dependencies
            priority = 0 if var.relation == 'output' else 1
            parent_depth = 0
            parent = var.parent
            while parent:
                parent_depth += 1
                parent = var_map[parent].parent if parent in var_map else None
            
            return (priority, parent_depth)

        return sorted(var_tuple_list, key=sort_key)
        
    def _step(self, x, output_next=None):
        # Do a single forward step of the model and update the state variables
        dx = self.forward(x[:, :-1, :])
        if output_next is not None:
            output_next = dx
            
        # Update state variables depending on model output dx
        for var, var_idx, parent_idx in self.output_dep_vars:
            if var.relation == 'output':
                x[:, -1, var_idx] = dx[:, parent_idx]
            elif var.relation == 'delta':
                x[:, -1, var_idx] = x[:, -2, var_idx] + dx[:, parent_idx]
            elif var.relation == 'derivative':
                x[:, -1, var_idx] = x[:, -2, var_idx] + dx[:, parent_idx]*self.dt
            
        # Update state variables depending on state x
        for var, var_idx, parent_idx in self.state_dep_vars:
            if var.relation == 'delta':
                x[:, -1, var_idx] = x[:, -2, var_idx] + x[:, -1, parent_idx]
            elif var.relation == 'derivative':
                x[:, -1, var_idx] = x[:, -2, var_idx] + x[:, -1, parent_idx]*self.dt

    def simulate(self, traj_solution, x0=None, u=None, output_traj=None):
        # Fills in the values of the state variables in traj_solution,
        # traj_solution (batch_size, sequence_length+sim_steps, num_states+num_control)
        # x0 - initial state (batch_size, sequence_length, num_states)
        # u - control inputs (batch_size, sim_steps+sequence_length, num_control)
        # output_traj - model outputs to fill in too (batch_size, sim_steps, num_outputs))
        
        # Initialize simulation data
        seq_length = self.config.sequence_length
        sim_steps = traj_solution.size(1) - seq_length
        if x0 is not None:
            traj_solution[:, :seq_length, :self.n_state] = x0
        if u is not None:
            traj_solution[:, :, -self.n_control:] = u

        with self.amp_context and torch.no_grad():
            for i in range(sim_steps):
                # Pass in the trajectory up to the t+1 step
                if output_traj is not None:
                    self._step(traj_solution[:, i:i+seq_length+1, :], output_traj[:, i, :])
                else:
                    self._step(traj_solution[:, i:i+seq_length+1, :])