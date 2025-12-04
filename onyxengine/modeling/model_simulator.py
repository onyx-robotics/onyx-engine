import torch
from typing import List, Dict, NamedTuple
from contextlib import nullcontext
from .model_features import Input, Output

class Operation(NamedTuple):
    """Represents a single operation in the simulation step."""
    feature_name: str
    parent_name: str
    relation: str
    feature_idx: int
    parent_idx: int


class FeatureTrajectory:
    """Wrapper class that provides both dictionary-style and tensor access to feature trajectories."""
    
    def __init__(self, tensor: torch.Tensor, name_to_idx: Dict[str, int], feature_names: List[str]):
        """
        Args:
            tensor: Full trajectory tensor of shape (batch_size, time_steps, n_features)
            name_to_idx: Dictionary mapping feature names to their indices in the tensor
            feature_names: Ordered list of all feature names
        """
        self.tensor = tensor
        self._name_to_idx = name_to_idx
        self._feature_names = feature_names
    
    def __getitem__(self, name: str) -> torch.Tensor:
        """Get trajectory for a specific feature by name.
        
        Args:
            name: Feature name
            
        Returns:
            Tensor of shape (batch_size, time_steps, 1) for the specified feature
        """
        if name not in self._name_to_idx:
            raise KeyError(f"Feature '{name}' not found. Available features: {list(self._name_to_idx.keys())}")
        idx = self._name_to_idx[name]
        return self.tensor[:, :, idx:idx+1]
    
    def __contains__(self, name: str) -> bool:
        """Check if a feature name exists."""
        return name in self._name_to_idx
    
    def keys(self):
        """Return an iterator over feature names."""
        return self._name_to_idx.keys()
    
    def items(self):
        """Return an iterator over (name, tensor) pairs."""
        for name in self._name_to_idx.keys():
            yield name, self[name]


class SimulationResult:
    """Result object from model simulation with named access to input and output trajectories."""
    
    def __init__(self, input_traj: torch.Tensor, output_traj: torch.Tensor, 
                 input_name_to_idx: Dict[str, int], output_name_to_idx: Dict[str, int],
                 input_feature_names: List[str], output_feature_names: List[str]):
        """
        Args:
            input_traj: Full input trajectory tensor (batch_size, time_steps, n_inputs)
            output_traj: Full output trajectory tensor (batch_size, time_steps, n_outputs)
            input_name_to_idx: Dictionary mapping input feature names to indices
            output_name_to_idx: Dictionary mapping output feature names to indices
            input_feature_names: Ordered list of input feature names
            output_feature_names: Ordered list of output feature names
        """
        self.inputs = FeatureTrajectory(input_traj, input_name_to_idx, input_feature_names)
        self.outputs = FeatureTrajectory(output_traj, output_name_to_idx, output_feature_names)


class ModelSimulator():
    def __init__(self, outputs: List[Output], inputs: List[Input], sequence_length: int, dt: float):
        self.sequence_length = sequence_length
        self.dt = dt
        self.amp_context = nullcontext()

        # Create feature name mappings for dictionary-based API
        # Store ordered lists of feature names (matching internal tensor order)
        self.derived_input_names = [input_item.name for input_item in inputs if input_item.relation is not None]
        self.external_input_names = [input_item.name for input_item in inputs if input_item.relation is None]
        self.output_names = [output.name for output in outputs]

        # Store full ordered list of input names matching the inputs list order (what model expects)
        self.input_names = [input_item.name for input_item in inputs]

        # Create name to index mappings
        self.derived_input_name_to_idx = {name: idx for idx, name in enumerate(self.derived_input_names)}
        self.external_input_name_to_idx = {name: idx for idx, name in enumerate(self.external_input_names)}
        self.output_name_to_idx = {name: idx for idx, name in enumerate(self.output_names)}

        # Also store full input name to index mapping (for input_traj which matches inputs list order)
        # Input trajectory matches the order of the inputs list (what the model expects)
        self.input_name_to_idx = {}
        for idx, input_item in enumerate(inputs):
            self.input_name_to_idx[input_item.name] = idx

        # Track which outputs are direct (not derived) and their indices
        # This is needed because the model's forward() only returns direct outputs
        self.direct_output_indices = [idx for idx, output in enumerate(outputs) if not output.is_derived]
        self.n_direct_outputs = len(self.direct_output_indices)

        self.n_state = len(self.derived_input_names)
        self.n_inputs = len(self.external_input_names)
        self.n_outputs = len(outputs)
        self.input_traj = None # (batch_size, sequence_length + sim_steps, num_inputs) - matches inputs list order
        self.output_traj = None # (batch_size, sim_steps, n_outputs)

        # Build operations using name-based approach
        self.derived_output_ops = self._build_derived_output_ops(outputs)        
        self.output_dep_input_ops = self._build_output_dep_input_ops(outputs, inputs)
        self.input_dep_input_ops = self._build_input_dep_input_ops(inputs)

    def _build_derived_output_ops(self, outputs: List[Output]) -> List[Operation]:
        """Build operations for derived outputs (outputs that depend on other outputs)."""
        ops = []
        output_name_to_idx = {output.name: idx for idx, output in enumerate(outputs)}

        for output in outputs:
            if output.parent is not None and output.relation is not None:
                ops.append(Operation(
                    feature_name=output.name,
                    parent_name=output.parent,
                    relation=output.relation,
                    feature_idx=output_name_to_idx[output.name],
                    parent_idx=output_name_to_idx[output.parent]
                ))

        # Sort: equal relations first, then by dependency depth
        return self._topological_sort_ops(ops, outputs)

    def _build_output_dep_input_ops(self, outputs: List[Output], inputs: List[Input]) -> List[Operation]:
        """Build operations for inputs that depend on outputs."""
        ops = []
        output_name_to_idx = {output.name: idx for idx, output in enumerate(outputs)}
        input_name_to_idx = {input_item.name: idx for idx, input_item in enumerate(inputs)}
        output_names = {output.name for output in outputs}

        for input_item in inputs:
            if input_item.relation is not None and input_item.parent in output_names:
                ops.append(Operation(
                    feature_name=input_item.name,
                    parent_name=input_item.parent,
                    relation=input_item.relation,
                    feature_idx=input_name_to_idx[input_item.name],
                    parent_idx=output_name_to_idx[input_item.parent]
                ))

        return self._topological_sort_ops(ops, inputs)

    def _build_input_dep_input_ops(self, inputs: List[Input]) -> List[Operation]:
        """Build operations for inputs that depend on other inputs."""
        ops = []
        input_name_to_idx = {input_item.name: idx for idx, input_item in enumerate(inputs)}
        output_names = set(self.output_names)
        input_names = {input_item.name for input_item in inputs}

        for input_item in inputs:
            if input_item.relation is not None and input_item.parent in input_names and input_item.parent not in output_names:
                ops.append(Operation(
                    feature_name=input_item.name,
                    parent_name=input_item.parent,
                    relation=input_item.relation,
                    feature_idx=input_name_to_idx[input_item.name],
                    parent_idx=input_name_to_idx[input_item.parent]
                ))

        return self._topological_sort_ops(ops, inputs)

    def _topological_sort_ops(self, ops: List[Operation], items: List) -> List[Operation]:
        """Sort operations topologically, ensuring parents always come before children."""
        # Create item map for dependency checking
        item_map = {item.name: item for item in items}

        def compute_depth(op: Operation) -> int:
            """Compute dependency depth for an operation."""
            depth = 0
            parent_name = op.parent_name
            while parent_name in item_map:
                depth += 1
                parent_item = item_map[parent_name]
                if hasattr(parent_item, 'parent') and parent_item.parent:
                    parent_name = parent_item.parent
                else:
                    break
            return depth

        # Sort by dependency depth only - parents (lower depth) come before children (higher depth)
        return sorted(ops, key=compute_depth)

    def _step(self, x, dx: torch.Tensor, prev_outputs: torch.Tensor = None):
        """Optimized step function with vectorized state updates and derived output computation.
        
        Args:
            x: Input trajectory slice (batch_size, sequence_length + 1, n_inputs)
            dx: Output trajectory slice to write to (batch_size, n_outputs) - modified in place
            prev_outputs: Previous outputs for delta/derivative operations (batch_size, n_outputs) or None
        """
        # Do a single forward step of the model - compute direct outputs
        dx[:, self.direct_output_indices] = self.forward(x[:, :-1, :])

        # Compute derived outputs FIRST, before updating inputs
        # This ensures inputs that depend on derived outputs use the correct values
        for op in self.derived_output_ops:
            if op.relation == 'equal':
                dx[:, op.feature_idx] = dx[:, op.parent_idx]
            elif op.relation == 'delta':
                if prev_outputs is None:
                    dx[:, op.feature_idx] = dx[:, op.parent_idx]
                else:
                    dx[:, op.feature_idx] = prev_outputs[:, op.feature_idx] + dx[:, op.parent_idx]
            elif op.relation == 'derivative':
                if prev_outputs is None:
                    dx[:, op.feature_idx] = dx[:, op.parent_idx] * self.dt
                else:
                    dx[:, op.feature_idx] = prev_outputs[:, op.feature_idx] + dx[:, op.parent_idx] * self.dt

        # Update inputs AFTER derived outputs so they can use the correct derived output values
        # Output-dependent inputs use dx (outputs) as source
        for op in self.output_dep_input_ops:
            if op.relation == 'equal':
                x[:, -1, op.feature_idx] = dx[:, op.parent_idx]
            elif op.relation == 'delta':
                x[:, -1, op.feature_idx] = x[:, -2, op.feature_idx] + dx[:, op.parent_idx]
            elif op.relation == 'derivative':
                x[:, -1, op.feature_idx] = x[:, -2, op.feature_idx] + dx[:, op.parent_idx] * self.dt

        # Input-dependent inputs use x (inputs) as source
        for op in self.input_dep_input_ops:
            if op.relation == 'delta':
                x[:, -1, op.feature_idx] = x[:, -2, op.feature_idx] + x[:, -1, op.parent_idx]
            elif op.relation == 'derivative':
                x[:, -1, op.feature_idx] = x[:, -2, op.feature_idx] + x[:, -1, op.parent_idx] * self.dt

    def _validate_feature_dict(self, feature_dict: Dict[str, torch.Tensor], 
                               expected_feature_names: List[str],
                               expected_time_steps: int,
                               batch_size: int = None,
                               context_name: str = "features") -> int:
        """Validate feature dictionary tensor shapes and consistency.
        
        Args:
            feature_dict: Dictionary mapping feature names to tensors
            expected_feature_names: List of expected feature names
            expected_time_steps: Expected number of time steps in tensors
            batch_size: Expected batch size (if None, inferred from first tensor)
            context_name: Context name for error messages (e.g., "x0", "external_inputs")
            
        Returns:
            Batch size (inferred or validated)
            
        Raises:
            ValueError: If validation fails with detailed error message
        """
        # Validate all required features are provided
        missing = set(expected_feature_names) - set(feature_dict.keys())
        if missing:
            raise ValueError(f"Missing required {context_name} features: {missing}. Provided: {list(feature_dict.keys())}")

        # Check for extra features
        extra = set(feature_dict.keys()) - set(expected_feature_names)
        if extra:
            raise ValueError(f"Unknown {context_name} features: {extra}. Expected: {expected_feature_names}")

        # Validate tensor shapes and batch size consistency
        batch_sizes = []

        for name in expected_feature_names:
            tensor = feature_dict[name]

            # Enforce 3D tensor with exact shape (batch_size, time_steps, 1)
            if tensor.dim() != 3:
                raise ValueError(
                    f"{context_name.capitalize()} feature '{name}' must be a 3D tensor. "
                    f"Expected shape: (batch_size, {expected_time_steps}, 1). "
                    f"Got shape: {tensor.shape} with {tensor.dim()} dimensions."
                )

            tensor_batch, tensor_time, tensor_feat = tensor.shape

            if tensor_feat != 1:
                raise ValueError(
                    f"{context_name.capitalize()} feature '{name}' has incorrect feature dimension: {tensor.shape}. "
                    f"Expected: (batch_size, {expected_time_steps}, 1). "
                    f"Got feature dimension of {tensor_feat}, expected 1."
                )

            if tensor_time != expected_time_steps:
                # Build detailed error message
                if context_name == "external_inputs":
                    error_msg = (
                        f"{context_name.capitalize()} feature '{name}' has incorrect time steps: {tensor.shape}. "
                        f"Expected: (batch_size, {expected_time_steps}, 1) where time_steps = sequence_length + sim_steps = {self.sequence_length} + {expected_time_steps - self.sequence_length} = {expected_time_steps}. "
                        f"Got time steps of {tensor_time}, expected {expected_time_steps}."
                    )
                else:
                    error_msg = (
                        f"{context_name.capitalize()} feature '{name}' has incorrect time steps: {tensor.shape}. "
                        f"Expected: (batch_size, {expected_time_steps}, 1). "
                        f"Got time steps of {tensor_time}, expected {expected_time_steps}."
                    )
                raise ValueError(error_msg)

            batch_sizes.append(tensor_batch)

        # Check batch size consistency
        if len(set(batch_sizes)) > 1:
            batch_info = ", ".join([f"'{name}': {bs}" for name, bs in zip(expected_feature_names, batch_sizes)])
            raise ValueError(
                f"Inconsistent batch sizes in {context_name}. Expected all features to have the same batch size. "
                f"Got: {batch_info}"
            )

        inferred_batch_size = batch_sizes[0]

        # If batch_size was provided, validate it matches
        if batch_size is not None and inferred_batch_size != batch_size:
            raise ValueError(
                f"{context_name.capitalize()} feature batch size mismatch. Expected batch_size={batch_size}, "
                f"but got batch_size={inferred_batch_size} from {context_name} tensors."
            )

        return inferred_batch_size

    def simulate(self, x0: Dict[str, torch.Tensor], 
                 external_inputs: Dict[str, torch.Tensor] = None, 
                 sim_steps: int = None, device=None, dtype=None) -> SimulationResult:
        """Simulate a trajectory with the model.
        
        Args:
            x0: Initial state as dictionary mapping feature names to tensors.
                     Dictionary keys should match derived input names (inputs with relations).
                     Each tensor MUST be 3D with shape (batch_size, sequence_length, 1).
            external_inputs: External inputs as dictionary mapping feature names to tensors.
                   Dictionary keys should match external input names (inputs without relations).
                   Each tensor MUST be 3D with shape (batch_size, sequence_length + sim_steps, 1).
                   If None, sim_steps must be provided.
            sim_steps: Number of simulation steps. If None, inferred from external_inputs tensor length.
            device: Target device for computation. If None, uses device of first tensor in x0.
            dtype: Target dtype for computation. If None, uses dtype of first tensor in x0.
            
        Returns:
            SimulationResult object with named access to input and output trajectories.
            
        Raises:
            ValueError: If tensor shapes are incorrect, batch sizes are inconsistent, or required features are missing.
        """
        seq_length = self.sequence_length

        # Determine device and dtype (always set, defaulting to first tensor's device/dtype)
        first_tensor = next(iter(x0.values()))
        device = device if device is not None else first_tensor.device
        dtype = dtype if dtype is not None else first_tensor.dtype

        # Validate x0 and get batch size
        batch_size = self._validate_feature_dict(
            feature_dict=x0,
            expected_feature_names=self.derived_input_names,
            expected_time_steps=self.sequence_length,
            batch_size=None,
            context_name="x0"
        )

        # Determine sim_steps (before validation so we can validate with correct expected time_steps)
        if sim_steps is None:
            if external_inputs is not None:
                # Infer from first input tensor - validation will catch any shape/dimension issues
                first_input_tensor = next(iter(external_inputs.values()))
                try:
                    input_time_steps = first_input_tensor.shape[1]
                    sim_steps = input_time_steps - seq_length
                except IndexError:
                    # Will be caught by validation with clearer error message
                    sim_steps = 0  # Temporary value, validation will catch the issue
            else:
                raise ValueError("sim_steps must be provided if external_inputs is None")

        # Validate external_inputs if provided (validation will catch shape/dimension issues)
        if external_inputs is not None:
            self._validate_feature_dict(
                feature_dict=external_inputs,
                expected_feature_names=self.external_input_names,
                expected_time_steps=self.sequence_length + sim_steps,
                batch_size=batch_size,
                context_name="external_inputs"
            )

        # Update amp_context based on dtype
        if dtype in (torch.bfloat16, torch.float16):
            device_type = device if isinstance(device, str) else device.type
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.amp_context = torch.amp.autocast(device_type=device_type, dtype=dtype)
        else:
            self.amp_context = nullcontext()

        # Initialize or reuse buffers (avoid reallocation when possible)
        total_length = seq_length + sim_steps
        total_inputs = len(self.input_names)  # Total inputs matching model's expected order
        need_realloc = (
            self.input_traj is None or 
            self.input_traj.size(0) != batch_size or 
            self.input_traj.size(1) != total_length or
            self.input_traj.device != device or
            self.input_traj.dtype != dtype
        )

        if need_realloc:
            self.input_traj = torch.zeros(batch_size, total_length, total_inputs, device=device, dtype=dtype)
            self.output_traj = torch.zeros(batch_size, sim_steps, self.n_outputs, device=device, dtype=dtype)
        else:
            # Reuse existing buffers - just zero them out
            self.input_traj.zero_()
            self.output_traj.zero_()

        # Move model to device once (check only if it's a PyTorch module)
        if isinstance(self, torch.nn.Module):
            params = list(self.parameters())
            if params and params[0].device != device:
                self.to(device=device)

        # Initialize simulation data using correct indices based on input_feature_names order
        # Copy x0 features to their correct positions in input_traj
        for name in self.derived_input_names:
            idx = self.input_name_to_idx[name]
            self.input_traj[:, :seq_length, idx:idx+1].copy_(x0[name].to(device=device, dtype=dtype))

        # Copy external inputs to their correct positions in input_traj
        if external_inputs is not None:
            for name in self.external_input_names:
                idx = self.input_name_to_idx[name]
                self.input_traj[:, :, idx:idx+1].copy_(external_inputs[name].to(device=device, dtype=dtype))

        # Main simulation loop
        with self.amp_context, torch.no_grad():
            for i in range(sim_steps):
                prev_outputs = self.output_traj[:, i-1, :] if i > 0 else None
                self._step(self.input_traj[:, i:i+seq_length+1, :], self.output_traj[:, i, :], prev_outputs)

        # Return SimulationResult with named access
        # SimulationResult creates FeatureTrajectory objects internally
        return SimulationResult(
            self.input_traj, 
            self.output_traj,
            self.input_name_to_idx,
            self.output_name_to_idx,
            self.input_names,
            self.output_names
        )
