# Helper function to validate model hyperparameters
def validate_param(param, name, min_val=None, max_val=None):
    if min_val is not None:
        if param < min_val:
            raise ValueError(f"Parameter {name} must be greater than {min_val}: {param}")
    if max_val is not None:
        if param > max_val:
            raise ValueError(f"Parameter {name} must be less than {max_val}: {param}")

# Helper function to validate model optimization parameters
def validate_opt_param(param, name, options=['select', 'range'], min_val=None, max_val=None, select_from=None):
    # Skip if param is not a dict (ie. not being optimized)
    if not isinstance(param, dict):
        return
    
    # Make sure there is only one key in the param
    if len(param) != 1:
        raise ValueError(f"Optimization parameter {name} dict must have only one key: {param}")
    
    # Make sure the param key is in the opt_type
    opt_type = next(iter(param.keys()))
    if opt_type not in options:
        raise ValueError(f"Optimization option for {name} must be one of {options}: {param}")
    
    if opt_type == 'select':
        # Validate select from
        if select_from is not None:
            for val in param['select']:
                if val not in select_from:
                    raise ValueError(f"All select values for {name} must be one of {select_from}: {param}")
        # Else make sure the values are between min and max
        else:
            for val in param['select']:
                if min_val is not None:
                    if val < min_val:
                        raise ValueError(f"Select value for {name} must be greater than {min_val}: {param}")
                if max_val is not None:
                    if val > max_val:
                        raise ValueError(f"Select value for {name} must be less than {max_val}: {param}")
    elif opt_type == 'range':
        # Make sure the list of the first key has 3 values
        if len(param['range']) != 3:
            raise ValueError(f"Range for {name} must have the format [min, max, step]: {param}")
        # Make sure min is between min and max
        if param['range'][0] < min_val or param['range'][0] > max_val:
            raise ValueError(f"Min value for {name} must be between {min_val} and {max_val}: {param}")
        # Make sure max is between min and max
        if param['range'][1] < min_val or param['range'][1] > max_val:
            raise ValueError(f"Max value for {name} must be between {min_val} and {max_val}: {param}")
        # Make sure step is greater than 0
        if param['range'][2] <= 0:
            raise ValueError(f"Range step value for {name} must be greater than 0: {param}")
