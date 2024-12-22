
def flatten_params(params, parent_key = None):
    flat_params = {}

    for key, value in params.items():
        if isinstance(value, dict):
            flat_params.update(flatten_params(value, key))
            continue
        if isinstance(value, tuple) or isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, dict):
                    flatten_params(v, key + "_" + str(i))
                else:
                    flat_params[key + "_" + str(i)] = v
        else:
            if key == 'name' and parent_key is not None:
                key = parent_key + "_" + key
            flat_params[key] = value

    return flat_params