def find_suffix(s, t):
    if not t.startswith(s):
        return t
    return t[len(s)+1:]


def get_config(config):
    name = config['optimizer']['name']
    if name == 'adam':
        beta1 = config['optimizer']['params']['beta1']
        beta2 = config['optimizer']['params']['beta2']
        del config['optimizer']['params']['beta1']
        del config['optimizer']['params']['beta2']
        config['optimizer']['params']['betas'] = (beta1, beta2)
    # config['device'] = 'cpu'

    # config = {}

    # for key, value in raw_config.items():
    #     new_key = find_suffix(name, key)
    #     config[new_key] = value

    return config

