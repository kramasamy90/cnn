def find_suffix(s, t):
    if not t.startswith(s):
        return t
    return t[len(s)+1:]


def get_config(params):
    raw_config = params
    name = params['optimizer']['name']
    if name == 'adam':
        beta1 = raw_config['optimizer']['adam_beta1']
        beta2 = raw_config['optimizer']['adam_beta2']
        del raw_config['optimizer']['beta1']
        del raw_config['optimizer']['beta2']
        raw_config['optimize']['betas'] = (beta1, beta2)
    raw_config['device'] = 'cpu'

    config = {}

    for key, value in raw_config.items():
        new_key = find_suffix(name, key)
        config[new_key] = value

    return config

