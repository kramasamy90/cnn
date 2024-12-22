import numpy as np
from hyperopt import hp

search_space = {
    'optimizer': hp.choice('optimizer', [
        {
            'name' : 'sgd',
            'params' : {
                'lr': hp.uniform('sgd_lr', 0.0001, 0.001),
                'weight_decay': hp.uniform('sgd_weight_decay', 1e-5, 1e-2),
                'momentum' : hp.choice('sgd_momentum_choice', [
                    hp.uniform('sgd_momentum', 0.8, 0.99)
                ]),
            },
            'lr_damp_pretrained': hp.loguniform('sgd_lr_damp_pretrained', np.log(0.01), np.log(1))
        },
        {
            'name' : 'adam',
            'params' : {
                'lr': hp.uniform('adam_lr', 0.0004, 0.001),
                'weight_decay': hp.choice('adam_weight_decay', [0.008]),
                'beta1' : hp.choice('adam_beta1', [0.91]),
                'beta2' : hp.choice('adam_beta2', [0.998])
            },
            'lr_damp_pretrained': hp.loguniform('adam_lr_damp_pretrained', np.log(0.01), np.log(1))
        },
        {
            'name' : 'rmsprop',
            'params' : {
                'lr': hp.uniform('rmsprop_lr', 0.0001, 0.001),
                'weight_decay': hp.uniform('rmsprop_weight_decay', 1e-5, 1e-2),
                'momentum' : hp.choice('rmsprop_momentum_choice', [
                    hp.uniform('rmsprop_momentum', 0.8, 0.99)
                ]),
                'alpha' : hp.uniform('rmsprop_alpha', 0.9, 0.99)
            },
            'lr_damp_pretrained': hp.loguniform('rmp_prop_lr_damp_pretrained', np.log(0.01), np.log(1))
        }
    ]),  
    'loss_fn' : {
        'name': hp.choice('loss_fn', ['ce_loss']),
    },
    'batch_size' : hp.choice('batch_size', [2]),
    'epochs' : hp.choice('epochs', [8]),
    'device' : 'cuda'
}