from hyperopt import hp

search_space = {
    'optimizer': hp.choice('optimizer', [
        {
            'name' : 'sgd',
            'params' : {
                'lr': hp.uniform('sgd_lr', 0.001, 0.1),
                'weight_decay': hp.uniform('sgd_weight_decay', 1e-5, 1e-2),
                'momentum' : hp.uniform('sgd_momentum', 0.001, 0.01)
            }
        },
        {
            'name' : 'adam',
            'params' : {
                'lr': hp.uniform('adam_lr', 0.001, 0.1),
                'weight_decay': hp.uniform('adam_weight_decay', 1e-5, 1e-2),
                'beta1' : hp.uniform('adam_beta1', 0.9, 0.99),
                'beta2' : hp.uniform('adam_beta2', 0.99, 0.999)
            }
        },
        {
            'name' : 'rmsprop',
            'params' : {
                'lr': hp.uniform('rmsprop_lr', 0.001, 0.1),
                'weight_decay': hp.uniform('rmsprop_weight_decay', 1e-5, 1e-2),
                'momentum' : hp.uniform('rmsprop_momentum', 0.001, 0.01),
                'alpha' : hp.uniform('rmsprop_alpha', 0.9, 0.99)
            }
        }
    ]),  
    'loss_fn' : 'ce_loss',
    'batch_size' : hp.choice('batch_size', [16, 32, 64, 128]),
    'epochs' : hp.choice('epochs', [5]),
    'device' : 'cuda'
}