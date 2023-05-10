import torch
from copy import deepcopy
from fastNfair.training import train_one_epoch, test


class TrainerSGD:

    def __init__(self, optimizer, scheduler=None, regularier=None, batch_size=5, max_epochs=10):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularier
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def train(self, fctn, data_train, data_val, data_test, robust=False, radius=2e-1, verbose=False):
        results = {'train': {'loss': [], 'accuracy': []},
                   'val': {'loss': [], 'accuracy': []},
                   'test': {'loss': None, 'accuracy': None},
                   'theta': []}

        # extract data
        x_train, y_train, s_train = data_train
        x_val, y_val, s_val = data_val
        x_test, y_test, s_test = data_test

        # initial evaluation
        loss_train, acc_train = test(fctn, x_train, y_train)
        loss_val, acc_val = test(fctn, x_val, y_val)

        # store results
        results['train']['loss'].append(loss_train)
        results['train']['accuracy'].append(acc_train)
        results['val']['loss'].append(loss_val)
        results['val']['accuracy'].append(acc_val)
        results['theta'].append(deepcopy(fctn.net.state_dict()))

        if verbose:
            print((8 * '{:<15s}').format('epoch', 'lr', 'running', ' ', 'train', ' ', 'val', ' '))
            print_frmt = '{:<15d}{:<15.4e}{:<15.4e}{:<15.4f}{:<15.4e}{:<15.4f}{:<15.4e}{:<15.4f}'
            print(print_frmt.format(-1, self.optimizer.state_dict()['param_groups'][0]['lr'],
                                    0.0, 0.0, loss_train, acc_train, loss_val, acc_val))

        # main iteration
        for i in range(self.max_epochs):
            loss_running, acc_running = train_one_epoch(fctn, self.optimizer, x_train, y_train, s_train,
                                                        batch_size=self.batch_size,
                                                        robust=robust, radius=radius,
                                                        regularizer=self.regularizer)

            loss_train, acc_train = test(fctn, x_train, y_train)
            loss_val, acc_val = test(fctn, x_val, y_val)

            # store results
            results['train']['loss'].append(loss_train)
            results['train']['accuracy'].append(acc_train)
            results['val']['loss'].append(loss_val)
            results['val']['accuracy'].append(acc_val)
            results['theta'].append(deepcopy(fctn.net.state_dict()))

            if verbose:
                print(print_frmt.format(i, self.optimizer.state_dict()['param_groups'][0]['lr'],
                                        loss_running, acc_running, loss_train, acc_train, loss_val, acc_val))

            if self.scheduler is not None:
                self.scheduler.step()

        # final evaluation
        loss_test, acc_test = test(fctn, x_test, y_test)
        results['test']['loss'] = loss_test
        results['test']['accuracy'] = acc_test

        if verbose:
            print(' ')
            print('{:<15s}{:<15.4e}{:<15.4f}'.format('TRAIN:', loss_train, acc_train))
            print('{:<15s}{:<15.4e}{:<15.4f}'.format('VALID:', loss_val, acc_val ))
            print('{:<15s}{:<15.4e}{:<15.4f}'.format('TEST:', loss_test, acc_test))

        return results


if __name__ == "__main__":
    import hessQuik.activations as act
    import hessQuik.layers as lay
    import hessQuik.networks as net
    from fastNfair.data import generate_unfair_data
    from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression

    # for reproducibility
    torch.manual_seed(42)

    # generate data
    data_train = generate_unfair_data(200)
    data_val = generate_unfair_data(50)
    data_test = generate_unfair_data(50)

    # create linear network
    my_net = net.NN(lay.singleLayer(2, 1, act=act.identityActivation(), bias=True))

    # create objective function
    fctn = ObjectiveFunctionLogisticRegression(my_net)

    # choose optimizer
    opt = torch.optim.Adam(fctn.parameters(), lr=1e-2)

    trainer = TrainerSGD(opt)

    out = trainer.train(fctn, data_train, data_val, data_test, robust=False, verbose=True)