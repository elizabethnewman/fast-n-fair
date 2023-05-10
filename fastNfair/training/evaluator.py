
import fastNfair.utils.statistics as stats
from sklearn import metrics


class Evaluator:
    def __init__(self, name=None):
        self.name = name

    def evaluate(self, fctn, data_train, data_val, data_test, threshold=0.0, names=('train', 'val', 'test')):
        results = {'train': {'full': None, 's = 0': None, 's = 1': None, 'fairness': None},
                   'val': {'full': None, 's = 0': None, 's = 1': None, 'fairness': None},
                   'test': {'full': None, 's = 0': None, 's = 1': None, 'fairness': None}}

        for i, data in enumerate((data_train, data_val, data_test)):
            x, y, s = data

            z = fctn.net(x)[0]
            y_pred = 1 * (z > threshold).view(-1)

            results[names[i]]['fairness'] = stats.fairness_metrics(y, y_pred, s)
            results[names[i]]['full'] = self.compute_statistics(z, y_pred, x, y)
            for j in (0, 1):
                results[names[i]]['s = ' + str(j)] = self.compute_statistics(z[s == j], y_pred[s == j],
                                                                             x[s == j], y[s == j])

        return results

    def compute_statistics(self, z, y_pred, x, y):
        out = stats.compute_statistics(y, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y, z.detach(), pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return {'stats': out, 'fpr': fpr, 'tpr': tpr, 'auc': auc}


if __name__ == "__main__":
    import torch
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

    evaluator = Evaluator()

    out = evaluator.evaluate(fctn, data_train, data_val, data_test)

