
import pandas as pd
import numpy as np


def extract_results(results, main_key='history', data_key='train'):
    df = None
    if main_key == 'history':
        df = pd.DataFrame(results['results_train']['history']['values'],
                          columns=results['results_train']['history']['headers'])

    elif main_key == 'fairness':
        user_dict = results['results_eval'][data_key]['fairness']

        new_results = {'independence': None, 'separation': None, 'sufficiency': None}
        for m in ('independence', 'separation', 'sufficiency'):
            tmp = []
            y_tuple = ('y = 0', 'y = 1')
            if m == 'sufficiency':
                y_tuple = ('y_pred = 0', 'y_pred = 1')
            for yi, y in enumerate(y_tuple):

                s_tuple = ('s = 0', 's = 1', 'Difference y=' + str(yi) + ' (s1-s0)')
                if m == 'sufficiency':
                    s_tuple = ('s = 0', 's = 1', 'Difference y_pred = ' + str(yi) + ' (s1-s0)')

                for s in s_tuple:
                    tmp.append(user_dict[m][y][s])
            new_results[m] = tmp

        headers = ('y = 0, s = 0', 'y = 0, s = 1', 'y = 0 (s0 - s1)', 'y = 1, s = 0', 'y = 1, s = 1', 'y = 1 (s0 - s1)')

        df = pd.DataFrame.from_dict(new_results, orient='index', columns=headers)

    return df


def concanenate_results(filenames, dir_name='./', main_key='history', data_key='train'):
    """
    filenames: list of filenames to be loaded and concatenated

    returns: out, rows, columns
    out : np.array of shape (len(filenames), len(rows), len(columns))
    """
    out, rows, columns = None, None, None
    for fname in filenames:
        with open(dir_name + fname + '.pkl', 'rb') as f:
            results = pickle.load(f)

        df = extract_results(results, main_key=main_key, data_key=data_key)
        rows, columns = list(df.index), list(df.columns)

        tmp = df.values

        if out is None:
            out = tmp.reshape([1, tmp.shape[0], -1])
        else:
            out = np.concatenate((out, tmp.reshape([1, tmp.shape[0], -1])), axis=0)

    return out, rows, columns


if __name__ == "__main__":
    import pickle

    # load results using pickle
    with open('../examples/unfair_2d_results/nonrobust.pkl', 'rb') as f:
        results = pickle.load(f)

    # extract results
    out1 = extract_results(results, main_key='history')

    out2 = extract_results(results, main_key='fairness', data_key='test')

    out3 = extract_results(results, main_key='fairness', data_key='train')

    out4, rows, columns = concanenate_results(['nonrobust', 'nonrobust'], dir_name='../examples/unfair_2d_results/',
                                              main_key='history', data_key='train')

    out5, rows, columns = concanenate_results(['nonrobust', 'nonrobust'], dir_name='../examples/unfair_2d_results/',
                                              main_key='fairness', data_key='val')

    out5[:, :, columns.index('y = 0 (s0 - s1)')]