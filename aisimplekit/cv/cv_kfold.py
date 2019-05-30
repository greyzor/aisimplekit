## https://www.kaggle.com/adarshchavakula/how-to-cross-validate-properly
from sklearn.model_selection import KFold, ShuffleSplit
import lightgbm as lgb
import numpy as np
from sklearn import metrics
import gc

def cross_validate_lgbm_regressor(x, y, folds=3, repeats=1, predictors=[], categorical=[],
        lgbm_params={}, early_stopping_rounds=50, verbose_eval=10, metric='rmse'):
    '''
    Function to do the cross validation - using stacked Out of Bag method instead
        of averaging across folds.

    :param x: input data.
    :param y: expected output.
    :param folds: K, the number of folds to divide the data into.
    :param repeats: number of times to repeat validation process for more confidence.
    :param predictors: columns to be used by predictor.
    :param categorical: columns considered as categorical.
    :param lgbm_params: use them to fine tune the lgbm regressor.
    :param early_stopping_rounds: number of rounds before early stopping.
    :param verbose_eval:  show verbose verbose_eval rounds.
    :param metric: metric name to determine best fold. Just rmse supported right now.
    :returns: scores array and estimated best round number array (one array entry per fold).
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    '''
    scores = np.zeros((repeats, folds))
    best_num_rounds = np.zeros((repeats, folds))

    for r in range(repeats):
        i = 0
        print('\nCross Validating - Run', str(r+1), 'out of', str(repeats))
#        x,y = shuffle(x,y,random_state=r) #shuffle data before each repeat
#        kf = KFold(n_splits=folds, random_state=(1000*r+50*i)) #random split, different each time
        kf = ShuffleSplit(n_splits=folds, test_size=0.10, random_state=(1000*r+50*i))

        # Iterate on each fold
        for train_ind, test_ind in kf.split(x):
            print('Fold', i+1, 'out of',folds)
            x_train, y_train = x[train_ind,:], y[train_ind]
            x_valid, y_valid = x[test_ind,:], y[test_ind]

            #############################################################
            # LGBM Dataset Formatting
            lgtrain = lgb.Dataset(x_train, y_train, feature_name=predictors,
                                    categorical_feature = categorical)
            lgvalid = lgb.Dataset(x_valid, y_valid, feature_name=predictors,
                                    categorical_feature = categorical)
            del (x_train)
            gc.collect()

            # Train the model
            lgb_clf = lgb.train(lgbm_params, lgtrain, num_boost_round=n_rounds,
                valid_sets=[lgtrain, lgvalid], valid_names=['train','valid'],
                early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval
            )
            #############################################################
            # Make predictions
            y_pred = lgb_clf.predict(x_valid)
            del(x_valid)
            del(lgtrain)
            del(lgvalid)
            gc.collect()

            # Compute rmse metric.
            if metric is 'rmse':
                rmse = np.sqrt(metrics.mean_squared_error(y_valid, y_pred))
            else:
                raise Exception('Unsupported metric: {}'.format(metric))

            best_num_rounds[r,i] = lgb_clf.best_iteration
            scores[r,i] = rmse

            del(lgb_clf)
            del(y_pred)
            del(y_valid)
            gc.collect()
            i+=1

    return (scores, best_num_rounds)