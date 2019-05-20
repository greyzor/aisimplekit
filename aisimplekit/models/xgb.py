"""
Xgboost algorithm wrapper.
"""
import xgboost as xgb
import gc

def xgb_train_cv(params, df_train, df_val, predictors, target='target',
    objective='binary:logistic', early_stopping_rounds=25,
    num_boost_round=200, verbose_eval=5
):
    """Train an xgboost model with cross-validation.

    :param params: parameters used to train the xgboost model
    :type params: dict
    :param df_train: input dataframe
    :type df_train: pd.DataFrame
    :param df_val: cross-validation dataframe
    :type df_val: pd.DataFrame
    :param predictors: list of predictor columns
    :type predictors: list[str]
    :param target: name of target column
    :type target: str
    :param objective: objective to train on.
    :type objective: str
    :param early_stopping_rounds: a value activates early stopping
    :type early_stopping_rounds: int or None
    :param num_boost_round: number of boosting iterations
    :type num_boost_round: int
    :param verbose_eval: specifies if the eval metric is printed on each boosting stage.
    :type verbose_eval: bool

    :returns: the boosted model
    :rtype: xgboost.Booster
    """
    xgb_params = {
        'eta': 0.15,
        'tree_method': "hist",
        'max_bin': 256,
        'grow_policy': "lossguide",
        'max_leaves': 31, #1400,  
        'max_depth': 5, #0
        'subsample': 0.9, 
        'colsample_bytree': 0.9, # 0.7
        'colsample_bylevel':0.9,
        'min_child_weight':0,
        'alpha': 3,
        'lambda': 2, # 1
        'objective': objective, 
        'scale_pos_weight': 50, # 9
        'eval_metric': 'auc',
        'nthread': 8,
        'random_state': 99, 
        'silent': True
    }
    xgb_params.update(params) # Overriding default params

    dtrain = xgb.DMatrix(df_train[predictors], df_train[target])
    dvalid = xgb.DMatrix(df_val[predictors], df_val[target])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    model = xgb.train(
        xgb_params, dtrain, num_boost_round, watchlist,
        maximize=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval
    )
    del(dtrain)
    del(dvalid)
    gc.collect()
    return model