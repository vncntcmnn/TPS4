from .folds import FoldsAverage


def objective(trial, model, folds, x_data, y_data, suggest, eval_set=None, aparams=None):
    param = suggest(trial)
    modeli = model(**param)
    fa = FoldsAverage(modeli, folds)
    fa.fit(x_data, y_data, eval_set=eval_set, aparams=aparams)
    return fa.get_score_train()


def lgbm_suggest(trial):
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    return param
