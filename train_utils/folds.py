import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class FoldsAverage:
    def __init__(self, model, folds):
        self.folds = folds
        self.model = model
        self.models = []
        self.fitted = False

    def fit(self, train_x, train_y, eval_set=None, aparams=None):
        oof_preds = np.zeros_like(train_y).astype(float)

        self.train_x = train_x
        self.train_y = train_y

        for tr_idx, va_idx in tqdm(self.folds):
            tr_x, va_x = self.train_x[tr_idx], self.train_x[va_idx]
            tr_y, va_y = self.train_y[tr_idx], self.train_y[va_idx]

            model = self.model
            kwargs = None
            if eval_set:
                kwargs = {eval_set: (va_x, va_y)}
            if aparams:
                if kwargs is not None:
                    kwargs = {**kwargs, **aparams}
                else:
                    kwargs = aparams
            if kwargs:
                model.fit(tr_x, tr_y, **kwargs)
            else:
                model.fit(tr_x, tr_y)
            self.models.append(model)

            oof_pred = model.predict_proba(va_x)[:, 1]
            oof_preds[va_idx] = oof_pred

        self.oof_preds = oof_preds
        self.fitted = True

    def get_score_train(self, threshold=0.5):
        if self.fitted:
            preds = self.oof_preds > threshold
            score = accuracy_score(self.train_y, preds)
            return score
        else:
            print('Model not fitted.')

    def predict_proba(self, test_x):
        preds = []
        for model in tqdm(self.models):
            pred = model.predict_proba(test_x)[:, 1]
            preds.append(pred)
        preds = np.mean(preds, axis=0)
        return preds

