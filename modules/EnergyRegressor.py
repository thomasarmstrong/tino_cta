import numpy as np

from astropy import units as u

from sklearn.ensemble import RandomForestRegressor


def proba_drifting(x):
    """ gives more weight to outliers -- i.e. close to 0 and 1 """
    return 10*x**3 - 15*x**4 + 6*x**5


def proba_weighting(xs, weights):
    s = np.sum(weights)

    for x, w in zip(xs, weights):
        print(x, x/w, s)
    print()
    return [x * (w / s) for x, w in zip(xs, weights)]


class mimi:
    # mimicks a dictionary but always returns the same value
    def __getitem__(self, item):
        return "mimimi"

cam_id_list = [
        #'GATE',
        #'HESSII',
        #'NectarCam',
        #'LSTCam',
        #'SST-1m',
        'FlashCam',
        'ASTRI',
        #'SCTCam',
        ]


class fancy_EnergyRegressor:
    def __init__(self, regressor=RandomForestRegressor,
                 cam_id_list=cam_id_list, **kwargs):

        self.cam_id_list=cam_id_list
        self.reg = regressor(**kwargs)
        self.reg_dict = {}
        for cam_id in cam_id_list:
            self.reg_dict[cam_id] = regressor(**kwargs)


    def __getattr__(self, attr):
        return getattr(self.reg, attr)

    def __str__(self):
        # return the class name of the used regressor
        return str(self.reg).split("(")[0]

    def fit(self, X, y):

        trainFeatures = {a: [] for a in cam_id_list}
        trainTarget   = {a: [] for a in cam_id_list}

        for evt, en in zip(X, y):
            for cam_id, tels in evt.items():
                trainFeatures[cam_id] += tels
                try:
                    trainTarget[cam_id] += [en.to(u.TeV).value]*len(tels)
                except:
                    trainTarget[cam_id] += [en]*len(tels)
        for cam_id in cam_id_list:
            if trainFeatures[cam_id]:
                self.reg_dict[cam_id].fit(trainFeatures[cam_id],
                                          trainTarget[cam_id])
        print()
        return self

    def predict(self, X):
        predict = []
        for evt in X:
            res = []
            for cam_id, tels in evt.items():
                res += self.reg_dict[cam_id].predict(tels).tolist()
            predict.append(np.mean(res))

        return np.array(predict)*u.TeV

    def save(self, path):
        from sklearn.externals import joblib
        joblib.dump(self.reg, path)

    @classmethod
    def load(cls, path):
        from sklearn.externals import joblib
        reg = joblib.load(path)
        self = cls(type(reg))

        self.reg = reg

        return self

    def show_importances(self, feature_labels=None):
        import matplotlib.pyplot as plt
        importances = self.clf.feature_importances_
        bins = range(importances.shape[0])
        plt.figure()
        plt.title("Feature Importances")
        if feature_labels:
            importances, feature_labels = \
                zip(*sorted(zip(importances, feature_labels), reverse=True))
            plt.xticks(bins, feature_labels, rotation=17)
        plt.bar(bins, importances,
                color='r', align='center')
