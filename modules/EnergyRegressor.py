import numpy as np

from astropy import units as u

from sklearn.ensemble import RandomForestRegressor


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

        trainFeatures = {a: [] for a in self.reg_dict.keys()}
        trainTarget   = {a: [] for a in self.reg_dict.keys()}

        for evt, en in zip(X, y):
            for cam_id, tels in evt.items():
                trainFeatures[cam_id] += tels
                try:
                    trainTarget[cam_id] += [en.to(u.TeV).value]*len(tels)
                except:
                    trainTarget[cam_id] += [en]*len(tels)
        for cam_id in self.reg_dict.keys():
            if trainFeatures[cam_id]:
                self.reg_dict[cam_id].fit(trainFeatures[cam_id],
                                          trainTarget[cam_id])
        return self

    def predict(self, X):
        predict = []
        predict_dict = []
        for evt in X:
            res = []
            res_dict = {}
            for cam_id, tels in evt.items():
                t_res = self.reg_dict[cam_id].predict(tels).tolist()
                res += t_res
                res_dict[cam_id] = np.mean(t_res)*u.TeV
            predict.append(np.mean(res))
            predict_dict.append(res_dict)

        return np.array(predict)*u.TeV, predict_dict

    def save(self, path):
        from sklearn.externals import joblib
        joblib.dump(self.reg_dict, path)

    @classmethod
    def load(cls, path):
        """
        Load the pickled dictionary of energy regressor from disk, create a husk
        `cls` instance and set the regressor dictionary
        """
        from sklearn.externals import joblib
        reg_dict = joblib.load(path)

        # need to get an instance of the initial regressor
        # `cam_id_list=[]` prevents `.__init__` to initialise `.reg_dict` itself,
        # since we are going to set it with the pickled one manually
        self = cls(cam_id_list=[])
        self.reg_dict = reg_dict

        # We also need some proxy-instance to relay all the function calls to.
        # So, since `reg_dict` is a dictionary, we loop over its values, set the first one
        # as `.reg` and break immediately
        for reg in reg_dict.values():
            self.reg = reg
            break

        return self

    def show_importances(self, feature_labels=None):
        import matplotlib.pyplot as plt
        n_tel_types = len(self.reg_dict)
        n_cols = np.ceil(np.sqrt(n_tel_types)).astype(int)
        n_rows = np.ceil(n_tel_types / n_cols).astype(int)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
        plt.suptitle("Feature Importances")
        for i, (cam_id, reg) in enumerate(self.reg_dict.items()):
            plt.sca(axs.ravel()[i])
            importances = reg.feature_importances_
            bins = range(importances.shape[0])
            plt.title(cam_id)
            if feature_labels:
                importances, feature_labels = \
                    zip(*sorted(zip(importances, feature_labels), reverse=True))
                plt.xticks(bins, feature_labels, rotation=17)
            plt.bar(bins, importances,
                    color='r', align='center')
