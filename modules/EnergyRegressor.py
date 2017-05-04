import numpy as np

from astropy import units as u

from sklearn.ensemble import RandomForestRegressor


cam_id_list = [
        # 'GATE',
        # 'HESSII',
        # 'NectarCam',
        # 'LSTCam',
        # 'SST-1m',
        'FlashCam',
        'ASTRI',
        # 'SCTCam',
        ]


def dd(obj):
    """
    wraps a dummy dict around `obj`
    """
    return {"d": obj}


class fancy_EnergyRegressor:
    def __init__(self, regressor=RandomForestRegressor,
                 cam_id_list=cam_id_list, energy_unit=u.TeV, **kwargs):

        self.reg = regressor(**kwargs)
        self.reg_dict = {}
        self.energy_unit = energy_unit
        for cam_id in cam_id_list:
            self.reg_dict[cam_id] = regressor(**kwargs)

    def __getattr__(self, attr):
        return getattr(self.reg, attr)

    def __str__(self):
        # return the class name of the used regressor
        return str(self.reg).split("(")[0]

    def reshuffle_event_list(self, X, y):
        trainFeatures = {a: [] for a in self.reg_dict.keys()}
        trainTarget = {a: [] for a in self.reg_dict.keys()}

        for evt, en in zip(X, y):
            for cam_id, tels in evt.items():
                trainFeatures[cam_id] += tels
                try:
                    trainTarget[cam_id] += [en.to(self.energy_unit).value]*len(tels)
                except:
                    trainTarget[cam_id] += [en]*len(tels)
        return trainFeatures, trainTarget

    def fit(self, X, y):

        for cam_id in self.reg_dict.keys():
            try:
                self.reg_dict[cam_id].fit(X[cam_id], y[cam_id])
            except Exception as e:
                print(e)
        return self

    def predict(self, X):
        predict_list = []
        for evt in X:
            res = []
            for cam_id, tels in evt.items():
                t_res = self.reg_dict[cam_id].predict(tels).tolist()
                res += t_res
            predict_list.append(np.mean(res))

        return np.array(predict_list)*self.energy_unit

    def predict_dict(self, X):
        predict_list_dict = []
        for evt in X:
            res_dict = {}
            for cam_id, tels in evt.items():
                t_res = self.reg_dict[cam_id].predict(tels).tolist()
                res_dict[cam_id] = np.mean(t_res)*self.energy_unit
            predict_list_dict.append(res_dict)

        return predict_list_dict

    def save(self, path):
        from sklearn.externals import joblib
        for cam_id, reg in self.reg_dict.items():
            try:
                # assume that there is a `{cam_id}` keyword to replace in the string
                joblib.dump(reg, path.format(cam_id=cam_id))
            except IndexError:
                # if not, assume there is a naked `{}` somewhere left
                # if not, format won't do anything, so it doesn't matter
                joblib.dump(reg, path.format(cam_id))

    @classmethod
    def load(cls, path, cam_id_list=cam_id_list, energy_unit=u.TeV):
        """
        Load the pickled dictionary of energy regressor from disk, create a husk
        `cls` instance and set the regressor dictionary
        """
        from sklearn.externals import joblib

        # need to get an instance of the initial regressor
        # `cam_id_list=[]` prevents `.__init__` to initialise `.reg_dict` itself,
        # since we are going to set it with the pickled models manually
        self = cls(cam_id_list=[], energy_unit=energy_unit)
        for key in cam_id_list:
            try:
                # assume that there is a `{cam_id}` keyword to replace in the string
                self.reg_dict[key] = joblib.load(path.format(cam_id=key))
            except IndexError:
                # if not, assume there is a naked `{}` somewhere left
                # if not, format won't do anything, so it doesn't matter
                self.reg_dict[key] = joblib.load(path.format(key))

        # We also need some proxy-instance to relay all the function calls to.
        # So, since `reg_dict` is a dictionary, we loop over its values, set the first one
        # as `.reg` and break immediately
        for reg in self.reg_dict.values():
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
                importances, s_feature_labels = \
                    zip(*sorted(zip(importances, feature_labels), reverse=True))
                plt.xticks(bins, s_feature_labels, rotation=17)
            plt.bar(bins, importances,
                    color='r', align='center')
