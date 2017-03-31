import numpy as np

from astropy import units as u
from astropy.table import Table

from ctapipe.io import CameraGeometry

from ctapipe.utils.fitshistogram import Histogram

from sklearn.ensemble.forest import ForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator


def convert_astropy_array(arr, unit=None):
    if unit is None:
        unit = arr[0].unit
        return (np.array([a.to(unit).value for a in arr])*unit).si
    else:
        return np.array([a.to(unit).value for a in arr])*unit


def proba_drifting(x):
    """ gives more weight to outliers -- i.e. close to 0 and 1 """
    return 10*x**3 - 15*x**4 + 6*x**5


def proba_weighting(xs, weights):
    s = np.sum(weights)

    for x, w in zip(xs, weights):
        print(x, x/w, s)
    print()
    return [x * (w / s) for x, w in zip(xs, weights)]


class fancy_EventClassifier:
    def __init__(self, classifier=RandomForestClassifier, **kwargs):

        self.kwargs = kwargs
        self.clf = classifier(**kwargs)

    def __getattr__(self, attr):
        return getattr(self.clf, attr)

    def __str__(self):
        # return the class name of the used classifier
        return str(self.clf).split("(")[0]

    def fit(self, X, y):

        trainFeatures = []
        trainClasses  = []

        for evt, cl in zip(X, y):
            for i, tel in evt.items():
                trainFeatures.append(tel)
            trainClasses  += [cl]*len(evt)
        return self.clf.fit(trainFeatures, trainClasses)

    def predict_proba(self, X):

        predict_proba = []
        for evt in X:
            tels = [tel for tel in evt.values()]
            proba = self.clf.predict_proba(tels)
            #predict_proba.append(np.mean(proba, axis=0))
            predict_proba.append(np.mean(proba_drifting(proba), axis=0))

        return np.array(predict_proba)

    def save(self, path):
        from sklearn.externals import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        from sklearn.externals import joblib
        return joblib.load(path)

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







class EventClassifier:
    def __init__(self,
                 class_list=['g', 'p'],
                 axis_names=["log(E / GeV)"],
                 ranges=[[2, 8]], nbins=[6],
                 cutflow=None):

        self.class_list = class_list
        self.axis_names = axis_names
        self.ranges = ranges
        self.nbins = nbins

        self.wrong = self.create_histogram_classes_dict()
        self.total = self.create_histogram_classes_dict()

        self.Features = self.create_empty_classes_dict()
        self.MCEnergy = self.create_empty_classes_dict()

        self.cutflow = cutflow

    def create_empty_classes_dict(self, class_list=None):
        if class_list is None:
            class_list = self.class_list

        mydict = {}
        for cl in class_list:
            mydict[cl] = []
        return mydict

    def create_histogram_classes_dict(self, class_list=None):
        if class_list is None:
            class_list = self.class_list

        mydict = {}
        for cl in class_list:
            mydict[cl] = Histogram(axis_names=self.axis_names,
                                   nbins=self.nbins, ranges=self.ranges)
        return mydict

    def equalise_nevents(self, NEvents=None):
        minEvents = min([len(self.Features[cl]) for cl in self.class_list])
        if NEvents is None or NEvents > minEvents:
            NEvents = minEvents

        for cl in self.Features.keys():
            self.Features[cl] = self.Features[cl][:minEvents]
            self.MCEnergy[cl] = self.MCEnergy[cl][:minEvents]

    def learn(self, clf=None):
        trainFeatures = []
        trainClasses  = []

        for cl in self.Features.keys():
            for ev in self.Features[cl]:
                trainFeatures += ev
                trainClasses  += [cl]*len(ev)

        if clf is None:
            clf = RandomForestClassifier(
                n_estimators=40, max_depth=None,
                min_samples_split=2, random_state=0)
        clf.fit(trainFeatures, trainClasses)
        self.clf = clf

        from sklearn.model_selection import cross_val_score
        score = cross_val_score(clf, trainFeatures, trainClasses, scoring='accuracy')
        print("cross validation score:", score)

    def save(self, path):
        from sklearn.externals import joblib
        joblib.dump(self.clf, path)

    def load(self, path):
        from sklearn.externals import joblib
        self.clf = joblib.load(path)

    def predict(self, ev):
        # predicted class for every telescope
        predict_tels = self.clf.predict(ev)

        # probability for each class for every telescope
        # index 0 is the probability for gamma
        proba_index = 0
        predict_proba_g = self.clf.predict_proba(ev)[..., proba_index]


        # gammaness as the (weighted) mean probability of each telescope for a gamma event
        gammaness_o = np.mean(predict_proba_g)
        gammaness_d = np.mean(proba_drifting(predict_proba_g))

        # check if prediction returned gamma (what we are selecting on)
        gamma_ratio = (len(predict_tels[predict_tels == 'g']) / len(ev))

        return gamma_ratio, gammaness_o, gammaness_d

    def show_importances(self, feature_labels=None, clf=None):
        import matplotlib.pyplot as plt
        if clf is None:
            self.learn()
            clf = self.clf
        importances = clf.feature_importances_
        bins = range(importances.shape[0])
        plt.figure()
        plt.title("Feature Importances")
        if feature_labels:
            importances, feature_labels = \
                zip(*sorted(zip(importances, feature_labels), reverse=True))
            plt.xticks(bins, feature_labels, rotation=17)
        plt.bar(bins, importances,
                color='r', align='center')

    def self_check(self, min_tel=3, agree_threshold=.75, clf=None,
                   split_size=None, pytable=None, verbose=True):
        import matplotlib.pyplot as plt

        pred_table = {'g': Table(names=("NTels", "Energy", "gammaness_o", "gammaness_d",
                                         "right ratio")),
                      'p': Table(names=("NTels", "Energy", "gammaness_o", "gammaness_d",
                                        "right ratio"))}
        self.prediction_results = pred_table

        start = 0
        NEvents = min(len(features) for features in self.Features.values())

        if split_size is None:
            split_size = 10*max(NEvents//1000, 1)

        print("NEvents:", NEvents)

        # a mask for the features; created here to not always have to recreate it in the
        # loop -- only reset it there
        masks = {}
        # make np arrays out of the feature lists
        for cl in self.Features.keys():
            self.Features[cl] = np.array(self.Features[cl])
            self.MCEnergy[cl] = convert_astropy_array(self.MCEnergy[cl], u.GeV)
            masks[cl] = np.ones(len(self.Features[cl]), dtype=bool)

        while start+split_size <= NEvents:
            trainFeatures = []
            trainClasses  = []

            # training the classifier on all events but a chunk taken
            # out at a certain position
            for cl in self.Features.keys():
                # reset the mask
                mask = masks[cl]
                mask[~mask] = True
                # mask the chunk of features to test as False
                mask[start:start+split_size] = False

                # filling the list of features to train on
                for evs in self.Features[cl][mask]:
                    trainFeatures += [tels for tels in evs]
                    trainClasses += [cl]*len(evs)

            if clf is None:
                clf = RandomForestClassifier(n_estimators=40, max_depth=None,
                                             min_samples_split=2, random_state=0)
            clf.fit(trainFeatures, trainClasses)
            self.clf = clf

            # test the training on the previously excluded chunk
            for cl in self.Features.keys():
                for ev, en in zip(self.Features[cl][~mask],
                                  self.MCEnergy[cl][~mask]):

                    if self.cutflow:
                        self.cutflow[cl].count("to classify")

                    log_en = np.log10(en/u.GeV)

                    gamma_ratio, gammaness_o, gammaness_d,= self.predict(ev)
                    right_ratio = gamma_ratio if cl == 'g' else (1-gamma_ratio)

                    pred_table[cl].add_row([len(ev), en, gammaness_o, gammaness_d,
                                            right_ratio])

                    # if sufficient telescopes agree, assume it's a gamma
                    if gamma_ratio > agree_threshold:
                        PredictClass = "g"
                    else:
                        PredictClass = "p"

                    if self.cutflow:
                        if PredictClass == cl:
                            self.cutflow[cl].count("corr predict")
                            if cl == 'g' and len(ev) >= min_tel:
                                self.cutflow[cl].count("min tel {}".format(min_tel))
                        else:
                            self.cutflow[cl].count("false predict")
                            if len(ev) >= min_tel:
                                if PredictClass == 'g':
                                    self.cutflow[cl].count("false positive")
                                elif cl == 'g':
                                    self.cutflow[cl].count("false negative")

                    if PredictClass != cl and len(ev) >= min_tel:
                        self.wrong[cl].fill([log_en])
                    self.total[cl].fill([log_en])

                if verbose and sum(self.total[cl].hist) > 0:
                    print("wrong {}: {} out of {} => {:2.2f}".format(
                                    cl,
                                    sum(self.wrong[cl].hist),
                                    sum(self.total[cl].hist),
                                    sum(self.wrong[cl].hist) /
                                    sum(self.total[cl].hist) * 100*u.percent))

            start += split_size

            print()

        print()
        print("-"*30)
        print()

        # plotting
        plt.style.use('seaborn-talk')
        fig, ax = plt.subplots(4, 2)

        try:
            from modules.EfficiencyUncertainties import get_efficiency_uncertainties
        except ImportError:
            pass

        for col, cl in enumerate(["g", "p"]):
            if sum(self.total[cl].hist) > 0:
                print("wrong {}: {} out of {} => {:2.2f}"
                      .format(cl, sum(self.wrong[cl].hist),
                                  sum(self.total[cl].hist),
                                  sum(self.wrong[cl].hist) /
                                  sum(self.total[cl].hist) * 100*u.percent))
            try:
                errors = get_efficiency_uncertainties(self.wrong[cl].hist,
                                                      self.total[cl].hist)
                y_eff         = errors[:, 0]
                y_eff_lerrors = errors[:, 1]
                y_eff_uerrors = errors[:, 2]
            except ImportError:
                y_eff         = self.wrong[cl].hist/ self.total[cl].hist
                y_eff_lerrors = [0]*len(self.total[cl].hist)
                y_eff_uerrors = [0]*len(self.total[cl].hist)

            particle = "gamma" if cl == "g" else "proton"

            tax = ax[0, col]
            tax.errorbar(self.wrong[cl].bin_centers(0), y_eff,
                         yerr=[y_eff_lerrors, y_eff_uerrors])
            tax.set_title("{} misstag".format(particle))
            tax.set_xlabel("log(E/GeV)")
            tax.set_ylabel("incorrect / all")

            tax = ax[1, col]
            tax.hist(pred_table[cl]["right ratio"], bins=20, range=(0, 1))
            tax.set_title("fraction agreeing to {}"
                          .format(particle))
            tax.set_xlabel("agree ratio")
            tax.set_ylabel("events")

            tax = ax[2, col]
            histo = np.histogram2d(pred_table[cl]["NTels"], pred_table[cl]["gammaness_d"],
                                   bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
            histo_normed = histo / histo.max(axis=0)
            im = tax.imshow(histo_normed, interpolation='none', origin='lower',
                            aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
            cb = fig.colorbar(im, ax=tax)
            tax.set_title("arbitrary gammaness per event")
            tax.set_xlabel("NTels")
            tax.set_ylabel("drifted gammaness")

            tax = ax[3, col]
            histo = np.histogram2d(pred_table[cl]["NTels"], pred_table[cl]["gammaness_o"],
                                   bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
            histo_normed = histo / histo.max(axis=0)
            im = tax.imshow(histo_normed, interpolation='none', origin='lower',
                            aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
            cb = fig.colorbar(im, ax=tax)
            tax.set_title("arbitrary gammaness per event")
            tax.set_xlabel("NTels")
            tax.set_ylabel("gammaness")

        return clf
