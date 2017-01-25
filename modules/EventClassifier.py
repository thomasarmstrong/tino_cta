import numpy as np

from itertools import chain

from astropy import units as u

from ctapipe.io import CameraGeometry

from ctapipe.utils import linalg
from ctapipe.utils.fitshistogram import Histogram


from datapipe.utils.EfficiencyErrors import get_efficiency_errors

from sklearn.ensemble import RandomForestClassifier


class EventClassifier:

    def __init__(self,
                 class_list=['g', 'p'],
                 axis_names=["log(E / GeV)"],
                 ranges=[[2, 8]],
                 nbins=[6],):

        self.class_list = class_list
        self.axis_names = axis_names
        self.ranges = ranges
        self.nbins = nbins

        self.wrong = self.create_histogram_class_dict()
        self.total = self.create_histogram_class_dict()

        self.Features = self.create_empty_class_dict()
        self.MCEnergy = self.create_empty_class_dict()

        self.Eventcutflow = None

    def create_empty_class_dict(self, class_list=None):
        if class_list is None:
            class_list = self.class_list

        mydict = {}
        for cl in class_list:
            mydict[cl] = []
        return mydict

    def create_histogram_class_dict(self, class_list=None):
        if class_list is None:
            class_list = self.class_list

        mydict = {}
        for cl in class_list:
            mydict[cl] = Histogram(axis_names=self.axis_names,
                                   nbins=self.nbins, ranges=self.ranges)
        return mydict

    def equalise_nevents(self, NEvents):
        for cl in self.Features.keys():
            self.Features[cl] = self.Features[cl][:NEvents]
            self.MCEnergy[cl] = self.MCEnergy[cl][:NEvents]

    def learn(self, clf=None):
        trainFeatures   = []
        trainClasses    = []
        for cl in self.Features.keys():
            for ev in self.Features[cl]:
                trainFeatures += [tel[1:] for tel in ev]
                trainClasses  += [cl]*len(ev)

        if clf is None:
            clf = RandomForestClassifier(
                n_estimators=40, max_depth=None,
                min_samples_split=2, random_state=0)
        clf.fit(trainFeatures, trainClasses)
        self.clf = clf

    def save(self, path):
        from sklearn.externals import joblib
        joblib.dump(self.clf, path)

    def load(self, path):
        from sklearn.externals import joblib
        self.clf = joblib.load(path)

    def predict(self, ev):
        return self.clf.predict(ev)

    def show_importances(self, feature_labels=None):
        import matplotlib.pyplot as plt
        self.learn()
        importances = self.clf.feature_importances_
        bins = range(importances.shape[0])
        plt.figure()
        plt.title("Feature Importances")
        plt.bar(bins, importances,
                color='r', align='center')
        if feature_labels:
            plt.xticks(bins, feature_labels, rotation=17)

    def self_check(self, min_tel=3, agree_threshold=.75, clf=None,
                   split_size=None, verbose=True):
        import matplotlib.pyplot as plt

        right_ratios = self.create_empty_class_dict()
        proba_ratios = self.create_empty_class_dict()
        NTels        = self.create_empty_class_dict()

        start = 0
        NEvents = min(len(features)
                      for features in self.Features.values())

        if split_size is None:
            split_size = 10*max(NEvents//1000, 1)

        print("nevents:", NEvents)
        while start+split_size <= NEvents:
            trainFeatures = []
            trainClasses  = []
            '''
            training the classifier on all events but a chunck taken
            out at a certain position '''
            for cl in self.Features.keys():
                for ev in chain(self.Features[cl][:start],
                                self.Features[cl][start+split_size:]):
                    trainFeatures += [tel[1:] for tel in ev]
                    trainClasses  += [cl]*len(ev)

            if clf is None:
                clf = RandomForestClassifier(n_estimators=40, max_depth=None,
                                             min_samples_split=2, random_state=0)
            clf.fit(trainFeatures, trainClasses)

            '''
            test the training on the previously excluded chunck '''
            for cl in self.Features.keys():
                for ev, en in zip(
                        self.Features[cl][start:start+split_size],
                        self.MCEnergy[cl][start:start+split_size]
                                  ):

                    if self.Eventcutflow:
                        self.Eventcutflow[cl].count("to classify")

                    log_en = np.log10(en/u.GeV)

                    PredictTels = clf.predict([tel[:1]+tel[2:] for tel in ev])

                    predict_proba = clf.predict_proba([tel[:1]+tel[2:] for tel in ev])

                    proba_index = 0  # if cl == "g" else 1
                    proba_ratios[cl].append(np.sum(
                        10*predict_proba**3 - 15*predict_proba**4 + 6*predict_proba**5,
                                                   axis=0)[proba_index] /
                                            len(predict_proba))

                    NTels[cl].append(len(predict_proba))

                    # check if prediction was right
                    right_ratio = (len(PredictTels[PredictTels == cl]) /
                                   len(PredictTels))
                    right_ratios[cl].append(right_ratio)

                    # check if prediction returned gamma (what we are selecting on)
                    gamma_ratio = (len(PredictTels[PredictTels == 'g']) /
                                   len(PredictTels))

                    # if sufficient telescopes agree, assume it's a gamma
                    if gamma_ratio > agree_threshold:
                        PredictClass = "g"
                    else:
                        PredictClass = "p"

                    if self.Eventcutflow:
                        if PredictClass == cl:
                            self.Eventcutflow[cl].count("corr predict")
                            if cl == 'g' and len(ev) >= min_tel:
                                self.Eventcutflow[cl].count("min_tel_{}".format(min_tel))
                        else:
                            self.Eventcutflow[cl].count("false predict")
                            if len(ev) >= min_tel:
                                if PredictClass == 'g':
                                    self.Eventcutflow[cl].count("false positive")
                                elif cl == 'g':
                                    self.Eventcutflow[cl].count("false negative")

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
        y_eff         = self.create_empty_class_dict()
        y_eff_lerrors = self.create_empty_class_dict()
        y_eff_uerrors = self.create_empty_class_dict()

        try:
            from utils.EfficiencyErrors import get_efficiency_errors
        except ImportError:
            pass

        for cl in self.Features.keys():
            if sum(self.total[cl].hist) > 0:
                print("wrong {}: {} out of {} => {}"
                      .format(cl, sum(self.wrong[cl].hist),
                                  sum(self.total[cl].hist),
                                  sum(self.wrong[cl].hist) /
                                  sum(self.total[cl].hist) * 100*u.percent))

            for wr, tot in zip(self.wrong[cl].hist, self.total[cl].hist):
                try:
                    errors = get_efficiency_errors(wr, tot)
                except:
                    errors = [wr/tot if tot > 0 else 0, 0, 0]
                y_eff        [cl].append(errors[0])
                y_eff_lerrors[cl].append(errors[1])
                y_eff_uerrors[cl].append(errors[2])


        plt.style.use('seaborn-talk')
        fig, ax = plt.subplots(4, 2)
        tax = ax[0, 0]
        tax.errorbar(self.wrong["g"].bin_centers(0), y_eff["g"],
                     yerr=[y_eff_lerrors["g"], y_eff_uerrors["g"]])
        tax.set_title("gamma misstag")
        tax.set_xlabel("log(E/GeV)")
        tax.set_ylabel("incorrect / all")

        tax = ax[0, 1]
        tax.errorbar(self.wrong["p"].bin_centers(0), y_eff["p"],
                     yerr=[y_eff_lerrors["p"], y_eff_uerrors["p"]])
        tax.set_title("proton misstag")
        tax.set_xlabel("log(E/GeV)")
        tax.set_ylabel("incorrect / all")

        tax = ax[1, 0]
        tax.bar(self.total["g"].bin_lower_edges[0][:-1], self.total["g"].hist,
                width=(self.total["g"].bin_lower_edges[0][-1] -
                       self.total["g"].bin_lower_edges[0][0]) /
                len(self.total["g"].bin_centers(0)))
        tax.set_title("gamma numbers")
        tax.set_xlabel("log(E/GeV)")
        tax.set_ylabel("events")

        tax = ax[1, 1]
        tax.bar(self.total["p"].bin_lower_edges[0][:-1], self.total["p"].hist,
                width=(self.total["p"].bin_lower_edges[0][-1] -
                       self.total["p"].bin_lower_edges[0][0]) /
                len(self.total["p"].bin_centers(0)))
        tax.set_title("proton numbers")
        tax.set_xlabel("log(E/GeV)")
        tax.set_ylabel("events")

        tax = ax[2, 0]
        tax.hist(right_ratios['g'], bins=20, range=(0, 1), normed=True)
        tax.set_title("fraction of classifiers per event agreeing to gamma")
        tax.set_xlabel("agree ratio")
        tax.set_ylabel("PDF")

        tax = ax[2, 1]
        tax.hist(right_ratios['p'], bins=20, range=(0, 1), normed=True)
        tax.set_title("fraction of classifiers per event agreeing to proton")
        tax.set_xlabel("agree ratio")
        tax.set_ylabel("PDF")

        tax = ax[3, 0]

        histo = np.histogram2d(NTels['g'], proba_ratios['g'],
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        tax.imshow(histo_normed, interpolation='none', origin='lower', aspect='auto',
                   extent=(1, 9, 0, 1), cmap=plt.cm.hot)
        tax.set_title("fraction of classifiers per event predicting gamma")
        tax.set_xlabel("NTels")
        tax.set_ylabel("agree ratio")

        tax = ax[3, 1]
        histo = np.histogram2d(NTels['p'], proba_ratios['p'],
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        tax.imshow(histo_normed, interpolation='none', origin='lower', aspect='auto',
                   extent=(1, 9, 0, 1), cmap=plt.cm.hot)
        tax.set_title("fraction of classifiers per event predicting gamma")
        tax.set_xlabel("NTels")
        tax.set_ylabel("agree ratio")
