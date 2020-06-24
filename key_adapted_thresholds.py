import pandas as pd
import numpy as np
import time


class statRecognition():
    def __init__(self, class_method=0, update_method=0, stp_threshold=0.1):
        self.scores = []
        self.aval = []
        self.updates = []
        self.class_method = class_method
        self.update_method = update_method
        self.gallery = []
        self.mean = []
        self.std = []
        self.median = []
        self.threshold = np.arange(0, 1, stp_threshold)
        return None

    def enrollment(self, samples):

        samples = samples.reset_index()
        i_max = len(samples.columns)
        if self.class_method == 0:
            for _ in self.threshold:
                self.gallery.append(samples)
                self.mean.append(samples.mean().iloc[1:i_max])
                self.std.append(samples.std().iloc[1:i_max])

        elif self.class_method == 1:

            for _ in self.threshold:
                self.gallery.append(samples)
                self.mean.append(samples.mean().iloc[1:i_max])
                self.std.append(samples.std().iloc[1:i_max])
                self.median.append(samples.median().iloc[1:i_max])

        return None

    def authentication(self, genuine_samples, impostor_samples):
        self.samples = pd.concat([genuine_samples, impostor_samples])
        self.samples = self.samples.reset_index()
        i_max = len(self.samples.columns)
        n_classificator = len(self.gallery)
        aux = [-1 for _ in range(len(self.samples))]
        far = []
        frr = []

        if self.class_method == 0:

            if len(self.scores) == 0:

                for n in range(n_classificator):
                    n_features = len(self.mean[n])
                    exp = -abs(self.mean[n]-self.samples.iloc[:, 1:i_max]) / (
                        self.std[n])
                    self.scores.append(1-((np.exp(exp)).sum(axis=1)) /
                                       n_features)
                    self.aval.append(self.scores[n] < self.threshold[n])
            else:

                for n in range(n_classificator):
                    n_features = len(self.mean[n])
                    exp = -abs(self.mean[n]-self.samples.iloc[:, 1:len(
                        self.samples.columns)])/self.std[n]
                    self.scores[n] = 1 - ((np.exp(exp)).sum(axis=1)) / (
                        n_features)
                    self.aval[n] = self.scores[n] < self.threshold[n]

        elif self.class_method == 1:

            if len(self.scores) == 0:

                for n in range(n_classificator):
                    score = pd.Series([], dtype="float64")
                    n_features = len(self.mean[n])
                    for m in range(len(self.samples)):
                        A = []
                        for i in range(n_features):
                            lw = min(self.mean[n][i], self.median[n][i])
                            hr = max(self.mean[n][i], self.median[n][i])
                            low_lim = lw*(0.95-self.std[n][i]/self.mean[n][i])
                            high_lim = hr*(1.05+self.std[n][i]/self.mean[n][i])
                            if (self.samples.iloc[m, i+1] <= high_lim and
                                    self.samples.iloc[m, i+1] >= low_lim):
                                if len(A) == 0 and i == 0:
                                    A.append(1)
                                elif A[i-1] > 0:
                                    A.append(1.5)
                                elif A[i-1] == 0:
                                    A.append(1)
                            else:
                                A.append(0)
                        score = pd.concat([score,
                                           pd.Series([1 -
                                                      sum(A) /
                                                      ((n_features-1)*1.5 + 1)])],
                                          ignore_index=True)
                    self.scores.append(score)
                    self.aval.append(self.scores[n] < self.threshold[n])

            else:
                for n in range(n_classificator):
                    score = pd.Series([], dtype="float64")
                    n_features = len(self.mean[n])
                    for m in range(len(self.samples)):
                        A = []
                        for i in range(n_features):
                            lw = min(self.mean[n][i], self.median[n][i])
                            hr = max(self.mean[n][i], self.median[n][i])
                            low_lim = lw*(0.95 - self.std[n][i]/self.mean[n][i])
                            high_lim = hr*(1.05 + self.std[n][i]/self.mean[n][i])
                            if self.samples.iloc[m, i+1] <= high_lim and \
                                    self.samples.iloc[m, i+1] >= low_lim:
                                if len(A) == 0 and i == 0:
                                    A.append(1)
                                elif A[i-1] > 0:
                                    A.append(1.5)
                                elif A[i-1] == 0:
                                    A.append(1)
                            else:
                                A.append(0)
                        score = pd.concat([score, pd.Series([1 - sum(A) / (
                            (n_features-1)*1.5 + 1)])], ignore_index=True)
                    self.scores[n] = score
                    self.aval[n] = self.scores[n] < self.threshold[n]

        for n in range(n_classificator):
            gen_act = 0
            gen_rej = 0
            imp_act = 0
            imp_rej = 0
            for index, value in self.aval[n].items():

                if value is True and index < 50:
                    aux[index] = self.samples.iloc[index]
                    gen_act = gen_act + 1
                elif value is True and index >= 50:
                    aux[index] = self.samples.iloc[index]
                    imp_act = imp_act + 1
            if len(self.updates) < len(self.gallery):
                self.updates.append(aux)
            else:
                self.updates[n] = aux

            aux = [-1 for _ in range(len(self.samples))]
            gen_rej = len(genuine_samples) - gen_act
            imp_rej = len(impostor_samples) - imp_act
            far_n = imp_act/(imp_act + imp_rej)
            frr_n = gen_rej/(gen_act + gen_rej)
            far.append(far_n)
            frr.append(frr_n)

        return self.threshold, far, frr

    def update(self, gallery_size=50):

        for n in range(len(self.gallery)):
            up = 0
            for i in range(len(self.updates[n])):
                try:
                    if self.updates[n][i] == -1:
                        None
                except ValueError:
                    self.gallery[n] = pd.concat([self.gallery[n],
                                                 self.updates[n][i].to_frame().T],
                                                ignore_index=True)
                    up = 1

            if self.update_method == 0:
                n_updates = len(self.gallery[n]) - gallery_size

                if n_updates > 0:
                    for drop_i in range(n_updates):
                        self.gallery[n] = self.gallery[n].drop(drop_i)
                    self.gallery[n] = self.gallery[n].reset_index(drop=True)
                    self.mean[n] = self.gallery[n].mean().iloc[
                        1:len(self.gallery[n].columns)]
                    self.std[n] = self.gallery[n].std().iloc[
                        1:len(self.gallery[n].columns)]
                    if self.class_method == 1:
                        self.median[n] = self.gallery[n].median().iloc[
                            1:len(self.gallery[n].columns)]
                    if self.threshold[n]-np.exp(-self.mean[n].mean() /
                                                self.std[n].std()) >= 0:
                        self.threshold[n] = self.threshold[n] - \
                            np.exp(-self.mean[n].mean()/self.std[n].std())

            elif up == 1 and self.update_method == 1:

                self.mean[n] = self.gallery[n].mean().iloc[1:len(self.gallery[n].columns)]
                self.std[n] = self.gallery[n].std().iloc[1:len(self.gallery[n].columns)]
                if self.class_method == 1:
                    self.median[n] = self.gallery[n].median().iloc[
                        1:len(self.gallery[n].columns)]
                if self.threshold[n] - \
                        np.exp(-self.mean[n].mean()/self.std[n].std()) >= 0:
                    self.threshold[n] = self.threshold[n] - \
                        np.exp(-self.mean[n].mean()/self.std[n].std())

        return None


timer = time.localtime()
ds = pd.read_csv('DSL-StrongPasswordData.csv', sep=',', header=0)
threshold = []
far = []
frr = []
eer = []
far_mean = []
frr_mean = []
s = []
n_users = 51
step = 0.1
n_models = len(np.arange(0, 1, step))
blank = np.empty((n_users, 7))
blank[:] = np.NaN

for model in range(n_models):
    far_mean.append([])
    frr_mean.append([])
    far_mean[model] = pd.DataFrame(blank).copy()
    frr_mean[model] = pd.DataFrame(blank).copy()

for u in range(n_users):
    s.append(statRecognition(class_method=1, update_method=1, stp_threshold=step))
    s[u].enrollment(ds.iloc[400*u:400*u+50, 3:34])
    threshold.append([])
    far.append([])
    frr.append([])

for u in range(n_users):
    impostor = ds.drop(list(range(400*u, 400*u+400)))
    impostor = impostor.iloc[:, 3:34]
    for section in range(1, 8):
        genuine = ds.iloc[400*u+50*section:400*u+50+50*section, 3:34]
        impostor_section = impostor.sample(50)
        threshold_sec, far_sec, frr_sec = s[u].authentication(genuine, impostor_section)
        threshold[u].append(threshold_sec.copy())
        far[u].append(far_sec.copy())
        frr[u].append(frr_sec.copy())
        s[u].update()
    far[u] = pd.DataFrame(far[u])
    frr[u] = pd.DataFrame(frr[u])

for model in range(n_models):
    for section in range(1, 8):
        for u in range(n_users):
            far_mean[model][section-1][u] = far[u][model][section-1]
            frr_mean[model][section-1][u] = frr[u][model][section-1]
    far_mean[model] = far_mean[model].mean().mean()
    frr_mean[model] = frr_mean[model].mean().mean()

for n in range(len(far_mean)):
    eer.append(abs(far_mean[n] - frr_mean[n]))

t = pd.Series(threshold[0][0])
eer = pd.Series(eer)
far_mean = pd.Series(far_mean)
frr_mean = pd.Series(frr_mean)
t.to_csv("threshold.csv")
eer.to_csv("eer.csv")
far_mean.to_csv("far.csv")
frr_mean.to_csv("frr.csv")

print(timer.tm_min)
