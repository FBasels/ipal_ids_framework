import os

from ids.ids import MetaIDS
import pandas as pd
import numpy as np
from sklearn import mixture
from sklearn.linear_model import Lasso
from sklearn import metrics
import ids.invariant_rules.Util as Util
import time
import json

class InvariantRulesIDS(MetaIDS):
    _name = "InvariantRulesIDS"
    _description = "Invariant rule mining from training data"
    _requires = ["train.state", "live.state"]
    _invariant_rules_default_settings = {
        # soon here will be beautiful default settings
        "eps": 0.01,  # same as in the paper
        "sigma": 1.1,  # buffer scaler
        "theta_value": 0.08,  # same as in the paper
        "gamma_value": 0.9,  # same as in the paper
        "max_k": 4,
        "max_comp": 4,   # number of mixture components

        # List of component identifiers. Separated in lists containing components of different parts. Here keyArray for SWaT
        "keyArray": [['FIT101','LIT101','MV101','P101','P102'], ['AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206'],
          ['DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302'], ['AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401'],
          ['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503'],['FIT601','P601','P602','P603']]
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._invariant_rules_default_settings)

    def distr_driven_pred(self, training_data: pd.DataFrame, cont_vars: []):
        print("Generating distribution-driven predicates...")
        for entry in cont_vars:
            print('generate distribution-driven predicates for', entry)
            X = training_data[entry].values
            X = X.reshape(-1, 1)  # INFO create list from row of values
            lowest_bic = np.infty
            bic = []
            n_components_range = range(1, self.settings["max_comp"])
            cluster_num = 0     # TODO used for testing purposes. Maybe have a look at this when implementing detection. maybe can be removed
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components)
                gmm.fit(X)
                bic.append(gmm.bic(X))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    clf = gmm  # INFO: = current lowest fit
                    cluster_num = n_components

            Y = clf.predict(X)
            training_data[entry + '_cluster'] = Y
            cluster_num = len(training_data[entry + '_cluster'].unique())  # INFO: number of different values in prediction,
            scores = clf.score_samples(X)
            score_threshold = scores.min() * self.settings["sigma"]
            training_data.drop(entry, axis=1, inplace=True)  # INFO remove [...]_update entries TODO used for testing purposes. Maybe have a look at this when implementing detection

    def event_driven_pred(self, training_data: pd.DataFrame, cont_vars: [], disc_vars: [], max_dict: {}, min_dict: {},
                          entry_trans_map: {}):
        print("Generating event-driven predicates...")
        invar_dict = {}  # INFO contains as key unrelated sensor values and as values intercept value +/- max_error, so that this value lies in [0,1]
        for entry in disc_vars:  # INFO iterate over the label of discrete states
            print('generate event-driven predicates for', entry)
            for roundi in range(0, len(entry_trans_map[entry])):
                trans = entry_trans_map[entry].pop()
                print('round: ' + str(roundi) + ' - shift: ' + trans)
                tempt_data = training_data.copy()
                tempt_data[entry] = 0

                tempt_data.loc[(tempt_data[entry + '_shift'] == trans), entry] = 99

                for target_var in cont_vars:
                    active_vars = list(cont_vars)
                    active_vars.remove(target_var)

                    # INFO (following) look at sensor values at time when actuator values have changed -> X. Look at the sensor value that your selected sensor has (target_var) -> Y
                    X = tempt_data.loc[tempt_data[entry] == 99, active_vars].values
                    Y = tempt_data.loc[tempt_data[entry] == 99, target_var].values

                    X_test = tempt_data[active_vars].values.astype(np.float)
                    Y_test = tempt_data[target_var].values.astype(np.float)

                    if len(Y) > 5:  # INFO case actuator changes and my current sensor (target_var) has at least 6 different states in that case
                        lgRegr = Lasso(alpha=1)

                        lgRegr.fit(X, Y)
                        y_pred = lgRegr.predict(X)

                        mae = metrics.mean_absolute_error(Y, y_pred)  # never used I guess...
                        dist = list(np.array(Y) - np.array(y_pred))
                        dist = map(abs, dist)  # INFO absolute distance between the values of Y and y_pred
                        max_error = max(dist)  # INFO max error/max distance of dist
                        mae_test = metrics.mean_absolute_error(Y_test, lgRegr.predict(X_test))

                        min_value = tempt_data.loc[tempt_data[entry] == 99, target_var].min()
                        max_value = tempt_data.loc[tempt_data[entry] == 99, target_var].max()
                        #                 print(target_var,max_error)
                        if max_error < self.settings["eps"]:
                            max_error = max_error * self.settings["sigma"]  # INFO to generate a little bit space between boundaries/to enable errors when checking sensor values ?
                            must = False
                            for coef in lgRegr.coef_:  # INFO if all coefs are 0, then the sensor only the current sensor (target_var) triggers the event
                                if coef > 0:
                                    must = True
                            if must == True:  # INFO case target_var is not unrelated
                                invar_entry = Util.conInvarEntry(target_var, lgRegr.intercept_ - max_error, '<',
                                                                 max_dict, min_dict, lgRegr.coef_,
                                                                 active_vars)  # INFO create label
                                training_data[invar_entry] = 0
                                training_data.loc[training_data[target_var] < lgRegr.intercept_ - max_error,
                                                  invar_entry] = 1  # INFO insert 1 whenever the current sensor has a lover vale than intercept - error

                                invar_entry = Util.conInvarEntry(target_var, lgRegr.intercept_ + max_error, '>',
                                                                 max_dict, min_dict, lgRegr.coef_, active_vars)
                                training_data[invar_entry] = 0
                                training_data.loc[training_data[target_var] > lgRegr.intercept_ + max_error,
                                                  invar_entry] = 1  # INFO same here but inverted
                            else:  # INFO case target_var is unrelated
                                if target_var not in invar_dict:
                                    invar_dict[target_var] = []
                                icpList = invar_dict[target_var]  # unused?

                                # INFO intercept is the point where function crosses y axis
                                if lgRegr.intercept_ - max_error > 0 and lgRegr.intercept_ - max_error < 1:
                                    invar_dict[target_var].append(lgRegr.intercept_ - max_error)

                                if lgRegr.intercept_ + max_error > 0 and lgRegr.intercept_ + max_error < 1:
                                    invar_dict[target_var].append(lgRegr.intercept_ + max_error)
            training_data.drop(entry + '_shift', axis=1, inplace=True)

        for target_var in invar_dict:  # INFO iterate over all sensors that are unrelated
            icpList = invar_dict[target_var]  # INFO icpList contains intercept +/- max error, and lies in [0,1]
            if icpList is not None and len(icpList) > 0:
                icpList.sort()
                if icpList is not None:  # not necessary?
                    for i in range(len(icpList) + 1):
                        if i == 0:
                            invar_entry = Util.conMarginEntry(target_var, icpList[0], 0, max_dict, min_dict)  # INFO create label
                            training_data[invar_entry] = 0
                            training_data.loc[training_data[target_var] < icpList[0], invar_entry] = 1  # INFO insert 1 whenever the current sensor is lower than icp_list value

                        elif i == len(icpList):
                            invar_entry = Util.conMarginEntry(target_var, icpList[i - 1], 1, max_dict, min_dict)
                            training_data[invar_entry] = 0
                            training_data.loc[training_data[target_var] >= icpList[i - 1], invar_entry] = 1

                        else:
                            invar_entry = Util.conRangeEntry(target_var, icpList[i - 1], icpList[i], max_dict,
                                                             min_dict)  # INFO create label
                            training_data[invar_entry] = 0
                            training_data.loc[(training_data[target_var] >= icpList[i - 1]) &
                                              (training_data[target_var] <= icpList[i]), invar_entry] = 1  # INFO insert 1 whenever the current sensor lies between two icpList values
        for var_c in cont_vars:
            training_data.drop(var_c, axis=1, inplace=True)

    def train(self, ipal=None, state=None):
        training_data = []

        print("Reading training data...")
        with self._open_file(state) as f:
            for line in f.readlines():
                curr = json.loads(line)
                training_data.append(curr["state"])
        training_data = pd.DataFrame.from_dict(training_data)
        training_data.reset_index(drop=True, inplace=True)

        print("Generating predicates...")
        # preparation for distribution driven
        cont_vars = []
        training_data.fillna(method='ffill', inplace=True)
        for entry in training_data:
            if training_data[entry].dtypes == np.float64:
                max_value = training_data[entry].max()
                min_value = training_data[entry].min()
                if max_value != min_value:
                    training_data[entry + '_update'] = training_data[entry].shift(-1) - training_data[entry]
                    cont_vars.append(entry + '_update')
        # remove last row, due to NaNs in _update entries
        training_data = training_data[:len(training_data)-1]

        self.distr_driven_pred(training_data, cont_vars)

        # saving intermediate results
        training_data.to_csv("./my_tests/data/swat_after_distribution_normal.csv", index=False)
        # training_data = pd.read_csv("./my_tests/data/swat_after_distribution_normal.csv")

        # preparation for event-driven
        cont_vars = []  # INFO = continuous variables. Here label of normalized values
        disc_vars = []  # INFO = discrete variables. Contains label of entries that are discrete in the original dataset
        entry_trans_map = {}    # INFO mapping the entry to a set of all transitions of states of the component

        max_dict = {}
        min_dict = {}

        onehot_entries = {} # what is its usecase? TODO: find out and remove it is not necessary...
        dead_entries = []  # INFO entries with only one value
        for entry in training_data:
            if entry.endswith('cluster') == True:  # INFO creation of predicates of GMM stuff
                newdf = pd.get_dummies(training_data[entry]).rename(columns=lambda x: entry + '=' + str(x))  # INFO one-hot encoding
                if len(newdf.columns.values.tolist()) <= 1:  # INFO case: only one value in list == only one GMM conponent responsible for these values
                    unique_value = training_data[entry].unique()[0]
                    dead_entries.append(entry + '=' + str(unique_value))
                    training_data = pd.concat([training_data, newdf], axis=1)  # INFO still added to training_data
                    training_data.drop(entry, axis=1, inplace=True)
                else:  # INFO case: one-hot encoding has more than one GMM encoding
                    onehot_entries[entry] = newdf.columns.values.tolist()
                    training_data = pd.concat([training_data, newdf], axis=1)
                    training_data.drop(entry, axis=1, inplace=True)
            else:  # INFO data that was not predicted by GMM
                if training_data[entry].dtypes == np.float64:  # INFO continuous values
                    max_value = training_data[entry].max()
                    min_value = training_data[entry].min()
                    if max_value == min_value:  # INFO drop values with only one value
                        training_data.drop(entry, axis=1, inplace=True)
                    else:
                        training_data[entry] = training_data[entry].apply(lambda x: float(x - min_value) / float(
                            max_value - min_value))  # INFO normalization of values to [0,1]
                        cont_vars.append(entry)
                        max_dict[entry] = max_value
                        min_dict[entry] = min_value
                else:  # INFO categorical values
                    newdf = pd.get_dummies(training_data[entry]).rename(columns=lambda x: entry + '=' + str(x))  # INFO one-hot encoding of categorical values
                    if len(newdf.columns.values.tolist()) <= 1:  # INFO case only one state
                        unique_value = training_data[entry].unique()[0]
                        dead_entries.append(entry + '=' + str(unique_value))
                        training_data = pd.concat([training_data, newdf], axis=1)  # INFO still added to training data...?
                        training_data.drop(entry, axis=1, inplace=True)
                    else:   # INFO more than one entry
                        # TODO: add predicate "not in state x" (entry!=x) in addition to "in state x" (entry=x)
                        disc_vars.append(entry)
                        trans = training_data[entry].shift(-1).fillna(method='ffill').astype(int).astype(str) + '->' + \
                                training_data[entry].astype(int).astype(str)
                        training_data[entry + '_shift'] = trans

                        # store only transitions into a different state different from the origin
                        trans = [x for x in set(trans.array) if x.split("->")[0] != x.split("->")[1]]
                        entry_trans_map[entry] = trans
                        training_data = pd.concat([training_data, newdf], axis=1)
                        training_data.drop(entry, axis=1, inplace=True)
                    #
                    #
                    #
                    # elif len(newdf.columns.values.tolist()) == 2:  # INFO case: exactly two entries
                    #     disc_vars.append(entry)
                    #     training_data[entry + '_shift'] = training_data[entry].shift(-1).fillna(method='ffill').astype(
                    #         int).astype(str) + '->' + training_data[entry].astype(int).astype(str)
                    #     onehot_entries[entry] = newdf.columns.values.tolist()   # INFO: add one-hot encoding of both states AND entry_shift with the transition of the states
                    #     training_data = pd.concat([training_data, newdf], axis=1)
                    #     training_data.drop(entry, axis=1, inplace=True)
                    # else:  # INFO case: more than two entries/states (only covers two states here)
                    #     disc_vars.append(entry)
                    #     training_data[entry + '_shift'] = training_data[entry].shift(-1).fillna(method='ffill').astype(
                    #         int).astype(str) + '->' + training_data[entry].astype(int).astype(str)
                    #
                    #     training_data[entry + '!=1'] = 1
                    #     training_data.loc[training_data[entry] == 1, entry + '!=1'] = 0
                    #
                    #     training_data[entry + '!=2'] = 1
                    #     training_data.loc[training_data[entry] == 2, entry + '!=2'] = 0
                    #     training_data.drop(entry, axis=1, inplace=True)

        self.event_driven_pred(training_data, cont_vars, disc_vars, max_dict, min_dict, entry_trans_map)

        # saving intermediate results
        training_data.to_csv("./my_tests/data/after_event_normal.csv", index=False)
        with open("./my_tests/data/dead_entries_after_event_normal.json", 'w') as f:
            json.dump(dead_entries, f)
        # reading intermediate results
        # training_data = pd.read_csv("./my_tests/data/after_event_normal.csv")
        # with open("./my_tests/data/dead_entries_after_event_normal.json", 'r') as f:
        #     dead_entries = json.load(f)

        print("Start rule mining...")
        print('Gamma=' + str(self.settings["gamma_value"]) + ', theta=' + str(self.settings["theta_value"]))
        # mode 0
        keyArray = self.settings["keyArray"]
        start_time = time.time()
        rule_list_0, item_dict_0 = Util.getRules(training_data, dead_entries, keyArray,
                                                 mode=0, gamma=self.settings["gamma_value"],
                                                 max_k=self.settings["max_k"], theta=self.settings["theta_value"])
        print('finish mode 0')
        print('mode 0 time cost: ' + str((time.time() - start_time) * 1.0 / 60))

        ##mode 2 is quite costly, use mode 1 if want to save time
        start_time_2 = time.time()
        rule_list_1, item_dict_1 = Util.getRules(training_data, dead_entries, keyArray,
                                                 mode=2, gamma=self.settings["gamma_value"],
                                                 max_k=self.settings["max_k"], theta=self.settings["theta_value"])
        print('finish mode 2')
        end_time = time.time()
        time_cost = (end_time - start_time) * 1.0 / 60
        print('mode 2 time cost: ' + str((end_time - start_time_2) * 1.0 / 60))
        print('rule mining time cost: ' + str(time_cost))

        rules = []
        for rule in rule_list_1:
            valid = False
            for item in rule[0]:
                if 'cluster' in item_dict_1[item]:
                    valid = True
                    break
            if valid == False:
                for item in rule[1]:
                    if 'cluster' in item_dict_1[item]:
                        valid = True
                        break
            if valid == True:
                rules.append(rule)
        rule_list_1 = rules
        print('rule count: ' + str(len(rule_list_0) + len(rule_list_1)))

        # arrange rules based on phase of testbed
        phase_dict = {}
        for i in range(1, len(keyArray) + 1):
            phase_dict[i] = []

        for rule in rule_list_0:
            strPrint = ''
            first = True
            for item in rule[0]:
                strPrint += item_dict_0[item] + ' and '
                if first == True:
                    first = False
                    for i in range(0, len(keyArray)):
                        for key in keyArray[i]:
                            if key in item_dict_0[item]:
                                phase = i + 1
                                break

            strPrint = strPrint[0:len(strPrint) - 4]
            strPrint += '---> '
            for item in rule[1]:
                strPrint += item_dict_0[item] + ' and '
            strPrint = strPrint[0:len(strPrint) - 4]
            phase_dict[phase].append(strPrint)

        for rule in rule_list_1:
            strPrint = ''
            first = True
            for item in rule[0]:
                strPrint += item_dict_1[item] + ' and '
                if first == True:
                    first = False
                    for i in range(0, 6):
                        for key in keyArray[i]:
                            if key in item_dict_1[item]:
                                phase = i + 1
                                break

            strPrint = strPrint[0:len(strPrint) - 4]
            strPrint += '---> '
            for item in rule[1]:
                strPrint += item_dict_1[item] + ' and '
            strPrint = strPrint[0:len(strPrint) - 4]
            phase_dict[phase].append(strPrint)

        invariance_file = "./my_tests/data/invariants/invariants_gamma=" + str(self.settings["gamma_value"]) + \
                          '&theta=' + str(self.settings["theta_value"]) + ".txt"
        with open(invariance_file, "w") as myfile:
            for i in range(1, len(keyArray) + 1):
                myfile.write('P' + str(i) + ':' + '\n')

                for rule in phase_dict[i]:
                    myfile.write(rule + '\n')
                    myfile.write('\n')

                myfile.write('--------------------------------------------------------------------------- ' + '\n')
            myfile.close()

    # TODO implement intrusion detection
    def new_state_msg(self, msg):
        pass

    # TODO implement model saving
    def save_trained_model(self):
        pass

    # TODO implement model loading
    def load_trained_model(self):
        pass
