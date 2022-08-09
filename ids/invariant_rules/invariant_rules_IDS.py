import ipal_iids.settings as settings
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
        "negated_states": 1,    # 1: create actuator predicates =x and !=x  -  0: only create actuator predicates =x

        # List of component identifiers. Separated in lists containing components of different parts. Here keyArray for SWaT
        "keyArray": [['FIT101','LIT101','MV101','P101','P102'], ['AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206'],
          ['DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302'], ['AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401'],
          ['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503'],['FIT601','P601','P602','P603']]
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._invariant_rules_default_settings)

        self.last_state = None  # used to calculate the change of a sensor value to previous message TODO maybe remove and let the transcriber calculate it. !!!BUT the original value can not be overwritten. Transcriber has to add a new key!!!
        self.sensors = []
        self.actuators = {}
        self.gmm_models = {}    # key: entry, values: (gmm component, score_threshold)
        self.rule_list = []

    def intermediate_model_saving(self, stage: str):
        model = {
            "_name": self._name,
            "settings": self.settings,
            "sensors": self.sensors,
            "actuators": self.actuators,
            "rules": self.rule_list,
            "gmm_keys": list(self.gmm_models.keys())  # makes loading the model easier
        }
        for entry in self.gmm_models:
            gauss = self.gmm_models[entry][0]
            model[entry] = {"gaussian": {
                "weights": gauss.weights_,
                "means": gauss.means_,
                "covariances": gauss.covariances_
            },
                "threshold": self.gmm_models[entry][1]}

        with self._open_file("./my_tests/intermediate_model_saving/{}.json".format(stage), mode='wt') as f:
            f.write(json.dumps(model, indent=4) + "\n")

    def restore_intermediate_model(self, stage: str):
        try:
            with self._open_file("./my_tests/intermediate_model_saving/{}.json".format(stage), mode='rt') as f:
                model = json.load(f)
        except FileNotFoundError:
            settings.logger.info("Model file {} not found.".format(str(self._resolve_model_file_path())))
            exit(1)

        assert self._name == model["_name"]
        self.settings = model["settings"]
        self.sensors = model["sensors"]
        self.actuators = model["actuators"]
        self.rule_list = model["rules"]
        for entry in model["gmm_keys"]:
            means = model[entry]["gaussian"]["means"]
            cov = model[entry]["gaussian"]["covariances"]
            gmm = mixture.GaussianMixture(n_components=len(means))
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
            gmm.weights_ = model[entry]["gaussian"]["weights"]
            gmm.means_ = means
            gmm.covariances_ = cov
            self.gmm_models[entry] = (gmm, model[entry]["threshold"])

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
            self.gmm_models[entry] = (clf, score_threshold)
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
                                print("--- DEBUG: related sensors: label {}".format(invar_entry))
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
                    self.sensors.append(entry)
        # remove last row, due to NaNs in _update entries
        training_data = training_data[:len(training_data)-1]

        self.distr_driven_pred(training_data, cont_vars)

        # saving intermediate results
        training_data.to_csv("./my_tests/data/swat_after_distribution_normal.csv", index=False)
        self.intermediate_model_saving("gaussian_mixture_models")
        # training_data = pd.read_csv("./my_tests/data/swat_after_distribution_normal.csv")
        # self.restore_intermediate_model("gaussian_mixture_models")

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
                        self.actuators[entry] = set(training_data[entry].values)
                        unique_value = training_data[entry].unique()[0]
                        dead_entries.append(entry + '=' + str(unique_value))
                        training_data = pd.concat([training_data, newdf], axis=1)  # INFO still added to training data...?
                        training_data.drop(entry, axis=1, inplace=True)
                    else:   # INFO more than one entry
                        disc_vars.append(entry)
                        self.actuators[entry] = set(training_data[entry].values)
                        trans = training_data[entry].shift(-1).fillna(method='ffill').astype(int).astype(str) + '->' + \
                                training_data[entry].astype(int).astype(str)
                        training_data[entry + '_shift'] = trans

                        # store only transitions into a different state different from the origin
                        trans = [x for x in set(trans.array) if x.split("->")[0] != x.split("->")[1]]
                        entry_trans_map[entry] = trans
                        training_data = pd.concat([training_data, newdf], axis=1)
                        # add all "not in state x" (entry!=x) predicates
                        if self.settings["negated_states"] == 1:
                            for v in set(training_data[entry].values):
                                training_data[entry + "!=" + str(v)] = 0
                                training_data.loc[training_data[entry] != v, entry + "!=" + str(v)] = 1
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
        self.intermediate_model_saving("gmm_and_regression")
        # reading intermediate results
        # training_data = pd.read_csv("./my_tests/data/after_event_normal.csv")
        # with open("./my_tests/data/dead_entries_after_event_normal.json", 'r') as f:
        #     dead_entries = json.load(f)
        # self.restore_intermediate_model("gmm_and_regression")

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

        # INFO I guess in the following, rules are filtered if they have no distribution-driven predicate. Remember this is done only for rules generated by mode 2
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

        self.rule_list = rule_list_0 + rule_list_1
        print('rule count: ' + str(len(self.rule_list)))

        # INFO the following is just for inspecting the rules afterwards by printing it to a file
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

    def new_state_msg(self, msg: {}):
        if not self.last_state:
            self.last_state = msg["state"]
            return False, {}
        states = msg["state"]

        # preprocess msg
        for s in states:
            if s in self.actuators["keys"]:
                # TODO: maybe check if state is allowed by checking if state ever occurred during training
                for x in self.actuators[s]:
                    if x != states[s]:
                        states[s + "!=" + str(x)] = 1
                        states[s + "=" + str(x)] = 0
                    else:
                        states[s + "!=" + str(x)] = 0
                        states[s + "=" + str(x)] = 1
                # TODO they check the amount of occurrences but this requires knowledge about all incoming messages (main.py line 175)
            elif s in self.sensors:
                # check corresponding gmm
                update = self.last_state[s] - states[s]
                gmm = self.gmm_models[s][0]
                pred = gmm.predict(update)
                states[s + "_update_cluster=" + str(pred)] = 1
                score = gmm.score_samples(states[s + "_update"])
                threshold = self.gmm_models[s][1]
                if score < threshold:   # does not belong to one of the mixture components of this sensor
                    return True, {"state": s, "alert": {"gmm_score": score, "threshold": threshold, "rule": "",
                                                        "msg": "sensor update never occurred during training"}}
            else:
                # TODO: how to handle components that have only one value and thus are dropped during training?
                return True, {"state": s, "alert": {"gmm_score": None, "threshold": None, "rule": "",
                                                    "msg": "component identifier never occurred during training"}}

        # check msg against rules
        for rule in self.rule_list:
            check = True
            # check if conclusion of rule has to be checked
            for pred in rule[0]:
                if "_cluster" in pred:
                    # sensor value, check cluster
                    if pred not in states:
                        check = False
                        break
                elif "=" or "!=" in pred:
                    # actuator, check states
                    if states[pred] == 0:
                        check = False
                        break
                elif "<" in pred:
                    # sensor value, check upper bound
                    pred = pred.split("<")
                    if len(pred) == 3:
                        # predicate defines upper and lower bound, check both
                        if states[pred[1]] <= pred[0] or states[pred[1]] >= pred[2]:
                            check = False
                            break
                    else:
                        if states[pred[0]] >= pred[1]:
                            check = False
                            break
                elif ">" in pred:
                    # sensor value, check lower bound
                    pred = pred.split(">")
                    if states[pred[0]] <= pred[1]:
                        check = False
                        break
                else:
                    settings.logger.critical("Invalid rule found. Premise {} invalid in rule {}".format(pred, rule))
                    check = False
                    break
            # check conclusion of rule
            if check:
                for pred in rule[1]:
                    if "_cluster" in pred:
                        # sensor value, check cluster
                        if pred not in states:
                            return True, {"state": pred.split("_update_cluster")[0], "alert": {"gmm_score": None,
                                                                                               "threshold": None,
                                                                                               "rule": rule,
                                                                                               "msg": "unsatisfied rule"}}
                    elif "=" or "!=" in pred:
                        # actuator value, check state
                        if states[pred] == 0:
                            return True, {"state": pred.split("=")[0], "alert": {"gmm_score": None,
                                                                                               "threshold": None,
                                                                                               "rule": rule,
                                                                                               "msg": "unsatisfied rule"}}
                    elif "<" in pred:
                        # sensor value, check upper bound
                        pred = pred.split("<")
                        if len(pred) == 3:
                            # predicate defines upper and lower bound, check both
                            if states[pred[1]] <= pred[0] or states[pred[1]] >= pred[2]:
                                return True, {"state": pred[1], "alert": {"gmm_score": None,
                                                                          "threshold": None,
                                                                          "rule": rule,
                                                                          "msg": "unsatisfied rule"}}
                        else:
                            if states[pred[0]] >= pred[1]:
                                return True, {"state": pred[0], "alert": {"gmm_score": None,
                                                                          "threshold": None,
                                                                          "rule": rule,
                                                                          "msg": "unsatisfied rule"}}
                    elif ">" in pred:
                        # sensor value, check lower bound
                        pred = pred.split(">")
                        if states[pred[0]] <= pred[1]:
                            return True, {"state": pred[0], "alert": {"gmm_score": None,
                                                                      "threshold": None,
                                                                      "rule": rule,
                                                                      "msg": "unsatisfied rule"}}
                    else:
                        settings.logger.critical("Invalid rule found. Conclusion {} invalid in rule {}".format(pred, rule))
                        break

            return False, {}

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
            "sensors": self.sensors,
            "actuators": self.actuators,
            "rules": self.rule_list,
            "gmm_keys": list(self.gmm_models.keys())    # makes loading the model easier
        }
        for entry in self.gmm_models:
            gauss = self.gmm_models[entry][0]
            model[entry] = {"gaussian": {
                                "weights": gauss.weights_,
                                "means": gauss.means_,
                                "covariances": gauss.covariances_
                            },
                            "threshold": self.gmm_models[entry][1]}

        with self._open_file(self._resolve_model_file_path(), mode='wt') as f:
            f.write(json.dumps(model, indent=4) + "\n")
        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:
            with self._open_file(self._resolve_model_file_path(), mode='rt') as f:
                model = json.load(f)
        except FileNotFoundError:
            settings.logger.info("Model file {} not found.".format(str(self._resolve_model_file_path())))
            return False

        assert self._name == model["_name"]
        self.settings = model["settings"]
        self.sensors = model["sensors"]
        self.actuators = model["actuators"]
        self.rule_list = model["rules"]
        for entry in model["gmm_keys"]:
            means = model[entry]["gaussian"]["means"]
            cov = model[entry]["gaussian"]["covariances"]
            gmm = mixture.GaussianMixture(n_components=len(means))
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
            gmm.weights_ = model[entry]["gaussian"]["weights"]
            gmm.means_ = means
            gmm.covariances_ = cov
            self.gmm_models[entry] = (gmm, model[entry]["threshold"])
        return True
