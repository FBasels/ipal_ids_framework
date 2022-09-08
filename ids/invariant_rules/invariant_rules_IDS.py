import ipal_iids.settings as settings
from ids.ids import MetaIDS
import pandas as pd
import numpy as np
from sklearn import mixture
from sklearn.linear_model import Lasso
import ids.invariant_rules.Util as Util
import time
import json


class InvariantRulesIDS(MetaIDS):
    _name = "InvariantRulesIDS"
    _description = "Invariant rule mining from training data"
    _requires = ["train.state", "live.state"]
    _invariant_rules_default_settings = {
        "eps": 0.01,  # same as in the paper
        "sigma": 1.1,  # buffer scaler
        "theta_value": 0.08,  # same as in the paper
        "gamma_value": 0.9,  # same as in the paper
        "max_k": 4,  # maximal length of items in frequent item sets
        "max_comp": 4,  # number of mixture components
        "negated_states": 0,    # 1: create actuator predicates =x and !=x  -  0: only create actuator predicates =x
        "merge_rules": 1,      # 1: combines rules with the same premises to one
        "parallel_filter": 0,   # 1: filter for closed patterns using multiprocessing  -  0: onyl use one process

        # List of component identifiers. Separated in lists containing components of different parts. Here keyArray for SWaT
        # keyArray: list of components in the testbed, seperated into different parts of the testbed
        # (optional) actuators: list of actuators in the testbed. Used to ensure that actuators are correctly handled
        ###
        # SWAT
        ###
        "keyArray": [['FIT101','LIT101','MV101','P101','P102'], ['AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206'],
          ['DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302'], ['AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401'],
          ['AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503'],['FIT601','P601','P602','P603']],
        "actuators": []
        ###
        # WADI
        ###
        # "keyArray": [['1_AIT_001_PV','1_AIT_002_PV','1_AIT_003_PV','1_AIT_004_PV','1_AIT_005_PV','1_FIT_001_PV','1_LS_001_AL','1_LS_002_AL','1_LT_001_PV','1_MV_001_STATUS',
        #       '1_MV_002_STATUS','1_MV_003_STATUS','1_MV_004_STATUS','1_P_001_STATUS','1_P_002_STATUS','1_P_003_STATUS','1_P_004_STATUS','1_P_005_STATUS','1_P_006_STATUS'],
        #     ['2_DPIT_001_PV','2_FIC_101_CO','2_FIC_101_PV','2_FIC_101_SP','2_FIC_201_CO','2_FIC_201_PV','2_FIC_201_SP','2_FIC_301_CO','2_FIC_301_PV','2_FIC_301_SP',
        #      '2_FIC_401_CO','2_FIC_401_PV','2_FIC_401_SP','2_FIC_501_CO','2_FIC_501_PV','2_FIC_501_SP','2_FIC_601_CO','2_FIC_601_PV','2_FIC_601_SP','2_FIT_001_PV','2_FIT_002_PV',
        #      '2_FIT_003_PV','2_FQ_101_PV','2_FQ_201_PV','2_FQ_301_PV','2_FQ_401_PV','2_FQ_501_PV','2_FQ_601_PV','2_LS_001_AL','2_LS_002_AL','2_LS_101_AH','2_LS_101_AL','2_LS_201_AH',
        #      '2_LS_201_AL','2_LS_301_AH','2_LS_301_AL','2_LS_401_AH','2_LS_401_AL','2_LS_501_AH','2_LS_501_AL','2_LS_601_AH','2_LS_601_AL','2_LT_001_PV','2_LT_002_PV','2_MCV_007_CO',
        #      '2_MCV_101_CO','2_MCV_201_CO','2_MCV_301_CO','2_MCV_401_CO','2_MCV_501_CO','2_MCV_601_CO','2_MV_001_STATUS','2_MV_002_STATUS','2_MV_003_STATUS','2_MV_004_STATUS','2_MV_005_STATUS',
        #      '2_MV_006_STATUS','2_MV_009_STATUS','2_MV_101_STATUS','2_MV_201_STATUS','2_MV_301_STATUS','2_MV_401_STATUS','2_MV_501_STATUS','2_MV_601_STATUS','2_P_001_STATUS',
        #      '2_P_002_STATUS','2_P_003_SPEED','2_P_003_STATUS','2_P_004_SPEED','2_P_004_STATUS','2_PIC_003_CO','2_PIC_003_PV','2_PIC_003_SP','2_PIT_001_PV','2_PIT_002_PV','2_PIT_003_PV',
        #      '2_SV_101_STATUS','2_SV_201_STATUS','2_SV_301_STATUS','2_SV_401_STATUS','2_SV_501_STATUS','2_SV_601_STATUS','2A_AIT_001_PV','2A_AIT_002_PV','2A_AIT_003_PV','2A_AIT_004_PV',
        #      '2B_AIT_001_PV','2B_AIT_002_PV','2B_AIT_003_PV','2B_AIT_004_PV'],
        #     ['3_AIT_001_PV','3_AIT_002_PV','3_AIT_003_PV','3_AIT_004_PV','3_AIT_005_PV','3_FIT_001_PV','3_LS_001_AL','3_LT_001_PV','3_MV_001_STATUS','3_MV_002_STATUS','3_MV_003_STATUS',
        #      '3_P_001_STATUS','3_P_002_STATUS','3_P_003_STATUS','3_P_004_STATUS','LEAK_DIFF_PRESSURE','PLANT_START_STOP_LOG','TOTAL_CONS_REQUIRED_FLOW']],
        # "actuators": ['1_LS_001_AL', '1_LS_002_AL', '1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS',
        #               '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS',
        #               '1_P_005_STATUS', '1_P_006_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL',
        #               '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', '2_LS_501_AL',
        #               '2_LS_601_AH', '2_LS_601_AL', '2_MCV_007_CO', '2_MV_001_STATUS', '2_MV_002_STATUS',
        #               '2_MV_003_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_006_STATUS', '2_MV_009_STATUS',
        #               '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', '2_MV_501_STATUS',
        #               '2_MV_601_STATUS', '2_P_003_STATUS', '2_P_004_STATUS', '2_PIC_003_SP', '2_SV_101_STATUS',
        #               '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS', '2_SV_601_STATUS',
        #               '3_LS_001_AL', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS',
        #               '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'PLANT_START_STOP_LOG']
        ###
        # HAI
        ###
        # "keyArray": [['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022',
        #                'P1_FCV01D','P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01',
        #                'P1_FT01Z', 'P1_FT02', 'P1_FT02Z','P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01',
        #                'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02D','P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_PP01AD',
        #                'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR', 'P1_PP02D','P1_PP02R', 'P1_STSP', 'P1_TIT01', 'P1_TIT02'],
        #               ['P2_24Vdc', 'P2_ASD', 'P2_AutoGO', 'P2_CO_rpm', 'P2_Emerg', 'P2_HILout', 'P2_MSD', 'P2_ManualGO',
        #                'P2_OnOff',' P2_RTR', 'P2_SIT01', 'P2_SIT02', 'P2_TripEx', 'P2_VT01', 'P2_VTR01', 'P2_VTR02',
        #                'P2_VTR03', 'P2_VTR04','P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03'],
        #               ['P3_FIT01', 'P3_LCP01D', 'P3_LCV01D', 'P3_LH', 'P3_LIT01', 'P3_LL', 'P3_PIT01'],
        #               ['P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_HT_PS', 'P4_LD', 'P4_ST_FD', 'P4_ST_GOV', 'P4_ST_LD',
        #                'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']],
        # "actuators": ["P1_PP01AD", "P1_PP01AR", "P1_PP01BD", "P1_PP01BR", "P1_PP02D", "P1_PP02R", "P1_STSP", "P2_AutoGO",
        #     "P2_Emerg", "P2_ManualGO", "P2_OnOff", "P2_TripEx"]

    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._invariant_rules_default_settings)

        self.last_state = None  # used to calculate the change of a sensor value to previous message
        self.sensors = []
        self.actuators = {}     # key: actuator name, value: [states of actuator]
        self.gmm_models = {}    # key: entry, values: (gmm component, score_threshold)
        self.rule_list = []     # entries: [[premises], [conclusions]]

    """
    Stores the current model as json file. Useful for debugging
    stage: name of the file to store to
    """
    def saving_intermediate_model(self, stage: str):
        model = {
            "_name": self._name,
            "settings": self.settings,
            "sensors": self.sensors,
            "actuators": self.actuators,
            "rules": self.rule_list,
            "gmm_keys": list(self.gmm_models.keys())  # simplifies loading the model
        }
        for entry in self.gmm_models:
            gauss = self.gmm_models[entry][0]
            model[entry] = {"gaussian": {
                "weights": gauss.weights_.tolist(),
                "means": gauss.means_.tolist(),
                "covariances": gauss.covariances_.tolist()
            },
                "threshold": self.gmm_models[entry][1]}

        with self._open_file("./my_tests/data/intermediate_model_{}.json".format(stage), mode='wt') as f:
            f.write(json.dumps(model, indent=4) + "\n")

    """
    Restores the model from json file. Useful for debugging
    stage: name of the file to read from
    """
    def restore_intermediate_model(self, stage: str):
        try:
            with self._open_file("./my_tests/data/intermediate_model_{}.json".format(stage), mode='rt') as f:
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
            means = np.array(model[entry]["gaussian"]["means"])
            cov = np.array(model[entry]["gaussian"]["covariances"])
            gmm = mixture.GaussianMixture(n_components=len(means))
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
            gmm.weights_ = np.array(model[entry]["gaussian"]["weights"])
            gmm.means_ = means
            gmm.covariances_ = cov
            self.gmm_models[entry] = (gmm, model[entry]["threshold"])

    def distr_driven_pred(self, training_data: pd.DataFrame, cont_vars: []):
        settings.logger.info("Generating distribution-driven predicates...")
        for entry in cont_vars:
            settings.logger.debug("Generate distribution-driven predicates for {}".format(entry))
            X = training_data[entry].values
            X = X.reshape(-1, 1)
            lowest_bic = np.infty
            bic = []
            n_components_range = range(1, self.settings["max_comp"] + 1)
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components)
                gmm.fit(X)
                bic.append(gmm.bic(X))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    clf = gmm

            Y = clf.predict(X)
            training_data[entry + '_cluster'] = Y
            scores = clf.score_samples(X)
            score_threshold = scores.min() * self.settings["sigma"]
            self.gmm_models[entry] = (clf, score_threshold)
            training_data.drop(entry, axis=1, inplace=True)

    def event_driven_pred(self, training_data: pd.DataFrame, cont_vars: [], disc_vars: [], max_dict: {}, min_dict: {},
                          entry_trans_map: {}):
        settings.logger.info("Generating event-driven predicates...")
        invar_dict = {}
        for entry in disc_vars:
            settings.logger.debug("Generate event-driven predicates for {}".format(entry))
            for roundi in range(0, len(entry_trans_map[entry])):
                trans = entry_trans_map[entry].pop()
                settings.logger.debug("Round: {} - Shift: {}".format(roundi, trans))
                tempt_data = training_data.copy()
                tempt_data[entry] = 0

                tempt_data.loc[(tempt_data[entry + '_shift'] == trans), entry] = 99

                for target_var in cont_vars:
                    active_vars = list(cont_vars)
                    active_vars.remove(target_var)

                    X = tempt_data.loc[tempt_data[entry] == 99, active_vars].values
                    Y = tempt_data.loc[tempt_data[entry] == 99, target_var].values

                    if len(Y) > 5:
                        lgRegr = Lasso(alpha=1)

                        lgRegr.fit(X, Y)
                        y_pred = lgRegr.predict(X)

                        dist = list(np.array(Y) - np.array(y_pred))
                        dist = map(abs, dist)
                        max_error = max(dist)

                        if max_error < self.settings["eps"]:
                            max_error = max_error * self.settings["sigma"]  # enable additional small error
                            must = False
                            for coef in lgRegr.coef_:  # check if other sensors are related to the event
                                if coef > 0:
                                    must = True
                            if must == True:    # other sensors are related
                                invar_entry = Util.conInvarEntry(target_var, lgRegr.intercept_ - max_error, '<',
                                                                 max_dict, min_dict, lgRegr.coef_,
                                                                 active_vars)
                                settings.logger.debug("Related sensors: label {}".format(invar_entry))
                                training_data[invar_entry] = 0
                                training_data.loc[training_data[target_var] < lgRegr.intercept_ - max_error,
                                                  invar_entry] = 1

                                invar_entry = Util.conInvarEntry(target_var, lgRegr.intercept_ + max_error, '>',
                                                                 max_dict, min_dict, lgRegr.coef_, active_vars)
                                training_data[invar_entry] = 0
                                training_data.loc[training_data[target_var] > lgRegr.intercept_ + max_error,
                                                  invar_entry] = 1
                            else:  # event can be described by one sensor value
                                if target_var not in invar_dict:
                                    invar_dict[target_var] = []

                                if lgRegr.intercept_ - max_error > 0 and lgRegr.intercept_ - max_error < 1:
                                    invar_dict[target_var].append(lgRegr.intercept_ - max_error)

                                if lgRegr.intercept_ + max_error > 0 and lgRegr.intercept_ + max_error < 1:
                                    invar_dict[target_var].append(lgRegr.intercept_ + max_error)
            training_data.drop(entry + '_shift', axis=1, inplace=True)

        # created predicate based on if an event is related to only one sensor or to multiple
        for target_var in invar_dict:
            icpList = invar_dict[target_var]
            if icpList is not None and len(icpList) > 0:
                icpList.sort()
                for i in range(len(icpList) + 1):
                    if i == 0:
                        invar_entry = Util.conMarginEntry(target_var, icpList[0], 0, max_dict, min_dict)
                        training_data[invar_entry] = 0
                        training_data.loc[training_data[target_var] < icpList[0], invar_entry] = 1
                    elif i == len(icpList):
                        invar_entry = Util.conMarginEntry(target_var, icpList[i - 1], 1, max_dict, min_dict)
                        training_data[invar_entry] = 0
                        training_data.loc[training_data[target_var] >= icpList[i - 1], invar_entry] = 1
                    else:
                        invar_entry = Util.conRangeEntry(target_var, icpList[i - 1], icpList[i], max_dict,
                                                         min_dict)
                        training_data[invar_entry] = 0
                        training_data.loc[(training_data[target_var] >= icpList[i - 1]) &
                                          (training_data[target_var] <= icpList[i]), invar_entry] = 1
        for var_c in cont_vars:
            training_data.drop(var_c, axis=1, inplace=True)

    def train(self, ipal=None, state=None):
        training_data = []

        settings.logger.info("Reading training data...")
        with self._open_file(state) as f:
            for line in f.readlines():
                curr = json.loads(line)
                training_data.append(curr["state"])
        training_data = pd.DataFrame.from_dict(training_data)
        training_data.reset_index(drop=True, inplace=True)
        training_data.fillna(method='ffill', inplace=True)
        for entry in training_data:
            if training_data[entry].isnull().values.all():
                training_data.drop(entry, axis=1, inplace=True)

        settings.logger.info("Generating predicates...")
        # preparation for distribution-driven predicate generation and convert dtype of actuators to int64
        cont_vars = []
        convert = {}
        for entry in training_data:
            if training_data[entry].dtypes == np.float64:
                if entry in self.settings["actuators"]:
                    convert[entry] = np.int64
                    continue
                max_value = training_data[entry].max()
                min_value = training_data[entry].min()
                if max_value != min_value:
                    training_data[entry + '_update'] = training_data[entry].shift(-1) - training_data[entry]
                    cont_vars.append(entry + '_update')
                    self.sensors.append(entry)
            elif training_data[entry].dtypes == np.int64 and len(set(training_data[entry].values)) > 10:
                convert[entry] = np.float64
        training_data = training_data.astype(convert)

        # create '_update' entries for converted entries
        for entry in convert:
            if convert[entry] == np.float64:
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
        # training_data.to_csv("./my_tests/data/swat_after_distribution_normal.csv", index=False)
        # self.saving_intermediate_model("gaussian_mixture_models")
        # # restore intermediate results
        # training_data = pd.read_csv("./my_tests/data/swat_after_distribution_normal.csv")
        # self.restore_intermediate_model("gaussian_mixture_models")

        # preparation for event-driven and naming GMM and actuator predicates
        cont_vars = []
        disc_vars = []
        entry_trans_map = {}

        max_dict = {}
        min_dict = {}

        dead_entries = []
        for entry in training_data:
            if entry.endswith('cluster') == True:  # naming predicates for GMMs
                newdf = pd.get_dummies(training_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                if len(newdf.columns.values.tolist()) <= 1:     # filter of predicates that hold all the time
                    unique_value = training_data[entry].unique()[0]
                    dead_entries.append(entry + '=' + str(unique_value))
                    training_data = pd.concat([training_data, newdf], axis=1)
                    training_data.drop(entry, axis=1, inplace=True)
                else:
                    training_data = pd.concat([training_data, newdf], axis=1)
                    training_data.drop(entry, axis=1, inplace=True)
            else:
                if training_data[entry].dtypes == np.float64:   # preparation of sensor values
                    max_value = training_data[entry].max()
                    min_value = training_data[entry].min()
                    if max_value == min_value:  # filter components with static value during training
                        training_data.drop(entry, axis=1, inplace=True)
                    else:
                        training_data[entry] = training_data[entry].apply(lambda x: float(x - min_value) / float(
                            max_value - min_value))
                        cont_vars.append(entry)
                        max_dict[entry] = max_value
                        min_dict[entry] = min_value
                else:   # naming predicates for actuators
                    newdf = pd.get_dummies(training_data[entry]).rename(columns=lambda x: entry + '=' + str(x))
                    if len(newdf.columns.values.tolist()) <= 1:  # filter of predicates that hold all the time
                        self.actuators[entry] = [int(x) for x in set(training_data[entry].values)]
                        unique_value = training_data[entry].unique()[0]
                        dead_entries.append(entry + '=' + str(unique_value))
                        training_data = pd.concat([training_data, newdf], axis=1)
                        training_data.drop(entry, axis=1, inplace=True)
                    else:
                        disc_vars.append(entry)
                        self.actuators[entry] = [int(x) for x in set(training_data[entry].values)]
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

        self.event_driven_pred(training_data, cont_vars, disc_vars, max_dict, min_dict, entry_trans_map)

        # saving intermediate results
        # training_data.to_csv("./my_tests/data/after_event_normal.csv", index=False)
        # with open("./my_tests/data/dead_entries_after_event_normal.json", 'w') as f:
        #     json.dump(dead_entries, f)
        # self.saving_intermediate_model("gmm_and_regression")
        # # restoring intermediate results
        # training_data = pd.read_csv("./my_tests/data/after_event_normal.csv")
        # with open("./my_tests/data/dead_entries_after_event_normal.json", 'r') as f:
        #     dead_entries = json.load(f)
        # self.restore_intermediate_model("gmm_and_regression")

        settings.logger.info("Start rule mining...")
        settings.logger.info('Gamma=' + str(self.settings["gamma_value"]) + ', theta=' + str(self.settings["theta_value"]))
        # mode 0
        keyArray = self.settings["keyArray"]
        start_time = time.time()
        rule_list_0, item_dict_0 = Util.getRules(training_data, dead_entries, keyArray,
                                                 mode=0, gamma=self.settings["gamma_value"],
                                                 max_k=self.settings["max_k"], theta=self.settings["theta_value"])
        settings.logger.debug('finish mode 0')
        settings.logger.debug('mode 0 time cost: ' + str((time.time() - start_time) * 1.0 / 60))

        # mode 2 is quite costly, use mode 1 if you want to save time
        start_time_2 = time.time()
        rule_list_1, item_dict_1 = Util.getRules(training_data, dead_entries, keyArray,
                                                 mode=2, gamma=self.settings["gamma_value"],
                                                 max_k=self.settings["max_k"], theta=self.settings["theta_value"],
                                                 parallel_filter=(self.settings["parallel_filter"] == 1))
        settings.logger.debug('finish mode 2')
        end_time = time.time()
        time_cost = (end_time - start_time) * 1.0 / 60
        settings.logger.debug('mode 2 time cost: ' + str((end_time - start_time_2) * 1.0 / 60))
        settings.logger.info('rule mining time cost: ' + str(time_cost))

        # filter for rules that contain one distribution-driven predicate, others are covered by rule_list_0
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

        settings.logger.info('rule count: ' + str(len(rule_list_0) + len(rule_list_1)))

        # arrange rules based on phase of testbed and prepare for printing
        # phase_dict = {}
        # for i in range(1, len(keyArray) + 1):
        #     phase_dict[i] = []
        #
        # for rule in rule_list_0:
        #     strPrint = ''
        #     first = True
        #     for item in rule[0]:
        #         strPrint += item_dict_0[item] + ' and '
        #         if first == True:
        #             first = False
        #             for i in range(0, len(keyArray)):
        #                 for key in keyArray[i]:
        #                     if key in item_dict_0[item]:
        #                         phase = i + 1
        #                         break
        #
        #     strPrint = strPrint[0:len(strPrint) - 4]
        #     strPrint += '---> '
        #     for item in rule[1]:
        #         strPrint += item_dict_0[item] + ' and '
        #     strPrint = strPrint[0:len(strPrint) - 4]
        #     phase_dict[phase].append(strPrint)
        #
        # for rule in rule_list_1:
        #     strPrint = ''
        #     first = True
        #     for item in rule[0]:
        #         strPrint += item_dict_1[item] + ' and '
        #         if first == True:
        #             first = False
        #             for i in range(0, len(keyArray)):
        #                 for key in keyArray[i]:
        #                     if key in item_dict_1[item]:
        #                         phase = i + 1
        #                         break
        #
        #     strPrint = strPrint[0:len(strPrint) - 4]
        #     strPrint += '---> '
        #     for item in rule[1]:
        #         strPrint += item_dict_1[item] + ' and '
        #     strPrint = strPrint[0:len(strPrint) - 4]
        #     phase_dict[phase].append(strPrint)
        #
        # invariance_file = "./my_tests/data/invariants/invariants_gamma=" + str(self.settings["gamma_value"]) + \
        #                   '&theta=' + str(self.settings["theta_value"]) + ".txt"
        # with open(invariance_file, "w") as myfile:
        #     for i in range(1, len(keyArray) + 1):
        #         myfile.write('P' + str(i) + ':' + '\n')
        #
        #         for rule in phase_dict[i]:
        #             myfile.write(rule + '\n')
        #             myfile.write('\n')
        #
        #         myfile.write('--------------------------------------------------------------------------- ' + '\n')
        #     myfile.close()

        all_rules = []
        for rule in rule_list_0:
            prem = []
            conc = []
            for pre in rule[0]:
                prem.append(item_dict_0[pre])
            for con in rule[1]:
                conc.append(item_dict_0[con])
            all_rules.append((prem, conc))

        for rule in rule_list_1:
            prem = []
            conc = []
            for pre in rule[0]:
                prem.append(item_dict_1[pre])
            for con in rule[1]:
                conc.append(item_dict_1[con])
            all_rules.append((prem, conc))

        if self.settings["merge_rules"] == 1:
            rules = {}
            for rule in all_rules:
                prem = []
                conc = []
                for pre in rule[0]:
                    prem.append(pre)
                prem.sort()
                prem = tuple(prem)
                if prem not in rules:
                    rules[prem] = set()
                for con in rule[1]:
                    conc.append(con)
                rules[prem].update(conc)
            for prem in rules.keys():
                self.rule_list.append((prem, list(rules[prem])))
            settings.logger.info("Reduced rule number to {}".format(len(self.rule_list)))
        else:
            self.rule_list = all_rules

    def new_state_msg(self, msg: {}):
        settings.logger.debug("Current state message {}".format(msg))
        if not self.last_state:
            self.last_state = msg["state"]
            return False, {}
        states = dict(msg["state"])

        # preprocess msg
        for s in msg["state"]:
            if s in self.actuators:
                if states[s] not in self.actuators[s]:
                    self.last_state = msg["state"]
                    return True, {"state": s, "alert": {"gmm_score": None, "threshold": None, "rule": "",
                                                        "msg": "actuator state never occurred during training"}}
                for x in self.actuators[s]:
                    if x != states[s]:
                        states[s + "!=" + str(x)] = 1
                        states[s + "=" + str(x)] = 0
                    else:
                        states[s + "!=" + str(x)] = 0
                        states[s + "=" + str(x)] = 1
            elif s in self.sensors:
                # check corresponding gmm
                update = np.array(self.last_state[s] - states[s]).reshape(-1, 1)
                gmm = self.gmm_models[s + "_update"][0]
                pred = gmm.predict(update)
                states[s + "_update_cluster=" + str(pred[0])] = 1
                score = gmm.score_samples(update)
                threshold = self.gmm_models[s + "_update"][1]
                if score < threshold:   # does not belong to one of the mixture components of this sensor
                    self.last_state = msg["state"]
                    return True, {"state": s, "alert": {"gmm_score": score[0], "threshold": threshold, "rule": "",
                                                        "msg": "sensor update never occurred during training"}}
            else:
                # static components were removed during training. Check if this component is static
                if self.last_state[s] and states[s] and self.last_state[s] != states[s]:
                    settings.logger.info("State contains change component which was static during training")
                    self.last_state = msg["state"]
                    return True, {"state": s, "alert": {"gmm_score": None, "threshold": None, "rule": "",
                                                        "msg": "static component changed"}}
        self.last_state = msg["state"]

        # check msg against rules
        for rule in self.rule_list:
            check = True
            # check if conclusion of rule has to be checked
            for pred in rule[0]:
                if "_cluster" in pred:  # sensor value, check cluster
                    if pred not in states:
                        check = False
                        break
                elif "=" in pred or "!=" in pred:   # actuator, check states
                    if states[pred] == 0:
                        check = False
                        break
                elif "<" in pred:   # sensor value, check upper bound
                    pred = pred.split("<")
                    if len(pred) == 3:  # predicate defines upper and lower bound, check both
                        if states[pred[1]] <= float(pred[0]) or states[pred[1]] >= float(pred[2]):
                            check = False
                            break
                    else:
                        if states[pred[0]] >= float(pred[1]):
                            check = False
                            break
                elif ">" in pred:   # sensor value, check lower bound
                    pred = pred.split(">")
                    if states[pred[0]] <= float(pred[1]):
                        check = False
                        break
                else:
                    settings.logger.critical("Invalid rule found. Premise {} invalid in rule {}".format(pred, rule))
                    check = False
                    break
            # check conclusion of rule
            if check:
                for pred in rule[1]:
                    if "_cluster" in pred:  # sensor value, check cluster
                        if pred not in states:
                            return True, {"state": pred.split("_update_cluster")[0], "alert": {"gmm_score": None,
                                                                                               "threshold": None,
                                                                                               "rule": rule,
                                                                                               "msg": "unsatisfied rule"}}
                    elif "=" in pred or "!=" in pred:   # actuator value, check state
                        if states[pred] == 0:
                            return True, {"state": pred.split("=")[0], "alert": {"gmm_score": None,
                                                                                               "threshold": None,
                                                                                               "rule": rule,
                                                                                               "msg": "unsatisfied rule"}}
                    elif "<" in pred:   # sensor value, check upper bound
                        pred = pred.split("<")
                        if len(pred) == 3:  # predicate defines upper and lower bound, check both
                            if states[pred[1]] <= float(pred[0]) or states[pred[1]] >= float(pred[2]):
                                return True, {"state": pred[1], "alert": {"gmm_score": None,
                                                                          "threshold": None,
                                                                          "rule": rule,
                                                                          "msg": "unsatisfied rule"}}
                        else:
                            if states[pred[0]] >= float(pred[1]):
                                return True, {"state": pred[0], "alert": {"gmm_score": None,
                                                                          "threshold": None,
                                                                          "rule": rule,
                                                                          "msg": "unsatisfied rule"}}
                    elif ">" in pred:   # sensor value, check lower bound
                        pred = pred.split(">")
                        if states[pred[0]] <= float(pred[1]):
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
            "gmm_keys": list(self.gmm_models.keys())    # simplifies loading the model
        }
        for entry in self.gmm_models:
            gauss = self.gmm_models[entry][0]
            model[entry] = {"gaussian": {
                                "weights": gauss.weights_.tolist(),
                                "means": gauss.means_.tolist(),
                                "covariances": gauss.covariances_.tolist()
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
            means = np.array(model[entry]["gaussian"]["means"])
            cov = np.array(model[entry]["gaussian"]["covariances"])
            gmm = mixture.GaussianMixture(n_components=len(means))
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
            gmm.weights_ = np.array(model[entry]["gaussian"]["weights"])
            gmm.means_ = means
            gmm.covariances_ = cov
            self.gmm_models[entry] = (gmm, model[entry]["threshold"])
        return True
