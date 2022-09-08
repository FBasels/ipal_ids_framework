'''
Created on 4 Sep 2017

@author: cf1510
'''
import time

import numpy as np
import ids.invariant_rules.RuleMiningUtil.MISTree as MISTree
import ids.invariant_rules.RuleMiningUtil.RuleGenerator as RuleGenerator
import ipal_iids.settings as settings

"""
    returns name of the predicate of the form:
    sensor_x [>|<] coef*other_sensor + coef*other_sensor + threshold_value
"""
def conInvarEntry(target_var, threshold, lessOrGreater, max_dict, min_dict, coefs, active_vars):
    threshold_value = threshold*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    msg = ''
    msg += target_var + lessOrGreater 
    count = 0
    for i in range(len(coefs)):
        if(coefs[i] > 0):
            msg += ' ' + str(coefs[i]) + '*' + active_vars[i]
            count += 1
    if count > 0:
        msg += '+' 
    msg += str(threshold_value)
    return msg


"""
    returns name of predicate of the form:
    sensor [>|<] threshold_value
"""
def conMarginEntry(target_var, threshold, margin, max_dict, min_dict):
    threshold_value = threshold*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var]  
    
    msg = ''
    if margin == 0:
        msg +=  target_var + '<' + str(threshold_value)
    else:
        msg +=  target_var + '>' + str(threshold_value)
    
    return msg   


"""
    returns name of predicate of the form:
    threshold_lower_bound < sensor < threshold_upper_bound
"""
def conRangeEntry(target_var, lb, ub, max_dict, min_dict):
    threshold_lb = lb*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    threshold_up = ub*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    
    msg = ''
    msg += str(threshold_lb) + '<' + target_var + '<' + str(threshold_up)
    
    return msg


""" 
        :training_data: contains the predicates generated in main.py before
        :dead_entries: contains predicates with only one value, which will be deleted
        :keyArray: contains the keys of the different parts of the SWaT testbed
        :mode: in {0,1,any-other-number}. 0 removes distribution-driven predicates, 1 only users distribution-driven predicates, any-other-number uses all predicates
        :gamma: as in paper, in (0,1), to define the requirement of the support of an invariant rule, to be larger than that (larger than gamma*min_sup of every item)
        :max_k: maximal length of frequent item sets
        :theta: as in paper, in (0, gamma), defines minimum fraction of samples in datalog. Very rare items are excluded
"""
def getRules(training_data, dead_entries, keyArray, mode=0, gamma=0.4, max_k=4, theta=0.1, parallel_filter=False):
    data = training_data.copy()
    data = data.astype(np.int64)
    # drop entries with only one value
    for entry in dead_entries:
        data.drop(entry, axis=1, inplace=True)
    
    for entry in data:
        if mode == 0:
            if 'cluster' in entry:
                data.drop(entry, axis=1, inplace=True)
        elif mode == 1:
            if 'cluster' not in entry:
                data.drop(entry, axis=1, inplace=True)

    index_dict = {}
    item_dict = {}
    minSup_dict = {}
    index = 100
    for entry in data:
        index_dict[entry] = index
        item_dict[index] = entry
        index += 1
    min_num = len(data)*theta
    for entry in data:
        minSup_dict[index_dict[entry]] = max(gamma*len(data[data[entry] == 1]), min_num)
        data.loc[data[entry] == 1, entry] = index_dict[entry]
    df_list = data.values.tolist()
    dataset = []
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)

    settings.logger.debug("calculating occurrences of every predicate")
    item_count_dict = MISTree.count_items(dataset)

    settings.logger.debug("generating MISTree...")
    s = time.time()
    root, MIN_freq_item_header_table, MIN, MIN_freq_item_header_dict = MISTree.genMIS_tree(dataset, item_count_dict, minSup_dict)
    settings.logger.debug("Finished building tree in {} minutes".format((time.time() - s)*1.0/60))

    settings.logger.debug("Starting CFP growth algorithm...")
    s = time.time()
    freq_patterns, support_data = MISTree.CFP_growth(root, MIN_freq_item_header_table, minSup_dict, max_k)
    settings.logger.debug("Finished CFP growth in {} minutes".format((time.time() - s)*1.0/60))

    settings.logger.debug("Filtering closed Patterns...")
    s = time.time()
    if parallel_filter:
        L = RuleGenerator.filterClosedPatternsParallel(freq_patterns, support_data, item_count_dict, max_k, MIN)
    else:
        L = RuleGenerator.filterClosedPatternsSeq(freq_patterns, support_data, item_count_dict, max_k, MIN)
    settings.logger.debug("Finished filtering closed patterns in {} minutes".format((time.time() - s)*1.0/60))


    settings.logger.debug("Generating Rules from closed patterns")
    s = time.time()
    rules = RuleGenerator.generateRules(L, support_data, MIN_freq_item_header_dict, minSup_dict, min_confidence=1)
    settings.logger.debug("Finished rule generation in {} minutes".format((time.time() - s)*1.0/60))
    
    valid_rules = []
    for rule in rules:
        valid = True
        for i in range(len(keyArray)):
            for key in keyArray[i]:
                belongAnteq = False
                belongConseq = False
                for item in rule[0]:
                    if key in item_dict[item]:
                        belongAnteq = True
                        break
                for item in rule[1]:
                    if key in item_dict[item]:
                        belongConseq = True
                        break
                if belongAnteq == True and belongConseq == True:
                    valid = False
                    break
        if valid == True:
            valid_rules.append(rule)
    
    return valid_rules, item_dict