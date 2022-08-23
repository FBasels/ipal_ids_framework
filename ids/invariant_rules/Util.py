'''
Created on 4 Sep 2017

@author: cf1510
'''
import numpy as np
from sklearn.metrics import confusion_matrix
import ids.invariant_rules.RuleMiningUtil.MISTree as MISTree
import ids.invariant_rules.RuleMiningUtil.RuleGenerator as RuleGenerator
import ipal_iids.settings as settings


def evaluate_prediction(actual_result,predict_result, verbose = 1):
    cmatrix = confusion_matrix(actual_result, predict_result)
    precision = cmatrix[1][1]*1.0/(cmatrix[1][1]+cmatrix[0][1])
    recall = cmatrix[1][1]*1.0/(cmatrix[1][1]+cmatrix[1][0])
    f1score = 2*precision*recall/(precision+recall)    
    accuracy = (cmatrix[1][1]+cmatrix[0][0])*1.0/(cmatrix[1][1]+cmatrix[0][1]+cmatrix[0][0]+cmatrix[1][0])
    FPR = cmatrix[0][1]*1.0/(cmatrix[0][1]+cmatrix[0][0])
    if(verbose == 1):
        print( 'actual 1: ' + str(actual_result.count(1)) +'  0: '+ str(actual_result.count(0)) )
        print( 'predict 1: ' + str(predict_result.count(1)) +'  0: '+ str(predict_result.count(0)) )
        print( 'precision: ' + str(precision) )
        print( 'recall: ' + str(recall) )
        print( 'f1score: ' + str(f1score) )
        print( 'accuracy: ' + str(accuracy) )
        print( 'TPR: ' + str(recall) )
        print( 'FPR: ' + str(FPR) )
    return precision, recall, f1score, accuracy, FPR

# INFO should return label with sensor_x >/< coef*other_sensor + coef*other_sensor + threshold_value
def conInvarEntry(target_var, threshold, lessOrGreater, max_dict, min_dict, coefs, active_vars):    # TODO I do not know if this gives back a correct msg...
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


# INFO should return label with sensor >/< threshold_value
def conMarginEntry(target_var, threshold, margin, max_dict, min_dict):
    threshold_value = threshold*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var]  
    
    msg = ''
    if margin == 0:
        msg +=  target_var + '<' + str(threshold_value)
    else:
        msg +=  target_var + '>' + str(threshold_value)
    
    return msg   


# INFO should return label with threshold_lower_bound < sensor < threshold_upper_bound
def conRangeEntry(target_var, lb, ub, max_dict, min_dict):
    threshold_lb = lb*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    threshold_up = ub*(max_dict[target_var]-min_dict[target_var]) + min_dict[target_var] 
    
    msg = ''
    msg += str(threshold_lb) + '<' + target_var + '<' + str(threshold_up)
    
    return msg

# TODO check description of max_k
""" 
        :training_data: contains the predicates generated in main.py before
        :dead_entries: contains predicates with only one value, which will be deleted
        :keyArray: contains the keys of the different parts of the SWaT testbed
        :mode: in {0,1,any-other-number}. 0 removes distribution-driven predicates, 1 only users distribution-driven predicates, any-other-number uses all predicates
        :gamma: as in paper, in (0,1), to define the requirement of the support of an invariant rule, to be larger than that (larger than gamma*min_sup of every item)
        :max_k: (guess) maximal length of code patterns 
        :theta: as in paper, in (0, gamma), defines minimum fraction of samples in datalog. Very rare items are excluded
"""
def getRules(training_data, dead_entries, keyArray, mode=0, gamma=0.4, max_k=4, theta=0.1):
    data = training_data.copy()
    data = data.astype(np.int64)
#     print(len(data))
    'drop dead entries'
    for entry in dead_entries:
        data.drop(entry, axis=1, inplace=True)
    
    for entry in data:
        if mode == 0:
            if 'cluster' in entry:
                data.drop(entry, axis=1, inplace=True)
        elif mode == 1:
            if 'cluster' not in entry:
                data.drop(entry, axis=1, inplace=True)
        
    # INFO idex_dict and item_dict are used to resolve an index to a specific item and vice versa. Like an id for every predicate
    index_dict = {}
    item_dict = {}
    minSup_dict = {}    # INFO dict containing the minimum support for each predicate/minimal number of occurrences*gamma or min_num. !KEY IS HERE THE INDEX
    index = 100
    for entry in data:
        index_dict[entry] = index
        item_dict[index] = entry
        index += 1
    # print index_dict
    min_num = len(data)*theta   # INFO number of minimal occurrences of predicates
#     print 'min_num: ' + str(min_num)
    for entry in data:
        minSup_dict[ index_dict[entry]  ] = max( gamma*len(data[data[entry] == 1]), min_num )
    #     minSup_dict[ index_dict[entry]  ] = 100
#         print entry + ': ' + str(len(data[data[entry] == 1])*1.0)
        data.loc[data[entry] == 1, entry] = index_dict[entry]  # INFO from now on, data contains the index of the entry whenever the predicate is fullfilled
    df_list = data.values.tolist()  # INFO list of rows, not list of columns! Will represent which predicates (index) follow each other
    dataset = []    # INFO list of predicates which hold. Predicates (index) that do not hold (value 0) are filtered
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)

    settings.logger.debug("calculating occurrences of every predicate")
    item_count_dict = MISTree.count_items(dataset)  # INFO dict that gives for every predicate (index) the number of occurrences

    """
        :root: the MISTree
        :MIN_freq_item_header_table: table containing all the minimal frequencies and the corresponding indexes (sorted)
        :MIN: the overall minimum support value of all entries
        :MIN_freq_item_header_dict: dict containing all entries of the tree
    """
    settings.logger.debug("generating MISTree...")
    root, MIN_freq_item_header_table, MIN, MIN_freq_item_header_dict = MISTree.genMIS_tree(dataset, item_count_dict, minSup_dict)

    """
        :freq_patterns: list of frequent patterns
        :support_data:  dics of count of nodes in MIS Tree, keys =
    """
    settings.logger.debug("Starting CFP growth algorithm")
    freq_patterns, support_data = MISTree.CFP_growth(root, MIN_freq_item_header_table, minSup_dict, max_k)
    settings.logger.debug("Filtering closed Patterns")
    L = RuleGenerator.filterClosedPatterns(freq_patterns, support_data, item_count_dict, max_k, MIN)

    settings.logger.debug("Generating Rules from closed patterns")
    rules = RuleGenerator.generateRules(L, support_data, MIN_freq_item_header_dict, minSup_dict, min_confidence=1)
    
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