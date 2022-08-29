'''
Created on 30 Aug 2017

@author: cf1510
'''

from multiprocessing import Process, Pipe
import numpy as np
import ipal_iids.settings as settings


def subfilter(LPrev, LNext, support_data, pipe, split=False):
    if split and (len(LPrev) > 100000 or len(LNext) > 1000000):
        try:
            closedLI = []
            # setting up subprocesses
            LPrev = np.array_split(LPrev, 4)
            proc = []
            pipes = []
            for i in range(0, 3):
                proc.append(None)
                pipes.append(None)
            for i in range(1, 4):
                rec, send = Pipe()
                pipes[i - 1] = rec
                proc[i - 1] = Process(target=subfilter, args=(frozenset(LPrev[i - 1]), LNext, support_data, send))
                proc[i - 1].start()
            # calculate
            for childSet in frozenset(LPrev[0]):    # do not ask why we use the first and not the last element of the array, although the last should be the smallest. It was faster everytime we tried it that way...
                valid = True
                for parentSet in LNext:
                    if childSet.issubset(parentSet) and support_data[childSet] == support_data[parentSet]:
                        valid = False
                        break
                if valid == True:
                    closedLI.append(childSet)
            # collecting results from subprocesses
            for i in range(0, len(pipes)):
                rec = pipes[i].recv()
                if type(rec) is list:
                    closedLI.extend(rec)
                else:
                    pipe.send(rec)
                    exit()
            pipe.send(closedLI)
        except Exception as e:
            pipe.send(e)
    else:
        try:
            closedLI = []
            for childSet in LPrev:
                valid = True
                for parentSet in LNext:
                    if childSet.issubset(parentSet) and support_data[childSet] == support_data[parentSet]:
                        valid = False
                        break
                if valid == True:
                    closedLI.append(childSet)
            pipe.send(closedLI)
        except Exception as e:
            pipe.send(e)


"""
    Filter freq_patterns for closed frequent patterns. Making use of multiprocessing
"""
def filterClosedPatternsParallel(freq_patterns, support_data, item_count_dict, max_k, MIN):
    L = []
    for i in range(max_k + 1):
        L.append([])

    for item in item_count_dict:
        if item_count_dict[item] >= MIN:
            key = frozenset([item])
            support_data[key] = item_count_dict[item]
            L[0].append(key)

    for pattern in freq_patterns:
        L[len(pattern) - 1].append(frozenset(pattern))

    proc = []
    pipes = []
    for i in range(0, len(L) - 1):
        proc.append(None)
        pipes.append(None)
    closedL = []
    for i in range(0, len(L) - 1):
        rec, send = Pipe()
        pipes[i] = rec
        proc[i] = Process(target=subfilter, args=(L[i], L[i + 1], support_data, send, True))
        proc[i].start()
        settings.logger.debug("Startet process {} with pid {} ".format(i, proc[i].pid))
    for i in range(0, len(pipes)):
        rec = pipes[i].recv()
        settings.logger.debug("Successfully read from process {}".format(proc[i].pid))
        if type(rec) is list:
            closedL.append(rec)
        else:
            settings.logger.critical("Subprocess crashed and send back: {}".format(rec))
    closedL.append(L[len(L) - 1])
    return closedL


"""
    Filter freq_patterns for closed frequent patterns.
"""
def filterClosedPatternsSeq(freq_patterns, support_data, item_count_dict, max_k, MIN):
    L = []
    for i in range(max_k+1):
        L.append([])
      
    for item in item_count_dict:
        if item_count_dict[item] >= MIN:
            key =frozenset([item])
            support_data[key] = item_count_dict[item]
            L[0].append(key)
    
    for pattern in freq_patterns:
        L[len(pattern)-1].append( frozenset(pattern) )
    
    closedL = []
    for i in range(0, len(L)-1):
        closedLI = []
        LPrev = L[i]
        LNext = L[i+1]
        for childSet in LPrev:
            valid = True
            for parentSet in LNext:
                if childSet.issubset(parentSet) and support_data[childSet] == support_data[parentSet]:
                    valid = False
                    break
            if valid == True:
                closedLI.append(childSet)
        closedL.append(closedLI)
    closedL.append(L[len(L)-1]) 
    return closedL 


"""
    Create the association rules
    L: list of frequent item sets
    support_data: support data for those itemsets
    min_confidence: minimum confidence threshold
"""
def generateRules(L, support_data, MIN_freq_item_header_table,min_sup,  min_confidence=1):
    rules = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
#             print "freqSet", freqSet, 'H1', H1
            if (i > 1):
                rules_from_conseq(freqSet, H1, support_data, rules,MIN_freq_item_header_table,min_sup, min_confidence)
            else:
                calc_confidence(freqSet, H1, support_data, rules,MIN_freq_item_header_table,min_sup, min_confidence)
    return rules


def calculateSupportCount(MIN_freq_item_header_table,itemlist,min_sup):
    item_mis_tuples = []
    for item in itemlist:
        im_tuple = (item, min_sup[item])
        item_mis_tuples.append(im_tuple)
        
    item_mis_tuples.sort(key=lambda tup: (tup[1],tup[0]))
    
    count = 0
    entry = MIN_freq_item_header_table[  item_mis_tuples[0][0] ]
    node = entry.node_link
    while node != None:
        i=1
        parent = node.parent_link
        while parent.parent_link != None and i<len(item_mis_tuples):
            if parent.item == item_mis_tuples[i][0]:
                i+=1
            parent = parent.parent_link
        
#         print 'i:' + str(i) + ' len:' + str( len(item_mis_tuples) )
        
        if i == len(item_mis_tuples):
            count += node.count
        
        node = node.node_link 
                     
    return count 


def calc_confidence(freqSet, H, support_data, rules, MIN_freq_item_header_table,min_sup, min_confidence=1):
    "Evaluate the rule generated"
    pruned_H = []
    for conseq in H:
        if freqSet-conseq not in support_data:
            itemlist = list(freqSet - conseq)
            support_data[freqSet - conseq] = calculateSupportCount(MIN_freq_item_header_table,itemlist,min_sup)
        conf = support_data[freqSet]*1.0 / support_data[freqSet - conseq]
        if conf >= min_confidence:
            rules.append((freqSet - conseq, conseq, conf))
            pruned_H.append(conseq)
    return pruned_H


"""
    Generate the joint transactions from candidate sets
"""
def aprioriGen(freq_sets, k):
    retList = []
    lenLk = len(freq_sets)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(freq_sets[i])[:k - 2]
            L2 = list(freq_sets[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(freq_sets[i] | freq_sets[j])
    return retList


"""
    Generate a set of candidate rules
"""
def rules_from_conseq(freqSet, H, support_data, rules, MIN_freq_item_header_table,min_sup, min_confidence=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calc_confidence(freqSet, Hmp1,  support_data, rules,MIN_freq_item_header_table,min_sup, min_confidence)
        if len(Hmp1) > 1:
            rules_from_conseq(freqSet, Hmp1, support_data, rules,MIN_freq_item_header_table,min_sup, min_confidence)