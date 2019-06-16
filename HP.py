"""
This is my implementation of HP  

Author: Zhao Minghao
"""

# -*- coding: utf-8 -*-
import networkx as nx
import csv
import copy
import os

def attack(G,edges,fuc):
    attackList = list()
    preds = fuc(G, edges)
    dic = dict()
    for u,v,p in preds:
        dic[(u,v)] = p
    predList = sorted(dic.iteritems(), key = lambda asd:asd[1], reverse = True)
    for each in predList:
        attackList.append(each[0])
    return attackList

def attack_strategy(attackList, m, G):
    del_set = set()
    add_set = set()
    index = 0
    while True:
        if index >= len(attackList):
            print 'pos_overflow'
            break
        elif attackList[index] in G.edges():
            if len(del_set) >= m:
                index += 1 
                continue
            else:
                del_set.add(attackList[index])
                G.remove_edge(attackList[index][0], attackList[index][1])      
                index += 1
        elif attackList[index] in test_edges:
            if len(del_set) >= m:
                index += 1 
                continue
            else:
                neighbors = nx.common_neighbors(G, attackList[index][0], attackList[index][1])
                degDict = dict()
                for node in neighbors:
                    degDict[node] = G.degree(node)
                try:
                    y = sorted(degDict.iteritems(), key = lambda asd : asd[1], reverse = False)[0][0]              
                except:
                    index += 1
                    continue
                if len(G.neighbors(attackList[index][0])) > len(G.neighbors(attackList[index][1])):
                    x = attackList[index][0] 
                else:
                    x = attackList[index][1]
                del_set.add((min(x, y), max(x, y)))
                G.remove_edge(x, y)              
                index += 1
                continue
            if len(add_set) >= m:
                index += 1
                continue
            else:
                neighbors = nx.common_neighbors(G, attackList[index][0], attackList[index][1])
                degDict = dict()
                for node in neighbors:
                    degDict[node] = G.degree(node)
                try:
                    degree_order = sorted(degDict.iteritems(), key = lambda asd : asd[1], reverse = False)
                    y = degree_order[0][0]
                    y_ = degree_order[1][0]
                except:
                    index += 1
                    continue
                if (min(y,y_),max(y,y_)) in nonexist_edges:
                    del_set.add((min(y, y_), max(y, y_)))
                    G.add_edge(y, y_)
                    index += 1
                else:
                    index += 1
        else:
            if len(add_set) >= m:
                index += 1
                continue
            else:
                neighbors_x = set(G.neighbors(attackList[index][0]))
                neighbors_y = set(G.neighbors(attackList[index][1]))
                neighbors = (neighbors_x | neighbors_y) - set([attackList[index][0], attackList[index][1]]) - (neighbors_x & neighbors_y)
                degDict = dict()
                for node in neighbors:
                    degDict[node] = G.degree(node)
                if len(neighbors) == 0:
                    index += 1
                else:
                    flag = False
                    degOrder = sorted(degDict.iteritems(), key = lambda asd : asd[1], reverse = False)  
                    for i in range(len(degDict)):
                        y = degOrder[i][0]
                        if y in neighbors_x:
                            x = attackList[index][1]
                        else:
                            x = attackList[index][0]
                        if (min(x,y),max(x,y)) not in test_edges:
                            add_set.add((min(x, y),max(x, y)))
                            G.add_edge(x, y)
                            index += 1
                            flag = True
                            break
                    if flag == False:
                        index += 1
        if len(del_set) == m and len(add_set) == m:
            break
    return del_set, add_set




if __name__ == "__main__":
    fileName = ''
    ori_path = ''
    with open(ori_path,'r') as ori_file:
        ori_G = nx.read_edgelist(ori_file, delimiter=',', nodetype = int)
    n = len(ori_G.nodes())    
    complete_edges = [(x,y) for x in range(1, n) for y in range(x+1, n+1)]
    for i in range(10):
        train_path = ''
        with open(train_path,'r') as train_file:
            train_G = nx.read_edgelist(train_file, delimiter=',', nodetype = int)
        train_G.add_nodes_from(range(1,n+1))
        test_path = ''
        with open(test_path,'r') as test_file:
            test_G = nx.read_edgelist(test_file, delimiter=',', nodetype = int)            
        train_edges = set(train_G.edges())
        global test_edges, nonexist_edges
        test_edges = set([(min(each[0], each[1]), max(each[0], each[1])) for each in test_G.edges()])
        ori_G_edges = [(min(each[0], each[1]), max(each[0], each[1])) for each in ori_G.edges()]
        nonexist_edges = set(complete_edges) - set(ori_G_edges)
        G = copy.deepcopy(train_G)
        m = #number of deleted/added links 
        fuc = nx.resource_allocation_index
        attackList = attack(G, complete_edges, fuc)
        del_set,  add_set = attack_strategy(attackList, m, G)
        new_edges = (train_edges - del_set) | add_set
        path = ''
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)        
        with open(path, 'wb') as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerows(new_edges)            
    print 'ends'
    
    
    