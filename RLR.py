"""
This is my implementation of RLR  

Author: Zhao Minghao
"""


# -*- coding: utf-8 -*-
import csv
import random
import networkx as nx
import copy
import os

if __name__ == "__main__":
    fileName = ''
    ori_path = ''
    with open(ori_path,'r') as ori_file:
        G = nx.read_edgelist(ori_file, delimiter=',', nodetype = int)
    ori_edges = set([(min(each[0], each[1]), max(each[0], each[1])) for each in G.edges()])
    n = len(G.nodes())
    num = #number of deleted/added links
    for i in range(10):
        train_path = '' 
        with open(train_path, 'r') as train_file:
            train_G = nx.read_edgelist(train_file, delimiter=',', nodetype = int)
        test_path = ''
        with open(test_path, 'r') as test_file:
            test_G = nx.read_edgelist(test_file, delimiter=',', nodetype = int)   
        test_edges = [(min(each[0], each[1]),max(each[0], each[1])) for each in test_G.edges()]
        test_edges = set(test_edges)
        for j in range(100):
            newG = copy.deepcopy(train_G)
            index = 0
            index_add = 0
            while index < num:
                del_candidate = random.choice(newG.edges())             
                if newG.degree(del_candidate[0]) > 1 and newG.degree(del_candidate[1]) > 1:
                    newG.remove_edge(del_candidate[0], del_candidate[1])
                    index += 1              
            while index_add < num:
                k = random.randint(1,n-1)
                l = random.randint((k+1),n)
                add_candidate = (k,l)
                newG_edges = set([(min(each[0], each[1]), max(each[0], each[1])) for each in newG.edges()])
                if add_candidate not in ori_edges and add_candidate not in newG_edges:
                    newG.add_edge(add_candidate[0], add_candidate[1])
                    index_add += 1
            path = ''
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)            
            with open(path, 'wb') as new_file:
                csvWriter = csv.writer(new_file)
                csvWriter.writerows(newG.edges())
    print 'ends'