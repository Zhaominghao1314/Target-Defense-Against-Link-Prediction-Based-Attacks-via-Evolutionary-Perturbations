"""
This is my implementation of RLS 

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
    for i in range(10): 
        train_path = '' 
        test_path = ''
        with open(train_path, 'r') as train_file:
            train_G = nx.read_edgelist(train_file, delimiter=',', nodetype = int)
        with open(test_path, 'r') as test_file:
            test_G = nx.read_edgelist(test_file, delimiter=',', nodetype = int)    
        for j in range(100):
            G = copy.deepcopy(train_G)
            num = #number of deleted/added links
            index = 0
            while index < num / 2:                
                nodeSet = set()
                candidate = random.sample(G.edges(),2)
                nodeSet.add(candidate[0][0])
                nodeSet.add(candidate[0][1])
                nodeSet.add(candidate[1][0])
                nodeSet.add(candidate[1][1])
                if len(nodeSet) < 4:
                    continue
                elif (min(candidate[0][0],candidate[1][0]),max(candidate[0][0],candidate[1][0])) not in G.edges() and \
                (min(candidate[0][0],candidate[1][1]),max(candidate[0][0],candidate[1][1])) not in G.edges() and \
                (min(candidate[0][1],candidate[1][0]),max(candidate[0][1],candidate[1][0])) not in G.edges() and \
                (min(candidate[0][1],candidate[1][1]),max(candidate[0][1],candidate[1][1])) not in G.edges() and \
                (min(candidate[0][0],candidate[1][1]),max(candidate[0][0],candidate[1][1])) not in test_G.edges() and \
                (min(candidate[0][1],candidate[1][0]),max(candidate[0][1],candidate[1][0])) not in test_G.edges() :
                    G.remove_edge(candidate[0][0],candidate[0][1])
                    G.remove_edge(candidate[1][0],candidate[1][1])
                    G.add_edge(candidate[0][0],candidate[1][1])
                    G.add_edge(candidate[0][1],candidate[1][0])
                    index += 1
            path = ''
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            with open(path,'wb') as file:
                csvWriter = csv.writer(file)
                csvWriter.writerows(G.edges())
    print 'ends'