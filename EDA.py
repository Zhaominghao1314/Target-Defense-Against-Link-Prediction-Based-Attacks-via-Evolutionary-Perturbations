# -*- coding: utf-8 -*-
"""
This is my implementation of EDA  

Author: Zhao Minghao
"""

#-*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
import math
import time
import copy
from itertools import combinations
import bisect
import csv

class perturb:
    def __init__(self, del_edges, add_edges):
        self.del_edges = del_edges
        self.add_edges = add_edges
        self.edges = (set(train_edges) - self.del_edges) | self.add_edges

class weight_sample:
    def __init__(self, items, weights, times, flag):
        self.items = items
        self.total = sum(weights)
        self.acc = list(self.accumulate(weights))
        self.times = times
        self.flag = flag
    def accumulate(self, weights):
        cur = 0
        for w in weights:
            cur += w
            yield cur
    def __call__(self):
        if self.flag:
            result = set()
            while len(result) < self.times:
                result.add(self.items[bisect.bisect_left(self.acc, random.uniform(0, self.total))])
            return result
        else:
            result = list()
            while len(result) < self.times:
                result.append(self.items[bisect.bisect_left(self.acc, random.uniform(0, self.total))])
            if self.times > 1:
                return result            
            else:    
                return result[0]

def selection(pop, fitness, num_selected, flag):
    return weight_sample(pop, fitness, num_selected, flag)() 
   
def init_generation(pop_size, num_change):
    pop = list()
    for i in range(pop_size):
        del_edges = set()
        add_edges = set()
        while len(del_edges) < num_change:
            del_edges.add(random.choice(train_edges))
        while len(add_edges) < num_change:
            add_edges.add(random.choice(nonexist_edges))
        pop.append(perturb(del_edges, add_edges))
    return pop
 
def perturb_recalc_delete(train_G, link, recalc):
    testList = set()
    neighborhood_x = sorted(train_G.neighbors(link[0]))
    neighborhood_y = sorted(train_G.neighbors(link[1]))
    for each in combinations(neighborhood_x, 2):
        if each not in train_edges_set:
            if each not in recalc:
                testList.add(each)
                recalc.add(each)
    for each in combinations(neighborhood_y, 2):
        if each not in train_edges_set:
            if each not in recalc:
                testList.add(each)
                recalc.add(each)
    return testList
    
def perturb_recalc_add(train_G, link, recalc): 
    testList = set()
    neighborhood_x = sorted(train_G.neighbors(link[0]))
    neighborhood_y = sorted(train_G.neighbors(link[1]))
    for each in combinations(neighborhood_x, 2):
        if each not in train_edges_set:
            if each not in recalc:
                testList.add(each)
                recalc.add(each)
    for each in combinations(neighborhood_y, 2):
        if each not in train_edges_set:
            if each not in recalc:
                testList.add(each)
                recalc.add(each)
    for each in neighborhood_x:
        min_node = min(each, link[1])
        max_node = max(each, link[1])
        if (min_node, max_node) not in train_edges_set:
            if (min_node, max_node) not in recalc:
                testList.add((min_node, max_node))
                recalc.add((min_node, max_node))
    for each in neighborhood_y:
        min_node = min(each, link[0])
        max_node = max(each, link[0])
        if (min_node, max_node) not in train_edges_set:
            if (min_node, max_node) not in recalc:
                testList.add((min_node, max_node))
                recalc.add((min_node, max_node))
    return testList

def new_increment_cal_obj_value(pop, calculated, fuc, neg_fitness):   
    result = list()    
    for each in pop:
        ordered_key = sorted(each.del_edges | each.add_edges)
        key = tuple(ordered_key)        
        if key in calculated:
            result.append(calculated[key])
        else: 
            temp = new_increment_link_prediction(each, fuc, neg_fitness) 
            result.append(temp)        
            calculated[key] = temp
    return result

def RA(G, ebunch):
    RA_dic = dict()
    for each in ebunch:
        a = sum(1.0 / len(G.neighbors(w)) for w in set(G.neighbors(each[0])) & set(G.neighbors(each[1])))#in nx.common_neighbors(G, each[0], each[1]))
        RA_dic[each] = a
    return RA_dic


def new_increment_link_prediction(perturb, fuc, neg_fitness):
    recalc = set()
    testList = set()
    for each in perturb.del_edges:
        if each not in init_recalc:
            init_recalc[each] = perturb_recalc_delete(train_G, each, recalc)
            testList |= init_recalc[each]
            
        else:    
            testList |= init_recalc[each]
    for each in perturb.add_edges:
        if each not in init_recalc:
            init_recalc[each] = perturb_recalc_add(train_G, each, recalc)
            testList |= init_recalc[each]
        else:
            testList |= init_recalc[each]
    perturb_G = copy.deepcopy(train_G)
    perturb_G.remove_edges_from(perturb.del_edges)
    perturb_G.add_edges_from(perturb.add_edges)
    test = testList | perturb.del_edges | test_edges_set
    perturb_G_nodes = set(perturb_G.nodes())
    if len(perturb_G_nodes) < n:
        for each in test:
            if each[0] not in perturb_G_nodes:
                perturb_G.add_node(each[0]) 
            if each[1] not in perturb_G_nodes:
                perturb_G.add_node(each[1])
    #preds = fuc(perturb_G, test)
    #preds_dic = dict()
    #for s, t, v in preds:
        #preds_dic[(s, t)] = v   
    preds_dic = RA(perturb_G, test)
    fitness = test_fuc(perturb, preds_dic, neg_fitness, testList)
    return fitness
    
def test_fuc(perturb, preds_dic, neg_fitness, testList):
    max_test = 0
    max_count = 0
    pos_fitness = 0
    for each in test_edges:
        pos_fitness += preds_dic[each]
        if preds_dic[each] > max_test:
            max_test = preds_dic[each]
    pos_fitness /= len(test_edges)
    for each in testList:#需要重新计算相似度的不存在的连边/the non-existent links which need to recalculate the similarity index 
        neg_fitness -= init_proximity_dic[each]
        neg_fitness += preds_dic[each]
        if preds_dic[each] >= max_test:
            max_count += 1
    for each in perturb.del_edges:
        neg_fitness += preds_dic[each]
        if preds_dic[each] >= max_test:
            max_count += 1
    for each in perturb.add_edges:
        neg_fitness -= init_proximity_dic[each]
    neg_fitness /= len(nonexist_edges)
    temp = testList | perturb.add_edges
    for each in decedent_order:
        if each[1] > max_test :
            if each[0] not in temp:
                max_count += 1
        else:    
            break
    fitness = pos_fitness - neg_fitness 
    fitness -= ratio*max_count
    return fitness 
      
def cal_fitness(obj_values):
    return map(lambda x: math.exp(-x), obj_values)

def best_elite(pop, obj_values):
    min_obj_values = min(obj_values)
    return min_obj_values 

def top_k_elite(pop, obj_values, k):
    array_obj_values = np.array(obj_values)
    top_k_index = np.argsort(-array_obj_values)[:k]
    return [pop[i] for i in top_k_index]
                         
def mutation(pop, num_mutation_pop, pm):
    sampled_pop = random.sample(pop, num_mutation_pop)
    new_pop = copy.deepcopy(sampled_pop)
    pop_unit_len = len(pop[0].del_edges)
    for i in range(len(new_pop)):
        del_edges = list(new_pop[i].del_edges)
        add_edges = list(new_pop[i].add_edges)
        random.shuffle(del_edges)
        random.shuffle(add_edges)
        for j in range(pop_unit_len):
            if random.random() < pm:
                del_edges.pop()
                add_edges.pop()
        new_pop[i].del_edges = set(del_edges)
        new_pop[i].add_edges = set(add_edges)
        while len(new_pop[i].del_edges) < pop_unit_len:
            new_pop[i].del_edges.add(random.choice(train_edges))
        while len(new_pop[i].add_edges) < pop_unit_len:
            new_pop[i].add_edges.add(random.choice(nonexist_edges))
    return new_pop

def eda(pop, fitness, num_eda_pop):
    selected_eda_pop = selection(pop, fitness, 3*num_eda_pop, 0)
    pop_unit_len = len(pop[0].del_edges)
    del_dict = dict()
    add_dict = dict()
    for each in selected_eda_pop:
        for del_edge in each.del_edges:
            if del_edge not in del_dict:
                del_dict[del_edge] = 1
            else :
                del_dict[del_edge] += 1
        for add_edge in each.add_edges:
            if add_edge not in add_dict:
                add_dict[add_edge] = 1
            else :
                add_dict[add_edge] += 1
    del_edges = list()
    del_weight = list()
    add_edges = list()
    add_weight = list()
    for each in del_dict:
        del_edges.append(each)
        del_weight.append(del_dict[each])
    for each in add_dict:
        add_edges.append(each)
        add_weight.append(add_dict[each])
    new_pop = list()
    for i in range(num_eda_pop):
        eda_del_edges = set(selection(del_edges, del_weight, pop_unit_len, 1))
        eda_add_edges = set(selection(add_edges, add_weight, pop_unit_len, 1))
        new_pop.append(perturb(eda_del_edges, eda_add_edges))
    return new_pop

def eda_creat_new_pop(pop, fitness, num_elite_pop, num_eda_pop, num_mutation_pop, pm):
    elite_pop = copy.deepcopy(top_k_elite(pop, fitness, num_elite_pop))
    eda_pop = eda(pop, fitness, num_eda_pop)
    mut_pop = mutation(eda_pop, num_mutation_pop, pm)
    new_pop = elite_pop + eda_pop + mut_pop
    return new_pop


def calc_similarity(perturb, fuc):
    train_G = nx.Graph(list(perturb.edges)) 
    train_G.add_nodes_from(range(1, n+1))
    testList = set(complete_edges) - set(perturb.edges)
    preds = fuc(train_G, testList)
    return preds

def calc_pre_and_auc(perturb, fuc = nx.resource_allocation_index):
    preds = calc_similarity(perturb, fuc = nx.resource_allocation_index)
    dic = dict()
    for s,t,v in preds:
        dic[(s,t)] = v
    topk_preds = sorted(dic.iteritems(), key = lambda x: x[1], reverse = True)
    raTest = [each[0] for each in topk_preds[:len(test_edges)]]
    items = set(raTest) & set(test_edges)
    precision = float(len(items)) / len(test_edges)
    edgesNotexit = set(complete_edges) - set(perturb.edges) - set(test_edges)
    auc = 0.0
    for k in test_edges:
       for l in edgesNotexit:
           if dic[k] > dic[l]:
               auc += 1
           if dic[k] == dic[l]:
               auc += 0.5
    auc /= len(test_edges)*len(edgesNotexit)        
    return precision, auc

             
if __name__ == "__main__":    
    fileName = 'jazz'
    start = time.time()
    ori_path = './dataset/'+fileName+'.csv'
    best = np.array([])
    eda_prd = np.array([])
    eda_auc = np.array([])
    eda_best = np.array([])
    start = time.time()
    for i in range(10):     
        with open(ori_path,'r') as ori_file:
            ori_G = nx.read_edgelist(ori_file, delimiter=',', nodetype = int)
        train_path = './dataset/'+fileName+'/'+fileName+'_train_'+str(i+1)+'.csv'
        with open(train_path, 'r') as train_file:
            train_G = nx.read_edgelist(train_file, delimiter=',', nodetype = int)
        test_path = './dataset/'+fileName+'/'+fileName+'_test_'+str(i+1)+'.csv'
        with open(test_path, 'r') as test_file:
            test_G = nx.read_edgelist(test_file, delimiter=',', nodetype = int)
        global n, complete_edges, train_edges, test_edges, nonexist_edges, ratio, init_recalc, decedent_order, init_proximity_dic, train_edges_set, nonexist_edges_set, test_edges_set 
        n = len(ori_G.nodes())
        train_G.add_nodes_from(range(1, n+1))
        ori_G_edges = [(min(each[0], each[1]), max(each[0], each[1])) for each in ori_G.edges()]
        train_edges = [(min(each[0], each[1]), max(each[0], each[1])) for each in train_G.edges()]
        train_edges_set = set(train_edges)
        test_edges =  [(min(each[0], each[1]), max(each[0], each[1])) for each in test_G.edges()]
        test_edges_set = set(test_edges)
        complete_edges = list(combinations(sorted(ori_G.nodes()), 2))
        nonexist_edges = list(set(complete_edges) - set(ori_G_edges))
        nonexist_edges_set = set(nonexist_edges)
        pop_init = init_generation(1, 0)
        ori_results = calc_pre_and_auc(pop_init[0])
        print 'ori pre and auc:', ori_results[0], ori_results[1]
        init_proximity_generator = calc_similarity(pop_init[0], nx.resource_allocation_index)
        init_proximity_dic = dict()
        for s, t, v in init_proximity_generator:
            init_proximity_dic[(s, t)] = v
        decedent_order = sorted(init_proximity_dic.iteritems(), key = lambda x: x[1], reverse = True)
        init_recalc = dict()
        neg_fitness = 0
        for each in nonexist_edges:
            neg_fitness += init_proximity_dic[each]
        ratio = 0#alpha in fitness
        pop_size = 100
        pm = 0.1
        iterations = 2000
        num_elite_pop = 5
        num_eda_pop = 50
        num_mutation_pop = 50
        num_change = 250
        fuc = nx.resource_allocation_index
        calculated = dict()
        index = 0
        for j in range(5):      
            bestpop_eda = list()
            pop = init_generation(pop_size, num_change)
            eda_pop = copy.deepcopy(pop)
            for k in range(iterations):
                eda_topk_obj_values = new_increment_cal_obj_value(eda_pop, calculated, fuc, neg_fitness)
                eda_fitness = cal_fitness(eda_topk_obj_values)
                eda_pop = eda_creat_new_pop(eda_pop, eda_fitness, num_elite_pop, num_eda_pop, num_mutation_pop, pm)
            eda_topk_obj_values = new_increment_cal_obj_value(eda_pop, calculated, fuc, neg_fitness)
            mini_eda = min(eda_topk_obj_values)
            for l in range(len(eda_topk_obj_values)):
                if eda_topk_obj_values[l] == mini_eda:
                    bestpop_eda.append(eda_pop[l])
            results = map(calc_pre_and_auc, bestpop_eda)
            eda_pop_pre = [each[0] for each in results]
            eda_pop_auc = [each[1] for each in results]
            eda_prd = np.append(eda_prd, sum(eda_pop_pre) / len(eda_pop_pre))
            eda_auc = np.append(eda_auc, sum(eda_pop_auc) / len(eda_pop_auc))
            for m in range(len(bestpop_eda)):
                index += 1
                with open('./elite/'+fileName+'/'+fileName+'_'+str(i+1)+'/'+fileName+'_'+str(index)+'.csv','wb') as attacked_file: 
                    writer = csv.writer(attacked_file)
                    writer.writerows(bestpop_eda[m].edges)                      
    print 'eda_prd, len', eda_prd.mean(), eda_prd.std(), len(eda_prd)
    print 'eda_auc, len', eda_auc.mean(), eda_auc.std(), len(eda_auc)  
    print 'all time:', time.time()-start


