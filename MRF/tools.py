from __future__ import annotations
import numpy as np
from scipy import stats
import json
import networkx as nx
import itertools
import csv
import matplotlib.pyplot as plt
import mpmath as mp
import math
from .cp_factor import Factor
import pandas as pd
import time
from .domain import Domain
import cupy as cp
from collections import Counter
from typing import Optional
import os
from tqdm import tqdm

mp.mp.dps = 1000

def check_FK(group_data, h_data, FK_col=-1):
    FK_list = []
    for group in group_data:
        FK_list.append(group[0][FK_col])
    FK_list = np.array(FK_list)

    # print(FK_list.shape)
    # print(h_data[:, 0].shape)
    return np.equal(FK_list, h_data[:, 0].flatten()).all()

def get_domain_by_attrs(dom_dict, columns):
    dom_dict = {attr: dom_dict[attr] for attr in dom_dict}
    dom_dict = {i: dom_dict[columns[i]] for i in range(len(columns))}
    domain = Domain(dom_dict, list(range(len(dom_dict))))
    return domain

def check_group_data_domain(group_data, domain):
    for attr in domain.attr_list:
        size = domain.dict[attr]['size']
        for group in group_data:
            if len(group) > 0:
                for record in group:
                    assert(record[attr] < size)
                assert((group >= 0).all())


def check_data_domain(data, domain):
    print('checking data domain')
    print('attr, attr size, actual min value, actual max value')
    assert(domain.attr_list == list(range(data.shape[1])))
    assert(data.dtype == int)
    for i in range(data.shape[1]):
        min_v = np.min(data[:, i])
        max_v = np.max(data[:, i])
        size = domain.dict[i]['size'] 
        print(i, size, min_v, max_v, end='')
        if min_v == 0 and max_v+1 == size:
            print('\t OK')
        else:
            print('\t -----------')

def assert_data_domain(data, domain):
    # print(data.shape[1], len(domain))
    # print(data[:5])
    assert(data.shape[1] == len(domain))
    for i in range(data.shape[1]):
        min_v = np.min(data[:, i])
        max_v = np.max(data[:, i])
        # print(i, min_v, max_v)
        size = domain.dict[i]['size'] 

        assert(min_v >= 0)
        assert(max_v < size)

# calculate/get mutual information from data
def get_mutual_info(MI_map, entropy_map, index_list, data, domain):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))

    if index_list not in MI_map:

        MI = -get_entropy(entropy_map, index_list, data, domain)
        for attr in index_list:
            MI += get_entropy(entropy_map, [attr], data, domain)

        MI_map[index_list] = MI

    return MI_map[index_list]

def get_marginal(data, domain, index_list, weights=None):
    bins = domain.project(index_list).edge()
    histogram, _ = np.histogramdd(data[:, index_list], bins=bins, weights=weights)
    return histogram

def get_marginal_fact(data, domain, index_list, to_cpu=True):
    histogram = get_marginal(data, domain, index_list)
    if to_cpu:
        fact = Factor(domain.project(index_list), histogram, np)
    else:
        fact = Factor(domain.project(index_list), histogram, cp)
    return fact

# calculate/get entropy from data
def get_entropy(entropy_map, index_list, data, domain):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))
    if index_list not in entropy_map:
        temp_domain = domain.project(index_list)
        bins = temp_domain.edge()
        size = temp_domain.size()

        if len(index_list) <= 14 and size < 1e7:
            histogram, _= np.histogramdd(data[:, index_list], bins=bins)
            histogram = histogram.flatten()
            entropy = stats.entropy(histogram)
        else:
            value, counts = np.unique(data[:, index_list], return_counts=True, axis=0)
            entropy = stats.entropy(counts)

        entropy_map[index_list] = entropy

    return entropy_map[index_list]


def get_histogram(index_list, data, domain, weights=None):
    temp_domain = domain.project(index_list)
    histogram, _ = np.histogramdd(data[:, index_list], bins=temp_domain.edge(), \
        weights=weights)
    return histogram

def string_2d_low_precision_array(array):
    string = ''
    for row in array:
        row_string = string_low_precision_array(row)
        string += row_string + '\n'
    return string

def string_low_precision_array(array):
    string = ['{:.3f}'.format(item) for item in array]
    string = ', '.join(string)
    return string

# accept only one latent varibale
def get_latent_weighted_histogram(index_list, data, domain, \
    weights, latent_variable_set):
    if tuple(sorted(index_list)) == tuple(sorted(list(latent_variable_set))):
        return np.sum(weights, axis=0)
    # print(index_list)
    # print(q.shape)
    # q[:, :, :] = 0
    # q[:, 0, 1] = 0.4
    # q[:, 1, 2] = 0.2

    temp_domain = domain.project(index_list)
    latent_domain = temp_domain.project(latent_variable_set)


    # merge latent variables of q and get the distribution of the required latent variable
    latent_var_start = min(latent_variable_set)
    axis = [var-latent_var_start+1 for var in latent_domain.attr_list]
    axis = tuple(set(range(1, len(weights.shape))) - set(axis))
    weights = np.sum(weights, axis=axis)
    # print('axis', axis) # debug

    # print('??', np.sum(q))
    # print('??', np.sum(weights))
    # debug_temp = 0

    
    final_histogram = np.zeros(shape=temp_domain.shape, dtype=float)
    # print(final_histogram.shape, weights.shape)

    ob_marginal = tuple(sorted(list(set(index_list) - latent_variable_set)))
    ob_domain = domain.project(ob_marginal)

    # print('ob_marginal', ob_marginal)
    # print(data[:, ob_marginal])
    # print(ob_domain)
    # print(ob_domain.edge())
    # print(ob_marginal)
    # print(np.unique(data[:, 7]))
    # print(np.unique(data[:, 9]))
    # print(np.sum(hist))
    
    # use weights to weight records and get their histogram
    latent_type_list = list(list(range(i)) for i in latent_domain.shape)
    # print(weights)

    # # debug
    # if index_list == (1, 18):
    #     print('get data 10')
    #     print(data[:10])
    #     print(weights[:10])
    #     print(weights[1000:1000+10])
    #     print(latent_type_list)
    for latent_type in itertools.product(*tuple(latent_type_list)):
        # print(latent_type)

        slc = [slice(None),]
        slc.extend(list(latent_type))
        slc = tuple(slc)
        histogram, _ = np.histogramdd(data[:, ob_marginal], bins=ob_domain.edge(), \
            weights=weights[slc])
        # print(slc)
        # print(weights[slc].shape)

        slc = [slice(None),] * len(ob_domain.shape)
        slc.extend(list(latent_type))
        slc = tuple(slc)
        final_histogram[slc] = histogram
        # print(slc)
        # print(histogram.shape)

        # if index_list == (1, 18):
        #     print(slc)
        #     print(latent_type, histogram)

    # if index_list == (1, 18):
    #     print(final_histogram)

    return final_histogram

# collect all the possible values for each attr and sort them by their ferquencies
def collect_domain(np_data, attr_list):
    print(attr_list, len(np_data))

    with open('./temp/attr_list.txt', 'w') as out_file:
        out_file.write(str(attr_list)+'\n')
    domain_dict = {}
    for col in range(len(attr_list)):
        attr = attr_list[col]
        values, cnts = np.unique(np_data[:, col], return_counts=True)
        values_cnts = [(values[i], cnts[i]) for i in range(len(values))]
        values_cnts.sort(key = lambda x: x[1], reverse=True)

        partial_total = sum([item[1] for item in values_cnts[:100]])

        print(col, attr, partial_total)
        print(values_cnts[:100])
        print('')

        domain_dict[attr] = values_cnts
    
    return domain_dict

def get_adaptive_domain(data):
    assert(data.dtype==int)
    dom_dict = {}
    for col in range(data.shape[1]):
        min_v = min(data[:, col])
        max_v = max(data[:, col])
        assert(min_v>=0)
        dom_dict[col] = {'size': max_v+1}
    dom = Domain(dom_dict, list(range(data.shape[1])))
    return dom

def random_data_TVD(data1, data2):
    dom = get_adaptive_domain(data1)
    print('dom:', dom)
    return random_TVD(data1, data2, dom)

def random_TVD(data1, data2, domain, k=3, n=100, normalize=False, marginal_list=None):
    assert(data1.shape[1] == len(domain))
    assert(data2.shape[1] == len(domain))

    if marginal_list is None:
        marginal_list = [marginal for marginal in itertools.combinations(domain.attr_list, k) ]
        np.random.shuffle(marginal_list)
        marginal_list = marginal_list[:n]
        
    mean_TVD = 0
    for marginal in marginal_list:
        hist1 = get_histogram(marginal, data1, domain)
        hist2 = get_histogram(marginal, data2, domain)
        marginal_TVD = get_TVD(hist1, hist2, normalize)
        mean_TVD += marginal_TVD
    return mean_TVD / len(marginal_list)

def attr_TVD(data1, data2, domain, normalize=False):
    assert(data1.shape[1] == len(domain))
    assert(data2.shape[1] == len(domain))
    for attr in range(len(domain)):
        marginal = (attr,)
        hist1 = get_histogram(marginal, data1, domain)
        hist2 = get_histogram(marginal, data2, domain)
        marginal_TVD = get_TVD(hist1, hist2, normalize)
        print('attr tvd: {}, {:.4f}'.format(attr, marginal_TVD))
    print('')


def triangulate(graph) -> nx.Graph:
    edges = set()
    G = nx.Graph(graph)

    nodes = sorted(graph.degree(), key=lambda x: x[1])
    for node, degree in nodes:
        local_complete_edges = set(itertools.combinations(G.neighbors(node), 2))
        edges |= local_complete_edges

        G.add_edges_from(local_complete_edges)
        G.remove_node(node)
    
    triangulated_graph = nx.Graph(graph)
    triangulated_graph.add_edges_from(edges)

    return triangulated_graph

# randomly round a prob array such that its summation equal to num
def random_round(prob, num, replace=True):
    # assert(len(prob.shape)==1)
    if np.sum(prob) == 0:
        prob += 1
    prob = prob * num/prob.sum()
    # print('prob', prob)
    frac, integral = np.modf(prob)
    integral = integral.astype(int)
    round_number = int(num - integral.sum())
    if frac.sum() == 0:
        return integral
    p = (frac/frac.sum()).flatten()

    # print('integral', integral, 'round_number', round_number)
    # print('frac', frac)

    if round_number > 0:
        # CRF should sample without replacement while MRF should not.
        # this is for keeping intra group structures
        # say, we are sampling a group of 2 records, we have p = [0.49, 0.49, 0.02]
        # sampling without replacement gives [0, 1] with a very high probability
        # while sampling with replacement gives [0, 0] p=0.25, [0, 1]/[1, 0], p=0.5, [1, 1] p=0.25
        # Apparently, we prefer [0, 1] as the group records instead of [0, 0], [1, 1]
        # Although this would detroy attribte correlations if you look into the records
        # For example, sampling a group of 3 records and we have p [0.49, 0.49, 0.02]
        # Sampling without replacemet gives [0, 1, 2] deterministically.

        index = np.random.choice(prob.size, round_number, p=p, replace=replace)
        unique, unique_counts = np.unique(index, return_counts=True)

        # print('unique', unique)
        # print('unique_counts', unique_counts)

        for i in range(len(unique)):
            idx = np.unravel_index(unique[i], prob.shape)
            integral[idx] += unique_counts[i]
    return integral

def expand_int_prob(int_prob, shuffle=True):
    if len(int_prob.shape) > 1:
        data = []
        for idx in np.ndindex(int_prob.shape):
            data.extend([idx,] * int_prob[idx])
        data = np.array(data)
        if shuffle:
            np.random.shuffle(data)
        return data
    else:
        data = np.repeat(np.arange(int_prob.size), int_prob)
        if shuffle:
            np.random.shuffle(data)
    return data

def generate_column_data(prob, num, replace=True):
    if (prob < 0).any():
        raise Exception(str(prob))
    if num < 0:
        raise Exception(str(prob)+' '+str(num))

    int_prob = random_round(prob, num, replace=replace)

    # This error may due to the excessive noise. Check log to confirm that.
    if (int_prob < 0).any():
        raise Exception(str(int_prob)+' '+str(prob)+' '+str(num))
    return expand_int_prob(int_prob)

def save_np_csv(array, attr_list, path):
    with open(path, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(attr_list)
        for line in array:
            writer.writerow(line)

def print_graph(G, path):
    plt.clf()
    nx.draw(G, with_labels=True, edge_color='b', node_color='g', node_size=20, font_size=4, width=0.5)
    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['savefig.dpi'] = 600
    # plt.show()
    plt.savefig(path)

def get_TVD_count(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2)) / 2

def get_TVD(hist1, hist2, normalize=False):
    temp = np.sum(hist1)
    if temp == 0:
        return 1
    if normalize:
        hist2 = hist2 * temp /np.sum(hist2)
    return float(get_TVD_count(hist1, hist2) / temp)

def get_normalized_TVD(hist1, hist2):
    hist2 = hist2 * np.sum(hist1)/np.sum(hist2)
    return get_TVD(hist1, hist2)

def get_normalized_TVD_count(hist1, hist2):
    hist2 = hist2 * np.sum(hist1)/np.sum(hist2)
    return get_TVD_count(hist1, hist2)

def split_array_uniformly(array, k):
    if k == 1:
        return [array,]

    flatten_array = array.flatten()
    res_array = np.zeros([k, len(flatten_array)], dtype=int)

    for i in range(len(flatten_array)):
        item = flatten_array[i]
        res_array[:, i] = int(flatten_array[i] / k)
        temp_sum = np.sum(res_array[:, i])

        temp_add = np.zeros(k, dtype=int)
        for j in range(item - temp_sum):
            temp_add[j] += 1
        np.random.shuffle(temp_add)
        # print(res_array[:, i], temp_sum, item)

        res_array[:, i] += temp_add

    res_list = [res_array[i].reshape(array.shape) for i in range(k)]
    
    return res_list

def erf_func(x):
    temp = 2.0/mp.sqrt(mp.pi)
    integral = mp.quad(lambda t: mp.exp(-t**2), [0, x])
    res = temp*integral
    return res

# ref: Data synthesis via Differentially Private Markov Random Field
def cal_privacy_budget(epsilon, error, delta):
    print('calculating privacy budget')
    start = 0
    end = epsilon

    def func(x):
        if x <= 0:
            return - 2*delta
        add1 = erf_func(math.sqrt(x)/2/math.sqrt(2) - epsilon/math.sqrt(2*x))
        add2 = erf_func(math.sqrt(x)/2/math.sqrt(2) + epsilon/math.sqrt(2*x))
        res = add1 + mp.exp(epsilon)*add2 - mp.exp(epsilon) + 1 - 2*delta
        # print(add1, add2)
        return res
    # return mp.findroot(func, start, tol=1e-30)

    # print(func(start), func(end))

    # gradient of func around its root is extemely small (maybe <= 1e-20 depending on epsilon)
    # which makes it is hard to set tol of mp.findroot and mp.mp.dps
    # we simply use binary search to ensure abs error of the root
    if func(start) > 0:
        start_geater = True
        if func(end) > 0:
            print('cant find root in given interval')
            exit(-1)
            return
    else:
        start_geater = False
        if func(end) < 0:
            print('cant find root in given interval')
            exit(-1)
            return
    
    while end - start > error:
        mid = (start + end)/2
        # print(mid)
        if func(mid) > 0:
            if start_geater:
                start = mid
            else:
                end = mid
        else:
            if start_geater:
                end = mid
            else:
                start = mid

    print((start + end)/2)
    return (start + end)/2

def get_privacy_budget(epsilon, delta=1e-5, tol=1e-10):

    budget = cal_privacy_budget(epsilon, tol, delta)

    return budget

def get_unnormalized_R_score(data, domain, edge, weights=None, cache_dict=None):
    edge = tuple(sorted(edge))

    if cache_dict is None:
        cache_dict = {}

    def get_fact(marginal, cache_dict):
        if marginal not in cache_dict:
            hist = get_marginal(data, domain, marginal, weights=weights)
            fact = Factor(domain.project(marginal), hist, np)
            cache_dict[edge] = fact
        return cache_dict[edge]

    fact1 = get_fact(edge, cache_dict)
    data_num = np.sum(fact1.values)
    
    edge_domain = domain.project(edge)
    fact2 = Factor.full(edge_domain, data_num, np)
    for attr in edge:
        attr_fact = get_fact((attr,), cache_dict)
        fact2 = attr_fact.expand(edge_domain) / data_num * fact2

    R_score = np.sum(np.abs(fact1.values - fact2.values)) / 2

    return R_score

# return h_data with q types in h_data 0-th col order
def concatenate_q_group(q, i_group_data, h_data_with_id, h_domain, type_first=False):
    undetermined_type = np.sum( q[q<0.9] > 1.0/np.prod(q.shape[1:]) )
    if undetermined_type > 0:
        print('warning: too many group types are undermined')
    print('undetermined_type ratio: {:.4f}'.format(undetermined_type/q.size))

    argmax_q = np.argmax(q.reshape((len(q), -1)), axis=1)
    argmax_q = np.unravel_index(argmax_q, shape=q.shape[1:])
    argmax_q = [item.reshape((-1, 1)) for item in argmax_q]
    argmax_q = np.concatenate(argmax_q, axis=1)

    # match household data and the type q
    i_group_FK = [group[0, -1] for group in i_group_data]
    h_to_q = {i_group_FK[i]: argmax_q[i] for i in range(len(i_group_FK))}

    q_h_data = np.zeros(shape=(h_data_with_id.shape[0], \
        h_data_with_id.shape[1]+argmax_q.shape[1]), dtype=int)
    q_h_data[:, :h_data_with_id.shape[1]] = h_data_with_id

    idx_list = []
    idx_list2 = []
    for idx in range(len(h_data_with_id)):
        h_id = h_data_with_id[idx][0]
        if h_id in h_to_q:
            q_h_data[idx, h_data_with_id.shape[1]:] = h_to_q[h_id]
            idx_list.append(idx)
        else:
            idx_list2.append(idx)

    q_h_data = q_h_data[idx_list]

    if len(q_h_data) < len(h_data_with_id):
        print('warning: missing h types of h_data: {} {}'.format(len(q_h_data), len(h_data_with_id)))

    if type_first:
        latent_var_num = len(q.shape) - 1

        q_h_data = np.concatenate([\
            q_h_data[:, [0,]], q_h_data[:, -latent_var_num:], q_h_data[:, 1:-latent_var_num]
            ], axis=1)

        temp_dict = {}
        attr = 0
        for q_size in q.shape[1:]:
            temp_dict[attr] = {'size': q_size}
            attr += 1
        for attr in h_domain.attr_list:
            temp_dict[attr+latent_var_num] = h_domain.dict[attr].copy()
        
        q_h_domain = domain.Domain(temp_dict, list(range(len(temp_dict))))
    else:
        q_h_domain = h_domain.copy()
        q_attr = len(h_domain)
        for q_size in q.shape[1:]:
            q_h_domain.add_variable(q_attr, q_size)
            q_attr += 1

    return argmax_q, q_h_data, h_data_with_id[idx_list2], q_h_domain

def get_group_data(np_data, group_id_attrs=[0,]):
    # print('get group data...')
    np_data = np_data[np.lexsort(np_data[:, group_id_attrs].T)]
    group_data_list = []
    data_len = len(np_data)
    i = 0

    with tqdm(total=data_len, desc="Grouping data", unit="rows") as pbar:
        while i < data_len:
            group = []
            row_id = np_data[i, group_id_attrs]

            while (np_data[i, group_id_attrs] == row_id).all():
                group.append(np_data[i])
                i += 1
                pbar.update(1)
                if i >= data_len:
                    break
            group = np.array(group)
            group_data_list.append(group)
    group_data_list = np.array(group_data_list, dtype=object)

    return group_data_list


def generate_group_id(histogram, data_len, id_array=None, shuffle=True):
    if id_array is None:
        id_array = np.arange(data_len)

    histogram = histogram / np.sum(histogram)
    max_size = len(histogram)
    assert(np.sum(histogram[1:]) > 0)

    id_data = np.full(shape=data_len, fill_value = -1, dtype=int)

    idx = 0
    group_idx = 0
    while idx < data_len:
        
        size = np.random.choice(max_size, p=histogram)

        size = min(size, data_len-idx)
        
        id_data[idx: idx+size] = id_array[group_idx]
        
        idx = idx + size
        group_idx += 1

    if shuffle:
        np.random.shuffle(id_data)

    return id_data
    
def get_q_h_data(q, i_group_data, h_data_with_id, h_domain):
    argmax_q, q_h_data, q_h_domain = concatenate_q_group(\
        q, i_group_data, h_data_with_id, h_domain)
    # print(h_data_with_id[:, 0].shape)
    # print(q_h_data.shape)
    # print(h_data_with_id.shape)

    return q_h_data, q_h_domain

def expand_group_data(group_data, max_group_size):
    res = -np.ones(\
        shape=(len(group_data), max_group_size*group_data[0].shape[1]), \
        dtype=int)
    for i in range(len(group_data)):
        group = group_data[i]
        assert(len(group) <= max_group_size)
        res[i, :group.size] = group.flatten().reshape(1, -1)
    return res

def get_time():
    return time.asctime(time.localtime(time.time()))

def get_data_by_FK(i_group_data_with_id, FK_set):
    i_group_list = []
    for i_group in i_group_data_with_id:
        if i_group[0, -1] in FK_set:
            i_group_list.append(i_group)
    return i_group_list

def get_sorted_data_by_FK(i_group_data_with_id, FK_set):
    i_group_list = get_data_by_FK(i_group_data_with_id, FK_set)
    i_data = np.concatenate(i_group_list, axis=0)
    i_data = i_data[np.argsort(i_data[:, -1], axis=0)]
    i_group_data = get_group_data(i_data, -1)
    return i_group_data

def get_domain(col, domain_dict):
    domain_dict = {i: domain_dict[col[i]] for i in range(len(col))}
    dom = domain.Domain(domain_dict, list(range(len(domain_dict))))
    return dom

def dict_add(d1, d2):
    res_dict = {}
    for key, value in d1.items():
        if type(value) == dict:
            res_dict[key] = dict_add(value, d2[key])
        else:
            res_dict[key] = value + d2[key]
    return res_dict

def dict_divide(d1, val):
    res_dict = {}
    for key, value in d1.items():
        if type(value) == dict:
            if type(val) is dict:
                res_dict[key] = dict_divide(value, val[key])
            else:
                res_dict[key] = dict_divide(value, val)
        else:
            if type(val) is dict:
                # print(key, value, val[key])
                # print(value / val[key])
                res_dict[key] = value / val[key]
            else:
                res_dict[key] = value / val
    return res_dict

def dict_mean(d):
    import copy
    res_dict = copy.deepcopy(d)
    for key, value in res_dict.items():
        if type(value) == dict:
            res_dict[key] = dict_mean(value)
        else:
            assert(type(value) == list)
            res_dict[key] = np.mean(np.array(value), axis=0)
    return res_dict

def down_sample(group_data, max_group_size, drop=False):
    print('downsampling...')
    res_group_data = []
    for group in group_data:
        group = group.copy()
        if len(group) > max_group_size:
            if drop:
                continue
            np.random.shuffle(group)
            group = group[:max_group_size]
        res_group_data.append(group)

    res_data = np.concatenate(res_group_data, axis=0)
    # res_data = res_data[:, 1:-1]

    res_group_data = np.array(res_group_data, dtype=object)
    total = sum([len(group) for group in group_data])
    print('downsample ratio {:.4f}'.format(len(res_data)/total))

    return res_data, res_group_data

def get_local_domain(domain):
    attr_list = domain.attr_list
    dom_dict = {i: domain.dict[attr_list[i]] for i in range(len(domain))}
    return Domain(dom_dict, list(range(len(domain))))

def get_local_marginal(marginal, attr_list, full_attr_list):
    assert(set(marginal) <= set(attr_list))
    assert(set(attr_list) <= set(full_attr_list))
    std_marginal = tuple(sorted(marginal))
    assert(std_marginal == tuple(marginal))
    local_marginal = [i for i in range(len(attr_list)) if attr_list[i] in marginal]
    local_marginal = tuple(local_marginal)
    return local_marginal

def get_external_marginal(local_marginal, attr_list):
    assert(len(local_marginal) <= len(attr_list))
    marginal = [attr_list[local_marginal[i]] for i in range(len(local_marginal))]
    marginal = tuple(marginal)
    return marginal

def collect_error(path, label='error:'):
    with open(path) as in_file:
        res_list = []
        for line in in_file:
            idx = line.find(label)
            if idx != -1:
                res_list.append(float(line[idx+len(label):].strip()))
        return res_list
    
def contains_list(main_list: list, sub_list: list) -> bool:
    main_list_counter = {}
    sub_list_counter = {}

    for item in main_list:
        main_list_counter[item] = main_list_counter.get(item, 0) + 1

    for item in sub_list:
        sub_list_counter[item] = sub_list_counter.get(item, 0) + 1

    for key, count in sub_list_counter.items():
        if key not in main_list_counter or main_list_counter[key] < count:
            return False

    return True

def get_cliques(attr_list: list, domain: Domain, size: float) -> list[list]:
    if len(attr_list) == 0:
        return []
    
    attr = attr_list[0]
    new_attr_list = attr_list[1:]
    new_size = size/domain.dict[attr]['size']
    clique_list1 = get_cliques(new_attr_list, domain, new_size)
    clique_list2 = get_cliques(new_attr_list, domain, size)

    res_clique_list = []
    for clique in clique_list1:
        clique.append(attr)
        res_clique_list.append(clique)
    res_clique_list.extend(clique_list2)

    return res_clique_list

def in_clique(marginal, clique_list: list[set]):
    marginal = set(marginal)
    for clique in clique_list:
        if marginal <= clique:
            return True
    return False

def get_candidiate_list(attr_list: list, domain: Domain, size: float, length=None) -> list[tuple]:
    res_list = []
    for i in range(1, length+1):
        for marginal in itertools.combinations(attr_list, i):
            if domain.project(marginal).size() < size:
                res_list.append(marginal)
    return res_list

def get_candidate_list_of_cliques(clique_list: list[list], domain: Domain, size: float, length=None) -> list[tuple]:
    result_list = []
    for clique in clique_list:
        result_list.extend(get_candidiate_list(clique, domain, size, length))
    result_list = list(set(result_list))
    return result_list

def get_maximal_cliques(attr_list: list, domain: Domain, size: float, check=None) -> set[tuple]:
    if size < 1:
        return set()
    if len(attr_list) == 0:
        return {(),}
    
    attr = attr_list[0]
    new_attr_list = attr_list[1:]
    new_size = size/domain.dict[attr]['size']
    clique_set1 = get_maximal_cliques(new_attr_list, domain, size, check=check)
    clique_set2 = get_maximal_cliques(new_attr_list, domain, new_size, check=check)

    if check is None:
        for clique in clique_set2:
            if clique in clique_set1:
                clique_set1.remove(clique)
            clique = tuple(sorted(list(clique) + [attr,]))
            clique_set1.add(clique)
    else:
        for clique in clique_set2:
            new_clique = tuple(sorted(list(clique) + [attr,]))
            if check(new_clique):
                if clique in clique_set1:
                    clique_set1.remove(clique)
                clique_set1.add(new_clique)

    return clique_set1

def get_CFS_score(target: int, conditions: list[int], edge_scores_dict: list[list]) -> float:
    conditions = sorted(set(conditions))
    if target in conditions:
        conditions.remove(target)

    numerator = 0
    for attr in conditions:
        numerator += edge_scores_dict[tuple(sorted([attr, target]))]
    numerator = max(numerator, 0.0)
    # print('numerator:', numerator)

    denominator = len(conditions)
    for attr1, attr2 in itertools.combinations(conditions, 2):
            # print(attr1, attr2)
            denominator += edge_scores_dict[(attr1, attr2)]
    denominator *= 2
    denominator = max(denominator, 1.0)
    # print('denominator:', denominator)
    
    return numerator / math.sqrt(denominator)

def expand_to_tuple_pair(group_data):
    output_data = []
    for group in group_data:
        if group.shape[0] <= 1:
            continue
        for i, j in itertools.combinations(range(group.shape[0]), 2):
            # print(group.shape)
            # print(group[i].shape)
            # print(group[j].shape)
            row = np.concatenate([group[i], group[j]], axis=0)
            row = row.reshape((1, -1))
            output_data.append(row)
    output_data = np.concatenate(output_data, axis=0)
    return output_data

def expand_to_unordered_tuple_pair(group_data):
    output_data = expand_to_tuple_pair(group_data)
    assert(output_data.shape[1] % 2 == 0)
    width = output_data.shape[1] // 2

    # print(output_data.shape, width)
    output_data1 = np.concatenate([output_data[:, width:], output_data[:, :width]], axis=1)

    output_data = np.concatenate([output_data, output_data1], axis=0)

    return output_data

def get_tuple_pair_domain(i_domain):
    pair_domain = i_domain.copy()
    var = len(i_domain)
    assert(i_domain.attr_list == list(range(var)))
    for i in i_domain.attr_list:
        pair_domain.add_variable(var, i_domain.dict[i]['size'], i_domain.dict[i])
        var += 1
    return pair_domain
    
def Naive_Bayes_one_dim_prob(hist, existing_data):
    assert(len(hist.shape) == 2)
    marginal = np.sum(hist, axis=1)
    prob = marginal.copy()
    for t in existing_data:
        prob *= hist[t] / marginal
        # sample t_i based on t_{i-1}, \cdots, t_1
        # p(t_i \mid t_{i-1},\cdots,t_1) \approx p(t_i) p(t_{i-1} \mid t_i) p(t_{i-2} \mid t_i) \cdots p(t_1 \mid t_i)
        # p(t_{i-1} \mid t_i) = p(t_{i-1}, t_i) / p(t_i) = hist[t_{i-1}] / marginal

    return prob / np.sum(prob)

def print_one_dim_group(group_data):
    for group in group_data:
        assert(group.shape[1] == 1)
        print(','.join([str(item) for item in group.flatten()]))

# get the counts of possible combinations of groups
def get_one_dim_group_cnt(group_data):
    # Conver groups to count dicts
    group_count_data = [Counter(group.flatten()) for group in group_data]

    # for cnt in group_count_data:
    #     print(cnt.items())
    #     print(tuple(sorted(cnt.items())))

    # Count the occurrences of dicts
    # count_values = Counter([frozenset(cnt.items()) for cnt in group_count_data])
    count_values = Counter([tuple(sorted(cnt.items())) for cnt in group_count_data])

    count_items = list(count_values.items())
    count_items.sort(key=lambda x: x[1], reverse=True)
    # print('count_items:')
    # for val, cnt in count_items:
    #     print(val, cnt)
    # print('')

    return count_values

def group_cnt_tvd(group_data1, group_data2):
    group_cnt1 = get_one_dim_group_cnt(group_data1)
    group_cnt2 = get_one_dim_group_cnt(group_data2)
    # print(group_cnt1)
    # print(group_cnt2)
    length1 = len(group_data1)
    length2 = len(group_data2)
    diff = 0
    for val in group_cnt1:
        cnt1 = group_cnt1[val]/length1
        cnt2 = group_cnt2[val]/length2
        if cnt1 > cnt2:
            diff += cnt1 - cnt2
    return diff

def get_junction_tree(graph) -> nx.Graph:
    assert(nx.is_chordal(graph))
    maximal_cliques = [tuple(sorted(clique)) for clique in nx.find_cliques(graph)]
    clique_graph = nx.Graph()
    clique_graph.add_nodes_from(maximal_cliques)
    for clique1, clique2 in itertools.combinations(maximal_cliques, 2):
        clique_graph.add_edge(clique1, clique2, weight=-len(set(clique1) & set(clique2)))
    junction_tree = nx.minimum_spanning_tree(clique_graph)
    return junction_tree

def get_graph_MI_sum(graph, data, domain, MI_map={}, entropy_map={}):
    MI = 0
    for attr1, attr2 in graph.edges:
        MI += get_mutual_info(MI_map, entropy_map, (attr1, attr2), data, domain)
    return MI

def get_junction_tree_entropy(junction_tree, data, domain, entropy_map={}):
    start_clique = list(junction_tree.nodes)[0]
    model_entropy = 0
    entropy = get_entropy(entropy_map, start_clique, data, domain)
    model_entropy += entropy
    for start, clique in nx.dfs_edges(junction_tree, source=start_clique):
        entropy = get_entropy(entropy_map, clique, data, domain)
        model_entropy += entropy
        separator = set(start) & set(clique)
        if len(separator) != 0:
            entropy = get_entropy(entropy_map, separator, data, domain)
            model_entropy -= entropy

    return model_entropy

def get_theoretic_loss(domain: Domain, marginal_list: list[tuple], noise: float):
    theoretic_loss = 0
    for marginal in marginal_list:
        theoretic_loss += domain.project(marginal).size() * noise ** 2
    return theoretic_loss

def get_Gaussian_noise(sensitivity, budget, query_num=1):
    noise = (sensitivity ** 2 * query_num / budget) ** 0.5
    return noise

def get_1norm(data, domain, marginal, marginal_hist):
    data_hist = get_marginal(data, domain, marginal)
    return np.sum(np.abs(data_hist - marginal_hist))

# flattened_domain and per_domain are almost the same. flattened_domain is at most max_group_size
# while per_domain is at most tuple_num
def get_flattened_domain(h_domain: Domain, i_domain: Domain, tuple_num):
    per_domain = h_domain.copy()
    var = len(per_domain)
    for t in range(tuple_num):
        print(f'individual: {t}, attr:', end='')
        for a in i_domain.attr_list:
            # add 1 to size as 0 is for invalid values (size < max_group_size)
            per_domain.add_variable(var, i_domain.dict[a]['size'], i_domain.dict[a])
            print(var, end=' ')
            var += 1
        print('')
    # print('h_domain:', h_domain)
    # print('per_domain:', per_domain)
    return per_domain

# For each group, get its all permuations
# return (group_num, permutation_num, h_attr_num + i_attr_num * tuple_num).
# As we normalize the count of each group permutation to 1
# if all groups have the same size, directly querying on flattened permutations has
# no bias even taking h_data - i_data into considerations.
def get_group_permutations(h_data, group_data, tuple_num, padding_value=0):

    assert(tuple_num <=3 ) # leads to exponential complexity
    assert(len(h_data) == len(group_data))


    total_permutations = sum(
        len(list(itertools.permutations(range(group_data[i].shape[0]), min(group_data[i].shape[0], tuple_num))))
        for i in range(len(h_data))
    )

    # Allocate memory for permutations and weights
    permutation_dim = h_data.shape[1] + group_data[0].shape[1] * tuple_num  # Adjust dimensions accordingly
    permutations = np.zeros((total_permutations, permutation_dim), dtype=np.int32)
    weights = np.zeros(total_permutations, dtype=np.float32)

    current_index = 0

    for i in tqdm(range(len(h_data)), desc="Generating permutations", unit="groups"):
        group = group_data[i]
        temp_tuple_num = min(group.shape[0], tuple_num)
        combs = np.array(list(itertools.permutations(range(group.shape[0]), temp_tuple_num)))
        per_num = len(combs)
        tuple_combs = group[combs].reshape(per_num, -1)
        h_per = np.tile(h_data[i], (per_num, 1))

        # Pad the first tuple if group_size < tuple_num
        ind1_tuple = tuple_combs[:, :group.shape[1]]
        if tuple_num > temp_tuple_num:  # Only pad if needed
            ind1_tuple = np.tile(ind1_tuple, (1, tuple_num - temp_tuple_num))
            tuple_combs = np.concatenate([tuple_combs, ind1_tuple], axis=1)
        
        tuple_combs_ind1 = np.concatenate([h_per, tuple_combs], axis=1)

        # Fill the pre-allocated permutations array
        end_index = current_index + per_num
        permutations[current_index:end_index] = tuple_combs_ind1
        weights[current_index:end_index] = 1 / per_num
        current_index = end_index  # Update the index

    assert current_index == total_permutations

    return permutations, weights

def get_unordered_marginal(marginal, domain, permutations, weights):
    return get_marginal(permutations, domain, marginal, weights)

# get flatten marginal from permutation marginal without breaking individual order
def permutation_to_flatten_marginal(marginal, h_attr_num, i_attr_num, possible_i: list):
    assert(sorted(marginal) == list(marginal))

    # get marginal of each individual
    i_to_attr = {i: [] for i in range(int((max(marginal) - h_attr_num)/i_attr_num)+1)}
    # print(i_to_attr)
    h_attrs = []
    for attr in marginal:
        if attr < h_attr_num:
            h_attrs.append(attr)
        else:
            i = int((attr - h_attr_num)/i_attr_num)
            i_to_attr[i].append((attr - h_attr_num)%i_attr_num)

    i_attr_comb_list = i_to_attr.values()
    i_attr_comb_list = [item for item in i_attr_comb_list if len(item) > 0]

    assert(len(i_attr_comb_list) <= len(possible_i))

    f_marginal_list = []
    for i_list in itertools.combinations(sorted(possible_i), len(i_attr_comb_list)):
        f_marginal = h_attrs.copy()
        for i in range(len(i_attr_comb_list)):
            i_marginal = [attr+h_attr_num+i_attr_num*i_list[i] for attr in i_attr_comb_list[i]]
            f_marginal.extend(i_marginal)
        f_marginal_list.append(tuple(f_marginal))

    return f_marginal_list

# get moved axes version of a per_marginal, 
# also do moving axes when the fact is given
def permutation_to_all_permutation_marginal(h_attr_num, i_attr_num, marginal, f_domain: Domain, fact: Optional[Factor] = None):
    if not fact is None:
        assert(tuple(fact.domain.attr_list) == marginal)
    # get marginal of each individual
    i_to_attr = {i: [] for i in range(int((max(marginal) - h_attr_num)/i_attr_num)+1)}
    i_to_original_attr = {i: [] for i in range(int((max(marginal) - h_attr_num)/i_attr_num)+1)}
    
    h_attrs = []
    for attr in marginal:
        if attr < h_attr_num:
            h_attrs.append(attr)
        else:
            i = int((attr - h_attr_num)/i_attr_num)
            i_to_attr[i].append((attr - h_attr_num)%i_attr_num)
            i_to_original_attr[i].append(attr)

    i_attr_comb_list = i_to_attr.values()
    i_attr_comb_list = [item for item in i_attr_comb_list if len(item) > 0]

    i_num = len(i_attr_comb_list)

    # get individual marginal permuations
    i_attr_comb_per_list = list(itertools.permutations(i_attr_comb_list, i_num))
    i_per_list = list(itertools.permutations(range(i_num), i_num))
    i_per_list = [list(item) for item in i_per_list]

    i_orginal_attr_comb_list = i_to_original_attr.values()
    i_orginal_attr_comb_list = [item for item in i_orginal_attr_comb_list if len(item) > 0]
    i_orginal_attr_comb_list = np.array(i_orginal_attr_comb_list, dtype=object)
    # print('i_orginal_attr_comb_list:')
    # print(i_orginal_attr_comb_list)
    # print('i_per_list:')
    # print(i_per_list)

    res_dict = {}
    for per_idx in range(len(i_attr_comb_per_list)):

        i_attr_comb_per = i_attr_comb_per_list[per_idx]

        # print('i_attr_comb_per:', i_attr_comb_per)
        temp_marginal = h_attrs.copy()
        for i in range(len(i_attr_comb_per)):
            i_marginal = [attr+h_attr_num+i_attr_num*i for attr in i_attr_comb_per[i]]
            # print('i_marginal:', i_marginal)
            temp_marginal.extend(i_marginal)

        temp_marginal = tuple(temp_marginal)
        # print('temp_marginal:', temp_marginal)

        if fact is None:
            res_dict[temp_marginal] = None
        else:
            i_per = i_per_list[per_idx]
            temp_new_axes = list(i_orginal_attr_comb_list[i_per])
            # print(temp_new_axes)
            temp_new_axes = [list(item) for item in temp_new_axes]
            # print(temp_new_axes)
            temp_new_axes = list(itertools.chain.from_iterable(temp_new_axes))
            # print(temp_new_axes)
            new_axes = h_attrs.copy()
            new_axes.extend(temp_new_axes)
            # print('new_axes')
            # print(f'{marginal} {temp_marginal} new_axes: {new_axes}')
            res_dict[temp_marginal] = Factor(f_domain.project(temp_marginal), fact.moveaxis(new_axes).values, np)
            # print(np.where(res_dict[temp_marginal].values>0))

    if fact is None:
        return list(res_dict.keys())
    else:
        # marginal itself should not be the moveaxis version
        # For example (ind1.A, ind2.A) can be (ind2.A, ind1.A) by moveaxis
        # But this will lose some information, ind2.A is empty for size1 groups.
        # So when you project (ind2.A, ind1.A) to (ind2.A,), size1 groups info will be lost.
        res_dict[marginal] = fact
        return res_dict
    

def flatten_to_permutation_marginal(marginal, h_attr_num, i_attr_num):

    assert(list(marginal) == sorted(marginal))

    # get marginal of each individual
    i_to_attr = {i: [] for i in range(int((max(marginal) - h_attr_num)/i_attr_num)+1)}
    # print(i_to_attr)
    h_attrs = []
    for attr in marginal:
        if attr < h_attr_num:
            h_attrs.append(attr)
        else:
            i = int((attr - h_attr_num)/i_attr_num)
            i_to_attr[i].append((attr - h_attr_num)%i_attr_num)

    i_attr_comb_list = i_to_attr.values()
    i_attr_comb_list = [item for item in i_attr_comb_list if len(item) > 0]

    # rearrange marginal of each individual to get permutation marginal
    per_marginal = h_attrs.copy()
    for i in range(len(i_attr_comb_list)):
        i_marginal = [attr+h_attr_num+i_attr_num*i for attr in i_attr_comb_list[i]]
        per_marginal.extend(i_marginal)

    return tuple(per_marginal)

def get_h_size_data(h_data, i_group_data, max_size):
    assert(h_data.shape[0] == len(i_group_data))
    size = np.array([len(group) for group in i_group_data]).reshape((-1, 1))
    size[size>max_size] = max_size
    return np.concatenate([h_data, size], axis=1)

def get_CFS_attrs(edge_scores: list[list], target_attr: int, attr_set: set, domain: Domain, size: float) -> list:
    print(f'possible_clqiue_attrs: {sorted(attr_set)}')
    if target_attr in attr_set:
        attr_set.remove(target_attr)

    edge_score_dict = {item[0]: item[1] for item in edge_scores}
    # print(edge_score_dict)
    
    candidate_cliques = get_maximal_cliques(list(attr_set), domain, size)
    max_score = -1e6
    max_clique = None
    for clique in candidate_cliques:
        score = get_CFS_score(target_attr, clique, edge_score_dict)
        if score > max_score:
            max_score = score
            max_clique = clique

    print('max_clique: {}, max_score: {:.4f}'.format(max_clique, max_score))

    if max_clique is None:
        assert(0)
    
    return max_clique

def check_consistency(fact1: Factor, fact2: Factor, tol=0.01) -> float:
    inter = sorted(set(fact1.domain.attr_list).intersection(fact2.domain.attr_list))
    if len(inter) == 0:
        return None
    val1 = fact1.project(inter).values
    val2 = fact2.project(inter).values
    tvd = float(get_TVD(val1, val2, normalize=True))
    if tvd > tol:
        print('WARNING: {}, {} -> {}, consistency tvd: {:.4f}'.format(
            fact1.domain.attr_list, fact2.domain.attr_list, inter, tvd
        ))
    return tvd

# return the log prob of x_idx in potential.
# At most one attr of x_idx can be None, others should be given.
def get_log_prob(potential, x_idx, partition_func):
    prob = 0
    maximal_cliques = list(potential.keys())
    for clique in maximal_cliques:
        # print(x_idx, clique)
        idx = tuple([x_idx[attr] for attr in clique])
        prob += potential[clique].values[idx]
        # print(potential[tuple(clique)].values)
        # print(clique, potential[tuple(clique)].values[idx])
        # print('')
    prob -= partition_func

    return prob

def print_memory_summary():
    import pympler

    all_objects = pympler.muppy.get_objects()
    mem_summary = pympler.summary.summarize(all_objects)
    pympler.summary.print_(mem_summary)

def get_select_marginal_schedule(marginal_num, iter_num):
    if marginal_num < iter_num:
        return [1] * marginal_num
    marginal_num_list = [int(marginal_num/iter_num)] * iter_num
    for i in range(marginal_num-sum(marginal_num_list)):
        marginal_num_list[i] += 1
    return marginal_num_list

def read_result(path, keyword='error:'):
    with open(path, 'r') as input_file:
        result_list = []
        for line in input_file:
            index = line.find(keyword)
            if index != -1:
                index += len(keyword)
                result = float(line[index:].strip())
                result_list.append(result)
        return result_list

def get_file_paths(directory):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    return file_paths

# sort 2d numpy array by the 1st column, the 2nd column, ..., the last column.
def sort_data_by_cols(data):
    assert(data.ndim == 2)
    return np.lexsort(data.T[::-1])

def get_group_size(data, group_id_attrs, max_size=None):
    if len(group_id_attrs) > 1:
        data = data[np.lexsort(data[:, group_id_attrs].T)]
        group_data = get_group_data(data, group_id_attrs)
    else:
        group_data = downsample_group(data, group_id_attrs[0], max_size)

    length = np.array([len(group) for group in group_data], dtype=int)
    if max_size is None:
        max_size = max(length)
    else:
        idx = length>max_size
        print('size < max_size proportion: {:.4f}'.format(1-np.sum(idx)/len(length)))
        length[idx] = max_size
    hist, _ = np.histogram(length, bins=list(range(max_size+2)))
    return hist

def shuffle_dataframe(df: pd.DataFrame):
    idx = np.random.permutation(len(df))
    shuffled_df = df.iloc[idx].reset_index(drop=True)
    return shuffled_df


def move_col(data, col, new_col):
    df = pd.DataFrame(data, columns=col)
    df = df[new_col]
    return df.to_numpy()

def generate_synthetic_column_data(df, factor, cond, target):
    print(f'    sampling attr {target} conditioned on {cond}')
    assert(target not in cond)
    assert(tuple(cond) == tuple(sorted(cond)))
    assert(factor.domain.attr_list == sorted(factor.domain.attr_list))

    if len(cond) == 0:
        prob = factor.project(target).values
        df.loc[:, target] = generate_column_data(prob, len(df))
    else:
        marginal_value = factor.project(cond + [target])
        marginal_value = marginal_value.moveaxis(cond + [target]).values

        def foo(group):
            idx = group.name
            vals = generate_column_data(marginal_value[idx], group.shape[0])
            group[target] = vals
            return group

        df.update(df.groupby(list(cond)).apply(foo))

def downsample_group(data: np.array, col: int, max_group_size: int, random_state=None, return_group_data=True) -> np.array:
    """
    Downsample groups in a NumPy array without using groupby.

    Parameters
    ----------
    data : np.array
        Input 2D array. Each row is a record; columns are features.
    col : int
        The index of the column by which to group (e.g., a FK, user_id).
    max_group_size : int
        The maximum number of rows to sample for each group.
    return_group_data : bool, optional
        If True, return a np array, each item is a downsampled group.
        If False, return the concatenated groups
    Returns
    -------
    np.array
        A 2D NumPy array containing the downsampled rows.
    """
    # Sort the array by the grouping column
    if isinstance(col, list):
        if len(col) == 1:
            col = col[0]
    sort_indices = np.argsort(data[:, col])
    data_sorted = data[sort_indices]
    group_vals = data_sorted[:, col]

    # Identify consecutive blocks of rows belonging to each unique value in 'col'
    block_indices = []
    block_start = 0
    n = len(group_vals)

    for i in range(1, n):
        if group_vals[i] != group_vals[i - 1]:
            # We have reached the end of a block
            block_indices.append((block_start, i))
            block_start = i
    # Handle the final block
    block_indices.append((block_start, n))

    # Randomly sample rows from each block
    rng = np.random.default_rng(seed=random_state)
    sampled_indices = []
    for start, end in block_indices:
        block_size = end - start
        sample_size = min(block_size, max_group_size)
        chosen = rng.choice(
            np.arange(start, end),
            size=sample_size,
            replace=False
        )
        sampled_indices.append(chosen)


    if return_group_data:
        res_data = []
        # print(sampled_indices)
        # print(data_sorted)
        for indices in sampled_indices:
            res_data.append(data_sorted[indices])
        return np.array(res_data, dtype=object)
    else:
        sampled_indices = np.concatenate(sampled_indices)
        return data_sorted[sampled_indices]

