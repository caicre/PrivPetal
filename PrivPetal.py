import pandas as pd
import numpy as np
import MRF
import sys
import utils
import multiprocessing
import json
import cupy as cp
import itertools
import os
import math
import pickle
import networkx as nx
import copy
from tqdm import tqdm
import time
import tracemalloc
import ctypes

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def get_clique_by_size(attr_list, domain, size):
    res_list = []
    cur_size = 1
    for attr in attr_list:
        cur_size *= domain.dict[attr]['size']
        if cur_size <= size:
            res_list.append(attr)
        else:
            break
    return res_list

def sort_attr_by_score(edge_scores, target_attr, attr_list, num=15):
    inter_edges = [item[0] for item in edge_scores if target_attr in item[0]]
    inter_attrs = []
    for edge in inter_edges:
        inter_attrs.extend(edge)
    inter_attrs = [item for item in inter_attrs if item != target_attr]
    inter_attrs = [item for item in inter_attrs if item in attr_list]
    inter_attrs = inter_attrs[:num]

    return inter_attrs

def recover_FK_data(f_data, i_domain, h_domain, h_key_array=None):
    i_attr_num = len(i_domain)
    h_size = len(h_domain)
    h_data = []
    i_data = []

    if h_key_array is None:
        h_key_array = np.arange(len(f_data))

    h_idx = 0
    for row in f_data:
        h_row = -np.ones(h_size+1, dtype=int)
        h_row[1:] = row[:h_size]
        h_row[0] = h_key_array[h_idx]
        h_data.append(h_row.reshape((1, -1)))

        group_size = row[h_size]

        i_group = []
        idx = h_size + 1
        for i in range(group_size):
            i_group.append(row[idx: idx+i_attr_num].reshape((1, -1)))
            idx += i_attr_num

        if len(i_group) > 0:
            i_group = np.concatenate(i_group, axis=0)
            i_group = np.concatenate([i_group, np.full(shape=(len(i_group), 1), fill_value=h_key_array[h_idx], dtype=int)], axis=1)
            i_data.append(i_group)

        h_idx += 1

    h_data = np.concatenate(h_data, axis=0)
    i_data = np.concatenate(i_data, axis=0)
    i_data = np.concatenate([np.arange(len(i_data)).reshape((-1, 1)), i_data], axis=1)

    return h_data, i_data


class Data:
    def __init__(self, h_df, h_domain, i_df, i_domain, check_domain=True) -> None:
        self.h_df, self.h_domain = h_df.copy(), h_domain.copy()
        self.i_df, self.i_domain = i_df.copy(), i_domain.copy()

        self.h_df = self.h_df.sort_values(by=self.h_df.columns[0])
        self.i_df = self.i_df.sort_values(by=self.i_df.columns[-1])

        self.h_data = self.h_df.to_numpy()
        self.i_data = self.i_df.to_numpy()

        if check_domain:
            MRF.tools.assert_data_domain(self.h_data[:, 1:], self.h_domain)
            MRF.tools.assert_data_domain(self.i_data[:, 1:-1], self.i_domain)

        # assert FK is valid
        assert np.all(np.isin(self.i_data[:, -1], self.h_data[:, 0]))

    @classmethod
    def load_data(cls, path):
        h_df = pd.read_csv(os.path.join(path, 'household.csv'))
        i_df = pd.read_csv(os.path.join(path, 'individual.csv'))

        h_domain = json.load(open(os.path.join(path, 'household_domain.json')))
        i_domain = json.load(open(os.path.join(path, 'individual_domain.json')))
        h_domain = MRF.tools.get_domain_by_attrs(h_domain, h_df.columns[1:])
        i_domain = MRF.tools.get_domain_by_attrs(i_domain, i_df.columns[1:-1])

        return cls(h_df, h_domain, i_df, i_domain)

    def factorize(self, h_attr_list, h_fact_size, i_attr_list, i_fact_size):
        h_data, h_domain = self.h_data.copy(), self.h_domain.copy()
        i_data, i_domain = self.i_data.copy(), self.i_domain.copy()

        if not h_attr_list is None:
            h_data, h_domain, h_attr_to_new_attr = utils.tools.factorize(
                h_attr_list, h_fact_size, h_data[:, 1:], h_domain)
            h_data = np.concatenate([self.h_data[:, [0,]], h_data], axis=1)

        if not i_attr_list is None:
            i_data, i_domain, i_attr_to_new_attr = utils.tools.factorize(
                i_attr_list, i_fact_size, i_data[:, 1:-1], i_domain)
            i_data = np.concatenate([self.i_data[:, [0,]], i_data, self.i_data[:, [-1,]]], axis=1)

        h_df = pd.DataFrame(h_data, columns=list(range(h_data.shape[1])))
        i_df = pd.DataFrame(i_data, columns=list(range(i_data.shape[1])))
        return Data(h_df, h_domain, i_df, i_domain), h_attr_to_new_attr, i_attr_to_new_attr

    def get_group_data(self, col, sample_size) -> None:
        
        if isinstance(col, list) and len(col) > 1:
            self.i_group_data = MRF.tools.get_group_data(self.i_data, col)
        else:
            self.i_group_data = MRF.tools.downsample_group(self.i_data, col, float('inf'))

        assert len(self.i_group_data) <= len(self.h_data)
        assert np.all(np.isin(self.i_data[:, -1], self.h_data[:, 0]))

        # group_size_hist
        size_array = np.array([len(group) for group in self.i_group_data])
        print('proportion size < sample_size: {:.4f}'.format(np.sum(size_array<=sample_size)/len(size_array)))
        group_size_hist, _ = np.histogram(size_array, bins=list(range(max(size_array)+2)))
        print('original group_size_hist:')
        print(group_size_hist)
        print('quantiles:')
        num_list = [0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
        quantiles = np.quantile(size_array, num_list)
        for i in range(len(num_list)):
            print('    {}: {}'.format(num_list[i], quantiles[i]))

        size_array[size_array>sample_size] = sample_size
        self.group_size_hist, _ = np.histogram(size_array, bins=list(range(max(size_array)+2)))

        FK_list = np.array([group[0, -1] for group in self.i_group_data])
        same_count = 0
        for i in range(min(len(FK_list), len(self.h_data))):
            if FK_list[i] == self.h_data[i, 0]:
                same_count += 1
        print('ratio of same FK: {:.4f}'.format(same_count/len(self.h_data)))


    # downsample, add size to h_data as a col, get permutations
    def prepare_data(self, sample_size, max_group_size, tuple_num) -> None:
        self.sample_size = sample_size
        self.max_group_size = max_group_size
        self.tuple_num = tuple_num

        # h_size_data
        self.h_size_data = MRF.tools.get_h_size_data(self.h_data, self.i_group_data, self.sample_size)
        self.h_size_domain = self.h_domain.copy()
        self.h_size_domain.add_variable(len(self.h_domain), size=self.sample_size+1)
        self.size_attr = len(self.h_domain)

        # h attr number including group size
        self.h_attr_num = len(self.h_size_domain)
        self.i_attr_num = len(self.i_domain)

        self.total = np.sum(self.group_size_hist)
        self.noisy_total = None

        # downsample
        # max_group_size does not influence the sensitivities of current FK, but may influence downstream FKs.
        self.i_data, self.i_group_data = MRF.tools.down_sample(
            self.i_group_data, self.max_group_size, drop=False)

        # permutation and flatten domain
        self.f_domain = MRF.tools.get_flattened_domain(self.h_size_domain, self.i_domain, self.sample_size)
        strip_i_group_data = np.array([group[:, 1:-1] for group in self.i_group_data], dtype=object)

        self.permutations, self.weights = MRF.tools.get_group_permutations(
            self.h_size_data[:, 1:], strip_i_group_data, self.tuple_num, padding_value=-1)



    def get_individual(self, attr):
        if attr is None: # None attr also means sampling h_attr
            return -1
        if attr >= self.h_attr_num:
            return int((attr - self.h_attr_num)/self.i_attr_num)
        return -1

    # should only consider marginals involving tuples smaller than tuple_num
    def get_individual_num(self, marginal):
        i_set = [self.get_individual(attr) for attr in marginal]
        i_set = set([item for item in i_set if item >= 0])
        return len(i_set)

def get_size_TVD(fact1, fact2, data):
    marginal = tuple(fact1.domain.attr_list)
    size_axis = marginal.index(data.size_attr)
    size_domain = data.f_domain.dict[data.size_attr]['size']

    res_tvd = []
    for s in range(1, size_domain):

        slices = [slice(None)] * fact1.values.ndim
        slices[size_axis] = s
        slices = tuple(slices)

        total1 = fact1.values[slices].sum()
        total2 = fact2.values[slices].sum()

        tvd = MRF.tools.get_TVD_count(fact1.values[slices], fact2.values[slices]) / total1

        res_tvd.append(tvd)
        print('marginal: {}, size: {} tvd: {:.5f}'.format(marginal, s, tvd))
    print('')     
    return res_tvd

def normalize_by_size(fact: MRF.Factor, data: Data):
    size_axis = fact.domain.attr_list.index(data.size_attr)
    size_domain = data.f_domain.dict[data.size_attr]['size']
    for s in range(1, size_domain):
        slices = [slice(None)] * fact.values.ndim
        slices[size_axis] = s
        slices = tuple(slices)

        fact_size_sum = np.sum(fact.values[slices])
        if fact_size_sum > 0:
            fact.values[slices] = data.group_size_hist[s] / fact_size_sum  * fact.values[slices]
        else:
            # when fact is too noisy
            fact.values[slices] = 0

    slices = [slice(None)] * fact.values.ndim
    slices[size_axis] = 0
    slices = tuple(slices)
    fact.values[slices] = 0


# Note: cells of low ind of high ind marginals are invalid
# For example, (size, ind1.A, ind2.A, ind3.A)'s cells of size=1 are meaningless
# since size=1 group can not have ind2 and ind3. 
# However, these cells have padding values because we pad 0 in the permutation data
# This is useful when projecting marginals. That is, we can project
# (size, ind1.A, ind2.A, ind3.A) -> (size, ind1.A, ind2.A) by summation
# and the result is stil correct.

# ShareMarginal can be written/read by multiple processes simultaneously
# They may overwrite each other's marginal
class SharedMarginal:
    def __init__(self) -> None:
        pass

    def init(self, manager, size_bins=[]) -> None:
        self.per_marginal_dict            = manager.dict()
        self.flatten_marginal_dict        = manager.dict()
        self.noisy_per_marginal_dict      = manager.dict()
        self.noisy_flatten_marginal_dict  = manager.dict()
        self.noisy_projectable_set = set()
        self.projectable_set = set()
        self.size_bins = size_bins.copy()

    def get_noisy_flatten_marginal(self, data: Data, f_marginal, noise, always_merge=False) -> MRF.Factor:
        if f_marginal not in self.noisy_flatten_marginal_dict:
            per_marginal = MRF.tools.flatten_to_permutation_marginal(
                f_marginal, data.h_attr_num, data.i_attr_num)
            self.get_noisy_per_marginal(data, per_marginal, noise, always_merge=always_merge)

            f_fact = self.get_noisy_flatten_marginal(data, f_marginal, noise)
            self.project_marginals(data, f_fact, noisy=True)

        return self.noisy_flatten_marginal_dict[f_marginal]
    
    def marginal_merge_to(self, marginal, array: np.ndarray, data: Data, size_bins) -> np.ndarray:
        size_axis = marginal.index(data.size_attr)
        shape = list(array.shape)
        shape[size_axis] = array.shape[size_axis] - len(size_bins) + 1
        shape = tuple(shape)
        res_array = np.zeros(shape=shape, dtype=float)

        # Get size map from original marginal to the merged marginal
        # So we can get the merged marginal values by summation
        size_domain = data.f_domain.dict[data.size_attr]['size']
        size_map = []
        ind1 = 0
        for ind2 in range(size_domain):
            if ind2 in size_bins:
                size_map.append(ind1)
            else:
                size_map.append(ind1)
                ind1 += 1
        assert(max(size_bins)+1 == size_domain)
        assert(max(size_bins)-min(size_bins)+1 == len(size_bins))

        for s in range(size_domain):
            org_slices = [slice(None)] * res_array.ndim
            org_slices[size_axis] = s
            org_slices = tuple(org_slices)

            slices = [slice(None)] * res_array.ndim
            slices[size_axis] = size_map[s]
            slices = tuple(slices)

            if s in size_bins:
                res_array[slices] += array[org_slices]
            else:
                res_array[slices] = array[org_slices]

        return res_array
    
    
    def marginal_merge_back(self, marginal, array: np.ndarray, data: Data, size_bins) -> np.ndarray:
        size_axis = marginal.index(data.size_attr)
        shape = data.f_domain.project(marginal).shape
        res_array = np.zeros(shape=shape, dtype=float)

        # Get size map to map merged marginal to original marginal
        size_domain = data.f_domain.dict[data.size_attr]['size']
        size_map = []
        ind1 = 0
        for ind2 in range(size_domain):
            if ind2 in size_bins:
                size_map.append(ind1)
            else:
                size_map.append(ind1)
                ind1 += 1

        proportion = data.group_size_hist.copy()
        proportion = proportion / proportion[size_bins].sum()

        for s in range(size_domain):
            org_slices = [slice(None)] * res_array.ndim
            org_slices[size_axis] = s
            org_slices = tuple(org_slices)

            slices = [slice(None)] * res_array.ndim
            slices[size_axis] = size_map[s]
            slices = tuple(slices)

            if s in size_bins:
                res_array[org_slices] = array[slices] * proportion[s]
            else:
                res_array[org_slices] = array[slices]

        return res_array

    def get_noisy_per_marginal(self, data: Data, per_marginal: tuple, noise, always_merge=False, normalize=True) -> MRF.Factor:
        if per_marginal not in self.noisy_per_marginal_dict:
            dom_size = data.f_domain.project(per_marginal).size()
            size_domain = data.f_domain.dict[data.size_attr]['size']
            fact = self.get_per_marginal(data, per_marginal)

            # size_bins should not include sizes smaller than the ind number of a marginal
            # For example, (ind1.A, ind2.A) should not include information of size 1 data
            # And the size_bins used by it should be at least [2, 3, ...]
            min_size = max([data.get_individual(attr) for attr in per_marginal])+1
            size_bins = [size for size in self.size_bins if size >= min_size]

            error1 = noise * (2/math.pi)**0.5 * dom_size
            if len(size_bins) > 0:
                error2 = noise * (2/math.pi)**0.5 * dom_size * (size_domain-len(size_bins)+1)/size_domain
                merged_val = self.marginal_merge_to(per_marginal, fact.values, data, size_bins)
                merged_back_val = self.marginal_merge_back(per_marginal, merged_val, data, size_bins)
                error2 += MRF.tools.get_TVD_count(fact.values, merged_back_val) * 2

            if len(size_bins) > 0 and (always_merge or error1 > error2):

                noisy_merged_val = merged_val + np.random.normal(scale=noise, size=merged_val.shape)
                noisy_val = self.marginal_merge_back(per_marginal, noisy_merged_val, data, size_bins)
                noisy_fact = MRF.Factor(fact.domain.copy(), noisy_val, np)

                tvd = MRF.tools.get_TVD(fact.values, noisy_fact.values)
                print('    get noisy merged per marginal: {}, dom: {}, query tvd: {:.5f}'.\
                    format(per_marginal, fact.domain.size(), tvd))
                
            else:
                noisy_val = fact.values + np.random.normal(scale=noise, size=fact.values.shape)
                noisy_fact = MRF.Factor(fact.domain.copy(), noisy_val, np)

                tvd = MRF.tools.get_TVD(fact.values, noisy_fact.values)
                print('    get noisy per marginal: {}, dom: {}, query tvd: {:.5f}'.\
                    format(per_marginal, fact.domain.size(), tvd))
            
            if normalize:
                normalize_by_size(noisy_fact, data)

            self.save_per_marginal(data, noisy_fact, noisy=True)
            self.noisy_projectable_set.add(per_marginal)

        
        return self.noisy_per_marginal_dict[per_marginal]
    

    def get_flatten_marginal(self, data: Data, f_marginal) -> MRF.Factor:
        if f_marginal not in self.flatten_marginal_dict:
            per_marginal = MRF.tools.flatten_to_permutation_marginal(
                f_marginal, data.h_attr_num, data.i_attr_num)
            self.get_per_marginal(data, per_marginal)

        return self.flatten_marginal_dict[f_marginal]

    # get per_marginal, save it, its moveaxis version and their flatten marginals
    def get_per_marginal(self, data: Data, per_marginal) -> MRF.Factor:
        if per_marginal not in self.per_marginal_dict:
            # print('    get per marginal: {}'.format(per_marginal))
            hist = MRF.tools.get_unordered_marginal(
                per_marginal, data.f_domain, data.permutations, data.weights)
            per_fact = MRF.Factor(data.f_domain.project(per_marginal), hist, np)
            self.save_per_marginal(data, per_fact, noisy=False)
            self.projectable_set.add(per_marginal)

        return self.per_marginal_dict[per_marginal]

    # save flatten marginal, save it, its moveaxis version and their flatten marginals
    def save_flatten_marginal(self, data: Data, f_fact, noisy=True):
        f_marginal = tuple(sorted(f_fact.domain.attr_list))
        per_marginal = MRF.tools.flatten_to_permutation_marginal(
            f_marginal, data.h_attr_num, data.i_attr_num)
        per_fact = MRF.Factor(data.f_domain.project(per_marginal), f_fact.values, np)
        self.save_per_marginal(data, per_fact, noisy=noisy)

    # save per_marginal and their flatten marginal
    def save_per_marginal(self, data: Data, per_fact, noisy=True):
        if noisy:
            per_marginal_dict       = self.noisy_per_marginal_dict
            flatten_marginal_dict   = self.noisy_flatten_marginal_dict
        else:
            per_marginal_dict       = self.per_marginal_dict
            flatten_marginal_dict   = self.flatten_marginal_dict

        per_marginal = tuple(sorted(per_fact.domain.attr_list))
        if per_marginal in per_marginal_dict:
            return

        res_dict = MRF.tools.permutation_to_all_permutation_marginal(
            data.h_attr_num, data.i_attr_num, per_marginal, data.f_domain, per_fact)
        for other_per_marginal in res_dict:
            if other_per_marginal in per_marginal_dict:
                continue
            # print('        save noisy: {} per to per: {} -> {}'.format(noisy, per_marginal, other_per_marginal))
            per_marginal_dict[other_per_marginal] = res_dict[other_per_marginal]

            f_marginal_list = MRF.tools.permutation_to_flatten_marginal(
                other_per_marginal, data.h_attr_num, data.i_attr_num, list(range(data.sample_size)))
            for f_marginal in f_marginal_list:
                # print('            save noisy: {} per to flatten: {} -> {}'.format(noisy, other_per_marginal, f_marginal))
                flatten_marginal_dict[f_marginal] = MRF.Factor(
                    data.f_domain.project(f_marginal),
                    res_dict[other_per_marginal].values,
                    np
                )

    # project given noisy marginal, and save their moveaxis versions
    # will only project per_marginal (in the projectable_set)
    def project_marginals(self, data: Data, fact: MRF.Factor, noisy=True):
        if noisy:
            per_marginal_dict = self.noisy_per_marginal_dict
            projectable_set = self.noisy_projectable_set
        else:
            per_marginal_dict = self.per_marginal_dict
            projectable_set = self.projectable_set

        f_marginal = tuple(fact.domain.attr_list)
        per_marginal = MRF.tools.flatten_to_permutation_marginal(
            f_marginal, data.h_attr_num, data.i_attr_num)
        if not per_marginal in projectable_set:
            return
        per_fact = MRF.Factor(data.f_domain.project(per_marginal), fact.values, np)

        for pro_f_marginal in self.get_project_to_list(per_marginal, data):
            pro_per_marginal = MRF.tools.flatten_to_permutation_marginal(
                pro_f_marginal, data.h_attr_num, data.i_attr_num)
            if pro_per_marginal in per_marginal_dict:
                continue
            pro_f_fact = per_fact.project(pro_f_marginal)
            print(f'        project noisy: {noisy} {f_marginal} -> {pro_f_marginal} -> {pro_per_marginal}')
            pro_per_fact = MRF.Factor(data.f_domain.project(pro_per_marginal), pro_f_fact.values, np)

            self.save_per_marginal(data, pro_per_fact, noisy=noisy)

        projectable_set.remove(per_marginal)

    def get_project_to_list(self, per_marginal, data: Data):
        res_list = set()

        for length in range(1, len(per_marginal)+1):
            for marginal in itertools.combinations(per_marginal, length):
                marginal = tuple(sorted(marginal))
                if not data.size_attr in marginal:
                    continue
                ind_list = [data.get_individual(attr) for attr in marginal]
                ind_list = list(sorted(set(ind_list)))
                if -1 in ind_list:
                    ind_list.remove(-1)
                
                # all 1 order marginal, household marginal
                if len(ind_list) <= 1:
                    res_list.add(marginal)
                # all prefix marginal
                elif ind_list == list(range(max(ind_list)+1)):
                    res_list.add(marginal)
                # ind 2 + ind a marginal
                elif len(ind_list) == 2 and 1 in ind_list:
                    res_list.add(marginal)

        return res_list

    def serialize(self):
        state = {
            'per_marginal_dict':            dict(self.per_marginal_dict),
            'flatten_marginal_dict':        dict(self.flatten_marginal_dict),
            'noisy_per_marginal_dict':      dict(self.noisy_per_marginal_dict),
            'noisy_flatten_marginal_dict':  dict(self.noisy_flatten_marginal_dict),
            'noisy_projectable_set':        self.noisy_projectable_set,
            'projectable_set':              self.projectable_set,
            'size_bins':                    self.size_bins
        }
        return state
    
    def deserialize(self, manager, state):
        self.per_marginal_dict              = manager.dict(state['per_marginal_dict'])
        self.flatten_marginal_dict          = manager.dict(state['flatten_marginal_dict'])
        self.noisy_per_marginal_dict        = manager.dict(state['noisy_per_marginal_dict'])
        self.noisy_flatten_marginal_dict    = manager.dict(state['noisy_flatten_marginal_dict'])
        self.noisy_projectable_set          = state['noisy_projectable_set']
        self.projectable_set                = state['projectable_set']
        self.size_bins                      = state['size_bins']

# Note: unordered marginals of small groups can not be used for large groups
# For example, we count marginals (ind1.A, ind2.A) from groups that are all [a, b]
# Then we get a marginal where P(ind1.A = b | ind2.A = a) = 1. (Other cells are similar)
# But we can not find a distribution of size 3 groups that are consistent with it.
# Even if its ind1-ind2 marginal may satisfy the marginal, 
# its ind2-ind3 marginal and in1-ind3 marginal can not all satisfy that.
# So we have to clear low ind marginal cells when learning high ind attrs.
def legalize_unordered_marginal(fact: MRF.Factor, current_ind, size_attr):
    fact = fact.copy()
    marginal = fact.domain.attr_list
    if current_ind <= 0:
        return fact
    assert(size_attr in marginal)
    idx = [slice(None)]*fact.values.ndim
    size_axis = marginal.index(size_attr)
    idx[size_axis] = list(range(current_ind+1))
    idx = tuple(idx)
    fact.values[idx] = 0

    return fact

class PrivPetal:
    def __init__(self):
        pass

    def init(self, data: Data, config) -> None:
        default_config = {
            'data_name':        'N/A',
            'epsilon':          'N/A',
            'exp_name':         'N/A',

            'max_clique_size':              1e7,
            'max_parameter_size':           3e7,

            'PrivMRF_clique_size':          1e7,
            'PrivMRF_max_parameter_size':   3e7,

            'max_attr_num':                 5,
            'marginal_query_num':           2,
            'structure_learning_it':        10,
            'norm_query_number':            400,

            'estimate_iter_num':            1000,
            'existing_estimate_iter_num':   1000,
            'final_estimate_iter_num':      3000,

            'sample_number':                None,

            'existing_marginal_query_num':      8,
            'existing_structure_learning_it':   8,

            'size_bins':                    [], 
            
            'budget':                       None,
            'R_score_budget':               0.05,
            'group_size_budget':            0.05,
            'init_budget':                  0.45,
            'refine_budget':                0.45,

            'max_dom_limit':                5e4,        
            'theta':                        6,

            'last_size_weight_clip':        30,

            # The number of groups are removed/inserted in neighboring databases
            # Should be 1 if the household table is the primary private relation
            'sensitivity':                  1,

            'query_iter_num':               None,

            # Sensitivity of group_size_hist.
            'group_size_sensitivity':       1,

            'clique_attr_num':              14,
            'single_clique':                False,
            'calculate_edge':               True,
            'learn_MRF':                    True
        }
        self.config = copy.deepcopy(default_config)
        for item in config:
            self.config[item] = copy.deepcopy(config[item])

        self.data = data
        self.manager = multiprocessing.Manager()
        self.shared_marginal = SharedMarginal()
        self.shared_marginal.init(self.manager, self.config['size_bins'])

        group_size_budget = self.config['budget'] * self.config['group_size_budget']
        noise = MRF.tools.get_Gaussian_noise(self.config['group_size_sensitivity'], group_size_budget, 1)
        print('group_size_hist noise: {:.4f}'.format(noise))
        print('group_size_hist:')
        print(self.data.group_size_hist)
        self.data.group_size_hist = self.data.group_size_hist + np.random.normal(scale=noise, size=data.group_size_hist.shape)
        self.data.group_size_hist[self.data.group_size_hist<0] = 0.5*noise
        self.data.group_size_hist[0] = 0
        
        self.data.noisy_total = np.sum(self.data.group_size_hist)
        print('total {:.4e} -> {:.4e}, error: {:.4f}'.format(self.data.total, self.data.noisy_total, abs(self.data.noisy_total-self.data.total)/self.data.total))

        print(self.data.group_size_hist)
        print('domain:')
        print(self.data.f_domain)

        group_size_fact = MRF.Factor(self.data.f_domain.project((self.data.size_attr,)), self.data.group_size_hist, np)
        self.shared_marginal.save_per_marginal(self.data, group_size_fact, noisy=True)
        
        edge_score_path = './temp/PrivPetal_'+self.config['exp_name']+'_edge_score.json'
        if self.config['calculate_edge'] or (not self.config['calculate_edge'] and not os.path.exists(edge_score_path)):
            R_score_budget = self.config['budget']*self.config['R_score_budget']
            edge_scores = utils.privacy.get_permutation_edge_score(R_score_budget, self.data.permutations, 
                self.data.weights, self.data.f_domain, self.data.h_attr_num, self.data.i_attr_num,
                self.data.sample_size, sensitivity=self.config['sensitivity']*2)
            edge_scores.sort(key=lambda x: x[1], reverse=True)
            json.dump(edge_scores, open(edge_score_path, 'w'))


        self.edge_scores = json.load(open(edge_score_path))
        self.edge_scores = [[tuple(item[0]), item[1], item[2]] for item in self.edge_scores]
        self.edge_dict = {tuple(edge): noisy_score/self.data.noisy_total for edge, noisy_score, true_score in self.edge_scores}


    def init_marginals(self):
        ##########################################################
        # household inter-attribute correlation marginals
        budget = self.config['budget']*self.config['init_budget'] / 4
        attrs = list(range(self.data.h_attr_num))
        print('init household marginals')

        if self.data.h_attr_num > 1:
            self.h_model = PrivMRF(self.data, self.shared_marginal, self.config, attrs, self.edge_scores, self.edge_dict, budget)
            print('current per marginals:', len(self.shared_marginal.noisy_per_marginal_dict))
            print(self.shared_marginal.noisy_per_marginal_dict.keys())
            print()

            start_time = time.time()
            self.h_syn_data = self.h_model.generate_synthetic_data()
            print('generate h_syn_data time:', time.time()-start_time)
        else:
            print('h data only have one attribute')
            marginal_noise = MRF.tools.get_Gaussian_noise(self.config['sensitivity'], budget, query_num=1)
            print('marginal_noise:', marginal_noise)
            dom_limit = self.data.noisy_total / (marginal_noise * (2/math.pi)**0.5) / self.config['theta']
            print('dom_limit: {:.4e}'.format(dom_limit))
            add_learn_marginal([(0,),], MRF.Potential({}), [(0,),],
            self.data, self.shared_marginal, marginal_noise, 0, [set([0,]),])
            print('current per marginals:', len(self.shared_marginal.noisy_per_marginal_dict))
            print(self.shared_marginal.noisy_per_marginal_dict.keys())
            print()

            fact = self.shared_marginal.get_noisy_flatten_marginal(self.data, (0,), marginal_noise)
            self.h_syn_data = MRF.tools.generate_column_data(fact.values, self.data.noisy_total).reshape((-1, 1))

        ##########################################################
        # individual inter-attribute correlation marginals
        if self.config['learn_MRF']:
            budget = self.config['budget']*self.config['init_budget'] / 4
            attrs = list(range(self.data.h_attr_num-1, self.data.h_attr_num+self.data.i_attr_num))
        
            print('init individual marginals')
            model = PrivMRF(self.data, self.shared_marginal, self.config, attrs, self.edge_scores, self.edge_dict, budget)
            print('current per marginals:', len(self.shared_marginal.noisy_per_marginal_dict))
            print(self.shared_marginal.noisy_per_marginal_dict.keys())
            
            syn_data = model.generate_synthetic_data()
            syn_df = pd.DataFrame(syn_data, columns=list(range(syn_data.shape[1])))
            # syn_df.to_csv('./temp/i_df.csv', index=False)

        ##########################################################
        # inter-relational correlation marginals
        budget = self.config['budget']*self.config['init_budget'] / 4
        attrs = list(range(self.data.h_attr_num+self.data.i_attr_num))
        marginal_noise = MRF.tools.get_Gaussian_noise(self.config['sensitivity'], budget, query_num=len(attrs))
        print('init inter-relational correlation marginals:')
        print('marginal_noise:', marginal_noise)

        candidate_list = self.get_candidiate_list(marginal_noise, attrs)
        init_marginal_list = MRF.select_CFS_marginal(candidate_list, attrs, self.edge_dict)
        print(init_marginal_list)
        for marginal in init_marginal_list:
            self.shared_marginal.get_noisy_flatten_marginal(self.data, marginal, marginal_noise)

        print('current per marginals:', len(self.shared_marginal.noisy_per_marginal_dict))
        print(self.shared_marginal.noisy_per_marginal_dict.keys())

        ##########################################################
        # intra-group correlation marginals
        budget = self.config['budget']*self.config['init_budget'] / 4
        attrs = list(range(self.data.h_attr_num+self.data.i_attr_num,
            self.data.h_attr_num+self.data.i_attr_num*self.data.tuple_num)) # >=1th individual have intra-group marginals
        marginal_noise = MRF.tools.get_Gaussian_noise(self.config['sensitivity'], budget, query_num=len(attrs))
        dom_limit = self.data.noisy_total / (marginal_noise * (2/math.pi)**0.5) / self.config['theta']
        dom_limit = min(dom_limit, self.config['max_dom_limit'])
        print(f'init intra-group correlation marginals:')
        print('marginal_noise:', marginal_noise)
        print('dom_limit: {:.4e}'.format(dom_limit))

        for ind in range(1, self.data.tuple_num):
            print(f'intra-group correlation marginals: ind: {ind}')
            attrs = list(range(self.data.h_attr_num+self.data.i_attr_num*ind,
                self.data.h_attr_num+self.data.i_attr_num*(ind+1)))
            candidate_list = self.get_intra_candidate_list(dom_limit, ind)
            init_marginal_list = MRF.select_CFS_marginal(candidate_list, attrs, self.edge_dict)
            print(init_marginal_list)
            for marginal in init_marginal_list:
                self.shared_marginal.get_noisy_flatten_marginal(self.data, marginal, marginal_noise)

            print('current per marginals:', len(self.shared_marginal.noisy_per_marginal_dict))
            print(self.shared_marginal.noisy_per_marginal_dict.keys())

        print('localtime:', MRF.tools.get_time())

    def get_intra_candidate_list(self, dom_limit, ind):
        attrs = list(range(self.data.h_attr_num, self.data.h_attr_num+self.data.i_attr_num*(ind+1)))
        size_attr_size = self.data.f_domain.dict[self.data.size_attr]['size']

        # get all marginals containing size_attr and satisfying theta
        candidate_list = MRF.tools.get_candidiate_list(
            attrs, self.data.f_domain, dom_limit/size_attr_size, self.config['max_attr_num']-1)
        candidate_list = [tuple(sorted(item+(self.data.size_attr,))) for item in candidate_list]

        # get marginals involving at least ind+1 individuals
        # ind1: find marginals involving 2 individuals
        # ind2: find marginals involving 3 individuals 
        candidate_list = [item for item in candidate_list if self.data.get_individual_num(item) >= ind+1]
        candidate_list = [item for item in candidate_list if self.data.get_individual_num(item) <= self.data.tuple_num]

        # remove queried marginals
        candidate_list = [marginal for marginal in candidate_list if marginal not in self.shared_marginal.noisy_flatten_marginal_dict]

        return candidate_list

    def get_candidiate_list(self, marginal_noise, attrs):
        dom_limit = self.data.noisy_total / (marginal_noise * (2/math.pi)**0.5) / self.config['theta']
        dom_limit = min(dom_limit, self.config['max_dom_limit'])
        print('dom_limit: {:.4e}'.format(dom_limit))

        # get all marginals containing size_attr and satisfying theta
        candidate_list = MRF.tools.get_candidiate_list(attrs, self.data.f_domain, dom_limit, self.config['max_attr_num'])
        candidate_list = [item for item in candidate_list if self.data.size_attr in item]
        candidate_list = [item for item in candidate_list if self.data.get_individual_num(item) <= self.data.tuple_num]

        # remove queried marginals
        candidate_list = [marginal for marginal in candidate_list if marginal not in self.shared_marginal.noisy_flatten_marginal_dict]

        return candidate_list
    
    def get_next_attr(self, ind, sampled_attrs):
        assert(len(sampled_attrs) >= self.data.h_attr_num)
        candidate_list = set(self.data.f_domain.attr_list) - sampled_attrs
        candidate_list = [attr for attr in candidate_list if self.data.get_individual(attr) == ind]

        candidate_dict = {attr: 0 for attr in candidate_list}
        for target in candidate_dict:
            for feature in sampled_attrs:
                candidate_dict[target] += self.edge_dict[tuple(sorted([target, feature]))]
            candidate_dict[target] /= self.data.f_domain.dict[target]['size'] ** 0.5

        candidate = list(candidate_dict.items())
        candidate.sort(key=lambda x: x[1], reverse=True)
        # print('    next attr candidate:')
        # print('   ', candidate[:10])

        return candidate[0][0]
    
    def get_attr_cliques(self, ind, target_attr, sampled_attrs):
        possible_clique_attrs = sampled_attrs.copy()
        possible_clique_attrs.remove(self.data.size_attr)
        size_attr_size = self.data.f_domain.dict[self.data.size_attr]['size']
        target_size = self.data.f_domain.dict[target_attr]['size']

        clique_attr_num = self.config['clique_attr_num']

        if self.config['single_clique']:
            possible_clique_attrs =  sort_attr_by_score(self.edge_scores, target_attr, possible_clique_attrs)
            clique_attrs = get_clique_by_size(possible_clique_attrs, self.data.f_domain,
                self.config['max_clique_size']/size_attr_size/target_size)
            clique_attrs = clique_attrs + [self.data.size_attr, target_attr]
        else:
            same_ind_attr = [temp_attr for temp_attr in possible_clique_attrs if self.data.get_individual(temp_attr) == ind]
            other_ind_attr = [temp_attr for temp_attr in possible_clique_attrs if self.data.get_individual(temp_attr) != ind]
            same_ind_attr = sort_attr_by_score(self.edge_scores, target_attr, same_ind_attr)[:int(clique_attr_num/2)]
            other_ind_attr = sort_attr_by_score(self.edge_scores, target_attr, other_ind_attr)[:clique_attr_num-len(same_ind_attr)]

            print('    same_ind_attr:', same_ind_attr)
            print('    other_ind_attr:', other_ind_attr)
            clique_attrs = same_ind_attr + other_ind_attr
            clique_attrs = clique_attrs + [self.data.size_attr, target_attr]

        clique_attrs = sorted(clique_attrs)

        return clique_attrs

    def main(self, existing_h_data=None, process_num=4):
        if existing_h_data is None:
            if self.config['sample_number'] is None:
                total = self.data.noisy_total
            else:
                total = self.config['sample_number']
        else:
            total = len(existing_h_data)
        total = int(total)

        width = len(self.data.h_size_domain)+len(self.data.i_domain)*self.data.sample_size
        synthetic_data = -np.ones(shape=(total, width), dtype=int)

        if existing_h_data is None:
            synthetic_data[:, :self.h_syn_data.shape[1]] = self.h_syn_data
        else:
            # sample h size by existing h attrs
            print('existing h data: {}, missing: {}'.format(
                existing_h_data.shape, list(range(existing_h_data.shape[1], self.h_syn_data.shape[1]))))
            # print(existing_h_data.shape, self.data.h_data.shape)
            assert(existing_h_data.shape[1] == self.data.h_data.shape[1]-1)
            h_syn_data = -np.ones(shape=(total, self.data.h_data.shape[1]), dtype=int)
            h_syn_data[:, :existing_h_data.shape[1]] = existing_h_data
            h_syn_data = self.h_model.generate_synthetic_data_by_substitution(
                h_syn_data, target_attr=existing_h_data.shape[1], max_workers=40)

            synthetic_data[:, :h_syn_data.shape[1]] = h_syn_data
            

        sampled_attrs = set(range(self.data.h_attr_num))
        if self.config['query_iter_num'] is None:
            query_iter_num = len(self.data.f_domain) - len(sampled_attrs)
        else:
            query_iter_num = self.config['query_iter_num']
        budget = self.config['budget'] * self.config['refine_budget'] / query_iter_num
        
        pool = multiprocessing.Pool(processes=process_num)
        num_gpus = cp.cuda.runtime.getDeviceCount()
        print('num_gpus:', num_gpus)
        gpu_id = 0
        sampled_cnt = 0

        if process_num > 1:
            data_path = './temp/'+self.config['exp_name']+'_data.pkl'
            pickle.dump(self.data, open(data_path, 'wb'), protocol=4)

            result_list = []
            for ind in range(self.data.sample_size):
            # for ind in range(1):
                print('individual:', ind)
                for _ in range(self.data.i_attr_num):
                    sampled_cnt += 1
                    if sampled_cnt > query_iter_num:
                        budget = None
                    print('    budget:', budget)
                    target_attr = self.get_next_attr(ind, sampled_attrs)
                    clique_attrs = self.get_attr_cliques(ind, target_attr, sampled_attrs)
                    print('target_attr: {} <- clique_attrs: {}\n'.format(target_attr, clique_attrs))
                    
                    if self.config['learn_MRF']:
                        result = pool.apply_async(unordered_MRF, args=(target_attr, clique_attrs, 
                            data_path, self.shared_marginal, self.config, self.edge_scores,
                            self.edge_dict, gpu_id, budget))
                    else:
                        result = None

                    model_path = './temp/'+self.config['exp_name']+'_'+str(target_attr)+'.mdl'
                    result_list.append((result, target_attr, model_path))
                    
                    gpu_id += 1
                    gpu_id %= num_gpus
                    sampled_attrs.add(target_attr)

            for result, target_attr, model_path in result_list:
                if result is not None:
                    result.get()

            pool.close()
            pool.join()

            for result, target_attr, model_path in tqdm(result_list, desc="Sampling attrs", unit="attr"):
            # for _, target_attr, model_path in result_list:
                model = MRF.MarkovRandomField.load_model(path=model_path)

                existing_attrs = model.domain.attr_list.copy()
                existing_attrs.remove(target_attr)

                print(synthetic_data.shape)

                ind = self.data.get_individual(target_attr)
                if ind >= 1:
                    pos = synthetic_data[:, self.data.size_attr] > ind
                    print('tuples to sample:', pos.sum())
                    synthetic_data[pos] = model.generate_synthetic_data_by_substitution(synthetic_data[pos], target_attr, max_workers=30)
                else:
                    synthetic_data = model.generate_synthetic_data_by_substitution(synthetic_data, target_attr, max_workers=30)

            for result, target_attr, model_path in result_list:
                os.remove(model_path)

            os.remove(data_path)
        else:
            result_list = []
            
            for ind in range(self.data.sample_size):
                print('individual:', ind)
                for _ in range(self.data.i_attr_num):
                    sampled_cnt += 1
                    if sampled_cnt > query_iter_num:
                        budget = 0
                    print('    budget:', budget)
                    target_attr = self.get_next_attr(ind, sampled_attrs)
                    clique_attrs = self.get_attr_cliques(ind, target_attr, sampled_attrs)
                    print('target_attr: {} <- clique_attrs: {}\n'.format(target_attr, clique_attrs))

                    model_path = './temp/'+self.config['exp_name']+'_'+str(target_attr)+'.mdl'
                    if self.config['learn_MRF']:
                        unordered_MRF(target_attr, clique_attrs, self.data, self.shared_marginal, self.config, self.edge_scores, self.edge_dict, 0, budget)
                    result_list.append((target_attr, model_path))
                    sampled_attrs.add(target_attr)
            
            print('start generating synthetic data')
            
            for target_attr, model_path in result_list:
                    model = MRF.MarkovRandomField.load_model(path=model_path)
                    existing_attrs = model.domain.attr_list.copy()
                    existing_attrs.remove(target_attr)

                    ind = self.data.get_individual(target_attr)
                    if ind >= 1:
                        pos = synthetic_data[:, self.data.size_attr] > ind
                        print('pos:', pos.sum())
                        synthetic_data[pos] = model.generate_synthetic_data_by_substitution(synthetic_data[pos], target_attr, max_workers=40)
                    else:
                        synthetic_data = model.generate_synthetic_data_by_substitution(synthetic_data, target_attr, max_workers=40)
                    
            for target_attr, model_path in result_list:
                os.remove(model_path)
                    
        return synthetic_data

    # output synthetic referenced table (with the FK) and referencing table (with the FK and its primary key)
    def run(self, data: Data, config, existing_h_data_with_key=None, process_num=4):
        print('localtime:', MRF.tools.get_time())
        self.init(data, config)
        self.init_marginals()

        if existing_h_data_with_key is None:
            syn_data = self.main(process_num=process_num)
        else:
            syn_data = self.main(existing_h_data_with_key[:, 1:], process_num=process_num)
            
        syn_df = pd.DataFrame(syn_data, columns=list(range(syn_data.shape[1])))
        syn_df.to_csv('./temp/'+config['exp_name']+'_syn.csv', index=False)

        syn_df = pd.read_csv('./temp/'+config['exp_name']+'_syn.csv')
        syn_data = syn_df.to_numpy()

        # model_path = './temp/'+config['exp_name']+'.mdl'
        # self.save_model(model_path)

        np.random.shuffle(syn_data)
        if existing_h_data_with_key is None:
            syn_h_data, syn_i_data = recover_FK_data(syn_data, data.i_domain, data.h_domain)
        else:
            # sort existing_h_data_with_key and syn_data such that they are corresponding to each other
            existing_h_data_with_key = existing_h_data_with_key[MRF.tools.sort_data_by_cols(existing_h_data_with_key[:, 1:])]
            syn_data = syn_data[MRF.tools.sort_data_by_cols(syn_data[:, :existing_h_data_with_key.shape[1]-1])]
            assert((syn_data[:, :existing_h_data_with_key.shape[1]-1] == existing_h_data_with_key[:, 1:]).all())
            h_key_array = existing_h_data_with_key[:, 0]

            # then use the keys in existing_h_data_with_key to generate the keys in syn_h_data, syn_i_data
            syn_h_data, syn_i_data = recover_FK_data(syn_data, data.i_domain, data.h_domain, h_key_array=h_key_array)

        syn_i_data = syn_i_data[np.argsort(syn_i_data[:, -1])]
        syn_h_data = syn_h_data[np.argsort(syn_h_data[:, 0])]
        
        print('localtime:', MRF.tools.get_time())

        # return h data with PK, i data with FK and PK
        return syn_h_data, syn_i_data
    

    def save_model(self, path):
        with open(path, 'wb') as out_file:
            print('save model:', path)

            pickle.dump(self.config, out_file)
            pickle.dump(self.data, out_file, protocol=4)
            pickle.dump(self.edge_dict, out_file)
            pickle.dump(self.edge_scores, out_file)
            pickle.dump(self.h_syn_data, out_file, protocol=4)

            pickle.dump(self.shared_marginal.serialize(), out_file)

    @classmethod
    def load_model(cls, path) -> 'PrivPetal':
        with open(path, 'rb') as out_file:
            print('load model:', path)

            model               = cls()
            model.config        = pickle.load(out_file)
            model.data          = pickle.load(out_file)
            model.edge_dict     = pickle.load(out_file)
            model.edge_scores   = pickle.load(out_file)
            model.h_syn_data    = pickle.load(out_file)

            # can not save/load/send manager directly
            model.manager = multiprocessing.Manager()
            shared_marginal_state = pickle.load(out_file)
            model.shared_marginal = SharedMarginal()
            model.shared_marginal.deserialize(model.manager, shared_marginal_state)

            return model

def check_marginal_dict(marginal_dict, data: Data, shared_marginal: SharedMarginal, current_ind):
    for f_marginal in marginal_dict:
        data_fact = shared_marginal.get_flatten_marginal(data, f_marginal)
        data_fact = legalize_unordered_marginal(data_fact, current_ind, data.size_attr)

        fact = marginal_dict[f_marginal]
        fact = legalize_unordered_marginal(fact, current_ind, data.size_attr)

        error = MRF.tools.get_TVD_count(data_fact.values, fact.values)/data_fact.values.sum()
        total = fact.values.sum()
        dom = fact.domain.size()

        print('marginal: {}, total: {:.4e}, dom: {:.4e}, query tvd: {:.4f}'.format(
            f_marginal, total, dom, error))
    print('')
        
def unordered_MRF(target_attr, clique_attrs, data_path, shared_marginal: SharedMarginal, config, edge_scores, edge_dict, gpu_id, budget):
    device = cp.cuda.Device(gpu_id)
    device.use()


    log_file = open('./log/'+config['exp_name']+'_'+str(target_attr)+'.txt', 'w')
    temp_out = sys.stdout
    sys.stdout = log_file

    if isinstance(data_path, str):
        data: Data = pickle.load(open(data_path, 'rb'))
    else:
        data: Data = data_path

    print ("local_time:", MRF.tools.get_time())
    print('target_attr: {} <- clique_attrs: {}'.format(target_attr, clique_attrs))

    if target_attr is None:
        current_ind = 0
    else:
        current_ind = data.get_individual(target_attr)
    size_attr_size = data.f_domain.dict[data.size_attr]['size']

    config = copy.deepcopy(config)
    print('current_ind:', current_ind)
    if current_ind + 1 == size_attr_size - 1:
        config['weight_clip'] = config['last_size_weight_clip']

    # get attribute graph
    if config['single_clique']:
        graph = nx.complete_graph(clique_attrs)
        junction_tree = MRF.tools.get_junction_tree(graph)

    else:
        graph = MRF.AttributeGraph(config, data.f_domain.project(clique_attrs))
        init_graph = nx.Graph()
        for attr in clique_attrs:
            if attr != target_attr:
                init_graph.add_edge(attr, target_attr)
            if attr != data.size_attr:
                init_graph.add_edge(attr, data.size_attr)

        graph, junction_tree = graph.local_search_by_R(data=None, edge_list=edge_scores, init_graph=init_graph)

    mrf_domain = data.f_domain.project(clique_attrs)

    print('current_ind:', current_ind)
    print('total_budget:', config['budget'])
    print('mrf budget:', budget)

    # get candidate list
    existing_marginal_set = set(shared_marginal.noisy_flatten_marginal_dict.keys())
    print('existing_marginal_set:', len(existing_marginal_set))
    max_attr_num = config['max_attr_num']

    
    existing_marginal_set = [set(item) for item in existing_marginal_set]
    clique_set = [set(clique) for clique in junction_tree.nodes]

    existing_candidate_list = []
    for item in existing_marginal_set:
        for clique in clique_set:
            if item.issubset(clique):
                existing_candidate_list.append(item)
                break
    existing_candidate_list = [tuple(sorted(item)) for item in existing_candidate_list]
    existing_candidate_list = list(set(existing_candidate_list))
    print('existing_candidate_list:', len(existing_candidate_list))
    print(existing_candidate_list)


    # learn an MRF model
    total = np.sum(data.group_size_hist[current_ind+1:])
    print('total: {:.4f}'.format(total))
    model = MRF.MarkovRandomField(total, mrf_domain, graph, junction_tree, config=config)
    learn_marginal_dict = MRF.Potential({})
    clique_list = [set(clique) for clique in junction_tree.nodes]

    existing_candidate_list = list(existing_candidate_list)
    init_marginal_list = MRF.select_CFS_marginal(
        existing_candidate_list, mrf_domain.attr_list, edge_dict) # reuse marginals
    add_learn_marginal(init_marginal_list, learn_marginal_dict, existing_candidate_list,
        data, shared_marginal, None, current_ind, clique_list)

         
    learn_marginal_dict.to_gpu()
    model.estimate_parameters(learn_marginal_dict, config['existing_estimate_iter_num'])

    marginal_num_list = MRF.tools.get_select_marginal_schedule(
        config['existing_marginal_query_num'], config['existing_structure_learning_it'])
    unordered_structure_learning(model, data, learn_marginal_dict, shared_marginal, existing_candidate_list,
        marginal_num_list, None, None, config['norm_query_number'], 
        current_ind, config['existing_estimate_iter_num'], h_sensitivity=config['sensitivity'])

    if not budget is None:
        marginal_budget = 0.80 * budget
        new_h_budget = 0.2 * budget
        marginal_query_num = config['marginal_query_num']
        per_marginal_noise = MRF.tools.get_Gaussian_noise(config['sensitivity'], marginal_budget, query_num=marginal_query_num)

        dom_limit = data.noisy_total / (per_marginal_noise*config['theta']*(2/math.pi)**0.5)
        dom_limit = min(dom_limit, config['max_dom_limit'])

        print('marginal_query_num:', marginal_query_num)
        print('per_marginal_noise: {:.4f}'.format(per_marginal_noise))
        print('theta:', config['theta'])
        print('max dom_limit: {:.4e}'.format(dom_limit))
    
        candidate_list = MRF.tools.get_candidate_list_of_cliques(junction_tree.nodes, data.f_domain, dom_limit, max_attr_num)
        candidate_list = [item for item in candidate_list if data.size_attr in item]
        candidate_list = [item for item in candidate_list if data.get_individual_num(item) <= data.tuple_num]
        new_candidate_list = []
        for item in candidate_list:
            if item not in existing_candidate_list:
                new_candidate_list.append(item)
        new_candidate_list = list(set(new_candidate_list))
        print('new_candidate_list:', len(new_candidate_list))
        print(new_candidate_list)

        marginal_num_list = MRF.tools.get_select_marginal_schedule(
            marginal_query_num, config['structure_learning_it'])
        unordered_structure_learning(model, data, learn_marginal_dict, shared_marginal, new_candidate_list, 
            marginal_num_list, per_marginal_noise, new_h_budget, config['norm_query_number'],
            current_ind, config['estimate_iter_num'], h_sensitivity=config['sensitivity'])
    
    model_path = './temp/'+config['exp_name']+'_'+str(target_attr)+'.mdl'
    model.save_model(model_path)
    

    print ("local_time:", MRF.tools.get_time())
    sys.stdout = temp_out
    return target_attr, model_path

def get_all_flatten_marginal(f_marginal, data: Data, shared_marginal: SharedMarginal,
                            per_marginal_noise, clique_list):
    per_marginal = MRF.tools.flatten_to_permutation_marginal(
        f_marginal, data.h_attr_num, data.i_attr_num)
    per_marginal_list = MRF.tools.permutation_to_all_permutation_marginal(
        data.h_attr_num, data.i_attr_num, per_marginal, data.f_domain)
    res_dict = {}
    for per_marginal in per_marginal_list:
        f_marginal_list = MRF.tools.permutation_to_flatten_marginal(
            per_marginal, data.h_attr_num, data.i_attr_num, list(range(data.sample_size)))
        f_marginal_list = [temp_f_marginal for temp_f_marginal \
            in f_marginal_list if MRF.tools.in_clique(temp_f_marginal, clique_list)]
        for temp_f_marginal in f_marginal_list:
            fact = shared_marginal.get_noisy_flatten_marginal(data, temp_f_marginal, per_marginal_noise)
            res_dict[temp_f_marginal] = fact
    
    return per_marginal, res_dict

# prepare flatten marginals and learn them
def add_learn_marginal(marginal_list, learn_marginal_dict, candidate_list,
                    data, shared_marginal, per_marginal_noise, current_ind, clique_list):
    for f_marginal in marginal_list:
        _, new_dict = get_all_flatten_marginal(
            f_marginal, data, shared_marginal, per_marginal_noise, clique_list)
        print(f'    marginal {f_marginal} ->')
        for other_f_marginal, fact in new_dict.items():
            fact = legalize_unordered_marginal(fact, current_ind, data.size_attr)
            learn_marginal_dict[other_f_marginal] = fact
            print(f'         {other_f_marginal}, total: {fact.values.sum()}')
            if other_f_marginal in candidate_list:
                candidate_list.remove(other_f_marginal)

def unordered_structure_learning(model: MRF.MarkovRandomField, data: Data, learn_marginal_dict: MRF.Potential,
    shared_marginal: SharedMarginal, candidate_list: list, marginal_num_list, per_marginal_noise,
    h_budget, norm_query_number, current_ind, iter_num, h_sensitivity=1):

    clique_list = [set(clique) for clique in model.junction_tree.nodes]
    for it in range(len(marginal_num_list)):
        select_num = marginal_num_list[it]
        print(f'structure learning it: {it}/{len(marginal_num_list)}')
        if len(candidate_list) == 0:
            continue

        # sample candidates to query
        if len(candidate_list) >= norm_query_number:
            query_marginal_list = np.random.choice(len(candidate_list), size=norm_query_number, replace=False)
            query_marginal_list = [candidate_list[idx] for idx in query_marginal_list]
        else:
            query_marginal_list = candidate_list

        if h_budget is None:
            h_noise = 0
        else:
            h_noise = MRF.tools.get_Gaussian_noise(h_sensitivity, h_budget/len(marginal_num_list),
                query_num=len(query_marginal_list))
        print('select {} from {}, h_noise: {}, ratio: {}'.format(
            select_num, len(candidate_list), h_noise, h_noise/data.noisy_total))

        # get errors of candidates
        model_marginal_dict, _ = model.cal_marginal_dict(model.potential, query_marginal_list, to_cpu=True)
        error_list = []
        for f_marginal, model_fact in model_marginal_dict.items():
            data_fact = shared_marginal.get_flatten_marginal(data, f_marginal)
            data_fact = legalize_unordered_marginal(data_fact, current_ind, data.size_attr)
            error = MRF.tools.get_TVD_count(data_fact.values, model_fact.values)
            error += np.random.normal(scale=h_noise)
            error_list.append((f_marginal, error, error/data.noisy_total))
        error_list.sort(key=lambda x: x[1], reverse=True)
        for marginal, error, rel_error in error_list[:5]:
            print('    {} error: {:.4f}, relative error: {:.4f}'.format(marginal, error, rel_error))
        select_list = error_list[:select_num]
        select_list = [item[0] for item in select_list]

        add_learn_marginal(select_list, learn_marginal_dict, candidate_list,
            data, shared_marginal, per_marginal_noise, current_ind, clique_list)
        
        learn_marginal_dict.to_gpu()
        model.estimate_parameters(learn_marginal_dict, iter_num)

        # check tvd
        model_marginal_dict, _ = model.cal_marginal_dict(model.potential, list(learn_marginal_dict.keys()), to_cpu=True)
        max_tvd = 0
        for f_marginal, model_fact in model_marginal_dict.items():
            data_fact = shared_marginal.get_flatten_marginal(data, f_marginal)
            data_fact = legalize_unordered_marginal(data_fact, current_ind, data.size_attr)
            error = MRF.tools.get_TVD_count(data_fact.values, model_fact.values)/model.total
            print('marginal: {}, total: {:.4f}, {:.4f}, true tvd: {:.4f}'.format(
                f_marginal, float(data_fact.values.sum()), float(model_fact.values.sum()), error))
            max_tvd = max(max_tvd, error)
        print('max true tvd: {:.4f}\n'.format(max_tvd))

        if error_list[0][1] < 0.01:
            break

def PrivMRF(data: Data, shared_marginal: SharedMarginal, config, mrf_attrs, 
            edge_scores, edge_dict, budget):
    size_attr_size = data.f_domain.dict[data.size_attr]['size']
    x_size_config = config.copy()
    x_size_config['max_clique_size']    = x_size_config['PrivMRF_clique_size'] / size_attr_size
    x_size_config['max_parameter_size'] = x_size_config['PrivMRF_max_parameter_size'] / size_attr_size
    x_size_config['marginal_query_num'] = int(0.5 * len(mrf_attrs))
    x_size_config['max_attr_num']       = x_size_config['max_attr_num'] - 1

    mrf_domain = data.f_domain.project(mrf_attrs)
    attrs_without_size = mrf_domain.attr_list.copy()
    attrs_without_size.remove(data.size_attr)

    print('mrf_attrs:', mrf_attrs)
    print('mrf_domain:')
    print(mrf_domain)

    # every clique should contain size_attr
    graph = MRF.AttributeGraph(x_size_config, data.f_domain.project(attrs_without_size))
    graph, junction_tree = graph.local_search_by_R(data=None, edge_list=edge_scores)
    for attr in attrs_without_size:
        graph.add_edge(attr, data.size_attr)
    junction_tree = MRF.tools.get_junction_tree(graph)

    # budget allocation and noise calculation
    marginal_budget = budget * 0.9
    norm_budget     = budget * 0.1
    marginal_noise              = MRF.tools.get_Gaussian_noise(x_size_config['sensitivity'], marginal_budget, query_num=len(mrf_attrs)+x_size_config['marginal_query_num'])
    dom_limit                   = data.noisy_total / (marginal_noise * (2/math.pi)**0.5) / x_size_config['theta'] / size_attr_size
    dom_limit = min(dom_limit, x_size_config['max_dom_limit'] / size_attr_size)
    print('init marginal_noise: {:.4f}'.format(marginal_noise))
    print('max dom_limit: {:.4e}'.format(dom_limit*size_attr_size))

    model = MRF.MarkovRandomField(
        data.noisy_total, mrf_domain, graph, junction_tree, config=x_size_config)
    model.marginal_noise = marginal_noise # for loss estimation

    candidate_list = []
    for clique in junction_tree.nodes:
        candidate_list.extend(MRF.tools.get_candidiate_list(
            clique, mrf_domain, dom_limit, x_size_config['max_attr_num']))
    candidate_list = [list(item) for item in candidate_list]
    for item in candidate_list:
        item.append(data.size_attr)
    candidate_list = [tuple(sorted(set(item))) for item in candidate_list]
    candidate_list = list(set(candidate_list))

    learn_marginal_dict = MRF.Potential({})
    clique_list = [set(clique) for clique in junction_tree.nodes]

    init_marginal_list = MRF.select_CFS_marginal(candidate_list, mrf_domain.attr_list, edge_dict)
    add_learn_marginal(init_marginal_list, learn_marginal_dict, candidate_list,
            data, shared_marginal, marginal_noise, 0, clique_list)
    
    learn_marginal_dict.to_gpu()
    model.estimate_parameters(learn_marginal_dict, x_size_config['estimate_iter_num'])

    marginal_num_list = MRF.tools.get_select_marginal_schedule(
        x_size_config['marginal_query_num'], x_size_config['structure_learning_it'])
    unordered_structure_learning(model, data, learn_marginal_dict, shared_marginal, candidate_list,
        marginal_num_list, marginal_noise, norm_budget, x_size_config['norm_query_number'], 0,
        x_size_config['estimate_iter_num'], h_sensitivity=x_size_config['sensitivity'])

    return model
