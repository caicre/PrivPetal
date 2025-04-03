import PrivPetal
import pandas as pd
import numpy as np
import MRF
import sys
import multiprocessing
import json
import cupy as cp
import os
import networkx as nx
from TPC_H_data_proc import proc_order_back, proc_lineitem_back
import argparse


def match_data(data1, data2, common_cols=None, shuffle=True):
    assert(data1.shape == data2.shape)
    assert(data1.ndim == 2)

    if shuffle:
        # The rows in data1, data2 may not be random, and thus can introduce 
        # unexpected correlations afer matching. Should shuffle in advance.
        np.random.shuffle(data1)
        np.random.shuffle(data2)
    if common_cols is None:
        common_cols = list(range(data1.shape[1]))
    if len(common_cols) == 0:
        return data1, data2

    def group_rows_by_column(data, col):
        groups = {}
        for row in data:
            key = row[col]
            if key not in groups:
                groups[key] = []
            groups[key].append(row)
        groups = {key: np.stack(group, axis=0) for key, group in groups.items()}
        return groups

    def cut_data(data1, data2):
        if data1.shape[0] > data2.shape[0]:
            return data1[:data2.shape[0]], data2, data1[data2.shape[0]:], None
        else:
            return data1, data2[:data1.shape[0]], None, data2[data1.shape[0]:]

    col = common_cols[0]
    group1 = group_rows_by_column(data1, col)
    group2 = group_rows_by_column(data2, col)

    key_set = set(group1.keys()).union(set(group2.keys()))

    residue_list1 = []
    residue_list2 = []
    result_list1 = []
    result_list2 = []
    for key in key_set:

        if not key in group1:
            residue_list2.append(group2[key])
            continue
        if not key in group2:
            residue_list1.append(group1[key])
            continue

        key_data1, key_data2, residue1, residue2 = cut_data(group1[key], group2[key])
        residue_list1.append(residue1)
        residue_list2.append(residue2)

        res1, res2 = match_data(key_data1, key_data2, common_cols[1:], shuffle=False)
        result_list1.append(res1)
        result_list2.append(res2)

    residue_list1 = [item for item in residue_list1 if not item is None]
    residue_list2 = [item for item in residue_list2 if not item is None]

    if len(residue_list1) > 0:
        residue1 = np.concatenate(residue_list1, axis=0)
        residue2 = np.concatenate(residue_list2, axis=0)
        residue1, residue2 = match_data(residue1, residue2, common_cols[1:], shuffle=False)

        result_list1.append(residue1)
        result_list2.append(residue2)

    return np.concatenate(result_list1, axis=0), np.concatenate(result_list2, axis=0)


class TPC_H_data():

    def load_data(self, path):
        l_df = pd.read_csv(os.path.join(path, 'lineitem.csv'))
        o_df = pd.read_csv(os.path.join(path, 'orders.csv'))
        c_df = pd.read_csv(os.path.join(path, 'customer.csv'))
        ps_df = pd.read_csv(os.path.join(path, 'partsupp.csv'))

        l_domain = json.load(open(os.path.join(path, 'lineitem_domain.json')))
        o_domain = json.load(open(os.path.join(path, 'orders_domain.json')))
        c_domain = json.load(open(os.path.join(path, 'customer_domain.json')))
        ps_domain = json.load(open(os.path.join(path, 'partsupp_domain.json')))

        l_domain = MRF.tools.get_domain_by_attrs(l_domain, l_df.columns[1:-3])
        o_domain = MRF.tools.get_domain_by_attrs(o_domain, o_df.columns[1:-1])
        c_domain = MRF.tools.get_domain_by_attrs(c_domain, c_df.columns[1:])
        ps_domain = MRF.tools.get_domain_by_attrs(ps_domain, ps_df.columns[2:])

        # print('l_df:')
        # print(l_df)

        # print('o_df:')
        # print(o_df)

        # print('c_df:')
        # print(c_df)

        # print('ps_df:')
        # print(ps_df)

        self.l_df, self.o_df, self.c_df, self.ps_df = l_df, o_df, c_df, ps_df
        self.l_domain, self.o_domain, self.c_domain, self.ps_domain = l_domain, o_domain, c_domain, ps_domain

    def get_l_o_data(self):
        col = list(self.l_df.columns)[:-2]
        l_df = self.l_df[col]

        col = list(self.o_df.columns)[:-1]
        o_df = self.o_df[col]
        return PrivPetal.Data(o_df, self.o_domain, l_df, self.l_domain)
    
    def get_o_c_data(self):
        # Assume that we know what C_CUSTKEY have groups
        # This can be done by adding a group_size col to the customer table, 
        # And then synthesize the group_size col with DP.
        # Should get the valid custkey with above method in the sampling stage
        valid_c_set = set(self.o_df['O_CUSTKEY'])
        idx = self.c_df['C_CUSTKEY'].isin(valid_c_set)
        valid_c_df = self.c_df[idx]
        x_valid_c_df = self.c_df[~idx]
        print('valid_c_df:', valid_c_df.shape)
        print('x_valid_c_df:', x_valid_c_df.shape)

        o_df = self.o_df.sort_values(by=['O_CUSTKEY', 'O_ORDERKEY'])

        return PrivPetal.Data(valid_c_df, self.c_domain, o_df, self.o_domain), x_valid_c_df
    
    def get_l_ps_data(self):

        max_suppkey = max(self.ps_df['PS_SUPPKEY'])+1

        col = list(self.l_df.columns)[:-3]
        col.extend(self.l_df.columns[-2:])
        l_df = self.l_df[col]
        l_df['L_PSKEY'] = l_df['L_PARTKEY'] * max_suppkey + l_df['L_SUPPKEY']
        l_df = l_df.drop(columns=['L_PARTKEY', 'L_SUPPKEY'])

        l_df = l_df.sort_values(by=['L_PSKEY', 'LINEITEMKEY'])

        ps_df = self.ps_df.copy()
        ps_df['PS_PSKEY'] = ps_df['PS_PARTKEY'] * max_suppkey + ps_df['PS_SUPPKEY']
        ps_df = ps_df.drop(columns=['PS_PARTKEY', 'PS_SUPPKEY'])

        col = ['PS_PSKEY',]
        col.extend(list(ps_df.columns)[:-1])
        ps_df = ps_df[col]

        valid_ps_set = set(l_df['L_PSKEY'])
        idx = ps_df['PS_PSKEY'].isin(valid_ps_set)
        valid_ps_df = ps_df[idx]
        x_valid_ps_df = ps_df[~idx]
        print('valid_ps_df:', valid_ps_df.shape)
        print('x_valid_ps_df:', x_valid_ps_df.shape)

        return PrivPetal.Data(valid_ps_df, self.ps_domain, l_df, self.l_domain)

    def recover_ps_key(self, syn_data):

        max_suppkey = max(self.ps_df['PS_SUPPKEY'])+1

        p_key, s_key = np.divmod(syn_data[:, -1], max_suppkey)
        # print('max_suppkey', max_suppkey)

        syn_data = syn_data[:, :-1]
        syn_data = np.concatenate([np.arange(syn_data.shape[0]).reshape((-1, 1)), 
                                   syn_data, p_key.reshape((-1, 1)), s_key.reshape((-1, 1))], axis=1)
        syn_df = pd.DataFrame(syn_data, columns=list(self.l_df.columns))

        return syn_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--exp_prefix', type=str, help='Experiment prefix')
    parser.add_argument('--epsilon', type=float, help='Epsilon value')
    parser.add_argument('--data_name', type=str, help='Data name')
    parser.add_argument('--tuple_num', type=int, help='Number of permutation tuples')
    parser.add_argument('--process_num', type=int, help='Number of processes', default=4)

    args = parser.parse_args()
    print(args)

    g_epsilon           = args.epsilon
    g_exp_name          = f"{args.exp_prefix}_{g_epsilon:.2f}"
    g_data_name         = args.data_name
    g_tuple_num         = args.tuple_num
    g_process_num       = args.process_num

    if not os.path.exists('./temp/'):
        os.mkdir('./temp/')

    multiprocessing.set_start_method('spawn', force=True)

    print('localtime:', MRF.tools.get_time())
    print('exp_name:', g_exp_name)
    print('epsilon:', g_epsilon)
    print('data_name:', g_data_name)

    data = TPC_H_data()
    data.load_data(os.path.join('./data/', g_data_name))

    g_budget = MRF.tools.get_privacy_budget(g_epsilon, delta=1/len(data.l_df))
    # g_budget = 1e9 # debug
    # g_budget = 0.421   # eps=3.2
    # g_budget = 0.03231 # eps=0.8
    print('total budget: {:.8f}'.format(g_budget))
    o_c_budget = g_budget * 0.5
    l_o_budget = g_budget * 0.25
    l_ps_budget = g_budget * 0.25

    # order - customer PrivPetal
    o_c_max_group_size = 10
    o_c_config = {
        'exp_name':     g_exp_name+'_o_c',
        'data_name':    g_data_name,
        'budget':       o_c_budget,
        'size_bins':    list(range(1, o_c_max_group_size+1)),
        'calculate_edge':   True,
        'theta':        0.25,

        'existing_marginal_query_num':  4,
        'marginal_query_num':           1,

        'group_size_sensitivity':       2,

        'max_clique_size':              3e6,
        'max_parameter_size':           1e7,
        'PrivMRF_clique_size':          3e6,
        'PrivMRF_max_parameter_size':   1e7,

        'max_attr_num':                 3
    }
    o_c_sample_size = o_c_max_group_size
    o_c_data, x_valid_c_df = data.get_o_c_data()
    o_c_data.get_group_data([-1,], o_c_sample_size)
    o_c_data.prepare_data(o_c_sample_size, o_c_max_group_size, g_tuple_num)

    model = PrivPetal.PrivPetal()
    syn_c_data, syn_o_data = model.run(o_c_data, o_c_config, o_c_data.h_data, process_num=g_process_num)

    syn_o_df = pd.DataFrame(syn_o_data, columns=list(data.o_df.columns))
    syn_o_df.to_csv('./temp/'+o_c_config['exp_name']+'_syn_o.csv', index=False)
    syn_o_df = pd.read_csv('./temp/'+o_c_config['exp_name']+'_syn_o.csv')
    syn_o_data = syn_o_df.to_numpy()

    # lineitem - order PrivPetal
    l_o_config = {
        'exp_name':     g_exp_name+'_l_o',
        'data_name':    g_data_name,
        'budget':       l_o_budget,
        'size_bins':    [],
        'calculate_edge':   True,
        'theta':        5,

        'existing_marginal_query_num':  4,
        'marginal_query_num':           1,

        'max_clique_size':              3e6,
        'max_parameter_size':           1e7,
        'PrivMRF_clique_size':          3e6,
        'PrivMRF_max_parameter_size':   1e7,

        'max_attr_num':                 3
    }
    l_o_max_group_size = 7
    l_o_sample_size = l_o_max_group_size
    l_o_data = data.get_l_o_data()
    l_o_data.get_group_data([-1,], l_o_sample_size)
    l_o_data.prepare_data(l_o_sample_size, l_o_max_group_size, g_tuple_num)

    model = PrivPetal.PrivPetal()
    _, syn_l_data = model.run(l_o_data, l_o_config, syn_o_data[:, :-1], process_num=g_process_num)

    syn_l_df = pd.DataFrame(syn_l_data, columns=list(range(syn_l_data.shape[1])))
    syn_l_df.to_csv('./temp/'+l_o_config['exp_name']+'_syn_l.csv', index=False)
    syn_l_df = pd.read_csv('./temp/'+l_o_config['exp_name']+'_syn_l.csv')
    syn_l_data = syn_l_df.to_numpy()

    # lineitem - partsupp PrivPetal
    l_ps_max_group_size = 8
    if g_epsilon < 0.81:
        l_ps_max_group_size = 7
    l_ps_config = {
        'exp_name':     g_exp_name+'_l_ps',
        'data_name':    g_data_name,
        'budget':       l_ps_budget,
        'size_bins':    list(range(1, l_ps_max_group_size+1)),
        'calculate_edge':   True,
        'theta':        0.20,

        'existing_marginal_query_num':  4,
        'marginal_query_num':           1,

        # Inserting/removing 1 order may insert/remove l_o_max_group_size=7 lineitems
        # And thus, insert/remove l_o_max_group_size=7 tuples in the permutation data of l_ps_data
        # Inserting/removing 1 tuple can only add/reduce 1 when counting in the permutation data
        'sensitivity':                  7, 
        'group_size_sensitivity':       14,

        'max_clique_size':              3e6,
        'max_parameter_size':           1e7,
        'PrivMRF_clique_size':          3e6,
        'PrivMRF_max_parameter_size':   1e7,

        'max_attr_num':                 3
    }
    l_ps_sample_size = l_ps_max_group_size
    l_ps_data = data.get_l_ps_data()
    l_ps_data.get_group_data([-1,], l_ps_sample_size)
    l_ps_data.prepare_data(l_ps_sample_size, l_ps_max_group_size, g_tuple_num)

    model = PrivPetal.PrivPetal()
    _, syn_l_data_ps = model.run(l_ps_data, l_ps_config, l_ps_data.h_data, process_num=g_process_num)

    syn_l_df_ps = pd.DataFrame(syn_l_data_ps, columns=list(range(syn_l_data_ps.shape[1])))
    syn_l_df_ps.to_csv('./temp/'+l_ps_config['exp_name']+'_syn_l.csv', index=False)
    syn_l_df_ps = pd.read_csv('./temp/'+l_ps_config['exp_name']+'_syn_l.csv')
    syn_l_data_ps = syn_l_df_ps.to_numpy()

    # match synthetic lineitem data to merge the order FK and the partsupp FK
    np.random.shuffle(syn_l_data)
    np.random.shuffle(syn_l_data_ps)
    length = min(syn_l_data.shape[0], syn_l_data_ps.shape[0])
    syn_l_data = syn_l_data[:length]
    syn_l_data_ps = syn_l_data_ps[:length]
    
    syn_l_data = syn_l_data[:, 1:]
    syn_l_data_ps = syn_l_data_ps[:, 1:]

    syn_l_data, syn_l_data_ps = match_data(syn_l_data, syn_l_data_ps, list(range(syn_l_data.shape[1]-1)))

    syn_l_data = np.concatenate([syn_l_data, syn_l_data_ps[:, -1].reshape((-1, 1))], axis=1)
    syn_l_df = data.recover_ps_key(syn_l_data)

    # postprocess
    origin_o_df = pd.read_csv(os.path.join('./data/input_data/', g_data_name, 'orders.tbl'), delimiter='|', header=None)
    origin_o_df = origin_o_df[list(origin_o_df.columns)[:-1]]
    origin_l_df = pd.read_csv(os.path.join('./data/input_data/', g_data_name, 'lineitem.tbl'), delimiter='|', header=None)
    origin_l_df = origin_l_df[list(origin_l_df.columns)[:-1]]
    origin_p_df = pd.read_csv(os.path.join('./data/input_data/', g_data_name, 'part.tbl'), delimiter='|', header=None)
    origin_p_df = origin_p_df[list(origin_p_df.columns)[:-2]]

    syn_o_df = proc_order_back(syn_o_df, origin_o_df.to_numpy())
    syn_l_df = proc_lineitem_back(syn_l_df, syn_o_df, origin_l_df.to_numpy(), origin_p_df.to_numpy())
    syn_o_df.to_csv('./temp/'+g_exp_name+'_syn_o.csv', index=False)
    syn_l_df.to_csv('./temp/'+g_exp_name+'_syn_l.csv', index=False)
