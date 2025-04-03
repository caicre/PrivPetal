import PrivPetal as PP
import pandas as pd
import numpy as np
import MRF
import sys
import multiprocessing
import json
import cupy as cp
import os
import networkx as nx
import argparse
import time
from tqdm import tqdm
import utils.tools as tools
from utils.instacart_proc import get_department_to_aisle_dict

class InstacartData():
    def load_data(self, path, frac=1, random_state=1):
        od_df = pd.read_csv(os.path.join(path, 'order_products.csv'))
        o_df = pd.read_csv(os.path.join(path, 'orders.csv'))
        u_df = pd.read_csv(os.path.join(path, 'users.csv'))

        # randomly sample percentile of users
        if frac < 1:
            u_df = u_df.sample(frac=frac, random_state=random_state)

            u_set = set(u_df.iloc[:, 0])
            idx = o_df.iloc[:, -1].apply(lambda x: x in u_set)
            o_df = o_df[idx]

            o_set = set(o_df.iloc[:, 0])
            idx = od_df.iloc[:, -1].apply(lambda x: x in o_set)
            od_df = od_df[idx]

        print(od_df.columns, od_df.shape)
        print(od_df.head())

        print(o_df.columns, o_df.shape)
        print(o_df.head())

        print(u_df.columns, u_df.shape)
        print(u_df.head())

        od_domain = json.load(open(os.path.join(path, 'order_products_domain.json')))
        o_domain = json.load(open(os.path.join(path, 'orders_domain.json')))
        u_domain = json.load(open(os.path.join(path, 'users_domain.json')))


        od_domain = MRF.tools.get_domain_by_attrs(od_domain, od_df.columns[1:-1])
        o_domain = MRF.tools.get_domain_by_attrs(o_domain, o_df.columns[1:-1])
        u_domain = MRF.tools.get_domain_by_attrs(u_domain, u_df.columns[1:])

        print(od_domain)
        print(o_domain)
        print(u_domain)

        self.od_df, self.o_df, self.u_df = od_df, o_df, u_df
        self.od_domain, self.o_domain, self.u_domain = od_domain, o_domain, u_domain

    def get_od_o_data(self):
        od_df = self.od_df
        col = self.o_df.columns[:-1]
        o_df = self.o_df[col]

        valid_o_set = set(od_df.iloc[:, -1])
        idx = o_df.iloc[:, 0].apply(lambda x: x in valid_o_set)
        valid_o_df = o_df[idx]
        not_vailid_o_df = o_df[~idx]

        # print(valid_o_df)

        return PP.Data(valid_o_df, self.o_domain, od_df, self.od_domain), not_vailid_o_df
    
    def get_o_u_data(self):
        o_df = self.o_df
        u_df = self.u_df

        valid_u_set = set(o_df.iloc[:, -1])
        idx = u_df.iloc[:, 0].apply(lambda x: x in valid_u_set)
        valid_u_df = u_df[idx]
        not_valid_u_df = u_df[~idx]

        return PP.Data(valid_u_df, self.u_domain, o_df, self.o_domain), not_valid_u_df

    def downsample(self, od_o_max_group_size, o_u_max_group_size, random_state=42):
        print('Downsampling o_df')
        start_time = time.time()

        data = MRF.tools.downsample_group(self.o_df.to_numpy(), list(self.o_df.columns).index('user_id'), o_u_max_group_size, random_state=random_state, return_group_data=False)
        self.o_df = pd.DataFrame(data, columns=self.o_df.columns)

        print('elapsed time:', time.time()-start_time)
        print('Filtering od_df')
        start_time = time.time()
        self.od_df = self.od_df[self.od_df['order_id'].isin(set(self.o_df['order_id']))]
        print('elapsed time:', time.time()-start_time)
        print('Downsampling od_df')
        start_time = time.time()

        data = MRF.tools.downsample_group(self.od_df.to_numpy(), list(self.od_df.columns).index('order_id'), od_o_max_group_size, random_state=random_state, return_group_data=False)
        self.od_df = pd.DataFrame(data, columns=self.od_df.columns)

        print('elapsed time:', time.time()-start_time)
        print('Downsampled shapes:')
        print(self.o_df.shape, self.od_df.shape)

    
    def remove_unused_data(self):
        print('Removing unused data')

        o_set = set(self.od_df['order_id'])
        self.o_df = self.o_df[self.o_df['order_id'].isin(o_set)]

        u_set = set(self.o_df['user_id'])
        self.u_df = self.u_df[self.u_df['user_id'].isin(u_set)]

        print('Filtered shapes:')
        print(self.o_df.shape, self.od_df.shape)

def process_o_data_back(syn_o_df):
    syn_o_df['order_hour_of_day'] = syn_o_df['order_hour_of_day_part1'] * 6 + syn_o_df['order_hour_of_day_part2']
    syn_o_df['days_since_prior_order'] = syn_o_df['days_since_prior_order_part1'] * 10 + syn_o_df['days_since_prior_order_part2']
    

    assert (syn_o_df['days_since_prior_order'] >= 0).all()

    syn_o_df['days_since_prior_order'] = np.where(syn_o_df['days_since_prior_order'] >= 40, -1, syn_o_df['days_since_prior_order']) # invalid value

    syn_o_df['days_since_prior_order'] = np.where(
        (syn_o_df['days_since_prior_order'] > 30) & (syn_o_df['days_since_prior_order'] < 40),
        30,
        syn_o_df['days_since_prior_order']
    ) # valid but > 30 should be clipped

    syn_o_df['days_since_prior_order'] = np.where(syn_o_df['days_since_prior_order'] == -1, 31, syn_o_df['days_since_prior_order'])

    syn_o_df = syn_o_df[['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'user_id']]

    return syn_o_df

def generate_depart_product_by_aisle(aisle_depart_product_df, syn_od_df):
    
    pos_list = np.random.choice(syn_od_df.shape[0], 30, replace=False)
    print('pos_list:', pos_list)
    print(syn_od_df.iloc[pos_list].to_string())

    department_idx_to_aisle_dict, aisle_to_idx_dict = get_department_to_aisle_dict(aisle_depart_product_df)
    print(syn_od_df.columns)
    syn_od_df['department_id'] = syn_od_df['department_id'] + 1

    ########################################################
    # generate aisle_id by department_id, aisle_id_idx

    # print department_id to aisle_id_idx distribution
    department_to_aisle_idx = syn_od_df.groupby('department_id')['aisle_id_idx'].apply(lambda x: x.value_counts().sort_index())
    for department_id, aisle_counts in department_to_aisle_idx.groupby(level=0):
        print(f"Department ID: {department_id}")
        for aisle_id_idx, count in aisle_counts.items():
            print(f"  Aisle ID Index: {aisle_id_idx}, Count: {count}")
        print()
    
    # generate aisle_id by department_id, aisle_id_idx
    start_time = time.time()
    print('generate aisle_id by department_id, valid aisle_id_idx')

    department_aisle_map = {
        (department_id, idx): aisle_id
        for department_id, aisle_ids in department_idx_to_aisle_dict.items()
        for idx, aisle_id in enumerate(aisle_ids)
    }
    department_ids = syn_od_df['department_id'].values
    aisle_id_indices = syn_od_df['aisle_id_idx'].values
    def fast_lookup(department_id, aisle_id_idx):
        return department_aisle_map.get((department_id, aisle_id_idx), -1)
    vectorized_lookup = np.vectorize(fast_lookup)
    syn_od_df['aisle_id'] = vectorized_lookup(department_ids, aisle_id_indices)

    print('Time elapsed:', time.time() - start_time)
    print('pos_list:', pos_list)
    print(syn_od_df.iloc[pos_list].to_string())

    # get department_id to aisle_id distribution
    department_to_aisle = aisle_depart_product_df.groupby(['department_id', 'aisle_id'], as_index=False)
    count_sums = department_to_aisle['count'].sum().rename(columns={'count': 'total_count'})
    department_to_aisle_dict = (
        count_sums.groupby('department_id')
        .apply(lambda df: df.set_index('aisle_id')['total_count'].to_dict())
        .to_dict()
    )
    print(department_to_aisle_dict)

    # generate aisle_id by department_id for invalid aisle_id_idx
    start_time = time.time()
    print('generate aisle_id by department_id, invalid aisle_id_idx')
    def flatten_aisle_id_prob(group):
        idx = group.name
        choices = list(department_to_aisle_dict[idx].keys())
        probs = np.array(list(department_to_aisle_dict[idx].values()))
        print(probs)
        
        probs = MRF.tools.random_round(probs, group.shape[0])
        vals = MRF.tools.expand_int_prob(probs)
        vals = np.array([choices[i] for i in vals])

        group['aisle_id'] = vals
        return group

    invalid_aisle_idx_df = syn_od_df[syn_od_df['aisle_id'] == -1]
    valid_aisle_idx_df = syn_od_df[syn_od_df['aisle_id'] != -1]
    invalid_aisle_idx_df = (
        invalid_aisle_idx_df.groupby('department_id', group_keys=False)
        .apply(flatten_aisle_id_prob)
    )
    syn_od_df = pd.concat([valid_aisle_idx_df, invalid_aisle_idx_df]).sort_index()

    print('Time elapsed:', time.time() - start_time)
    print('pos_list:', pos_list)
    print(syn_od_df.iloc[pos_list])
    # syn_od_df = syn_od_df.reset_index(drop=True)
    

    ########################################################
    # generate product_id by aisle_id
    start_time = time.time()
    print('generate product_id by aisle_id')

    aisle_to_product = aisle_depart_product_df.groupby(['aisle_id'], as_index=False)
    count_sums = aisle_to_product['count'].sum().rename(columns={'count': 'total_count'})
    aisle_to_product = pd.merge(aisle_depart_product_df, count_sums, on=['aisle_id'], how='left')
    aisle_to_product['prob'] = aisle_to_product['count'] / aisle_to_product['total_count']
    aisle_to_product = aisle_to_product[['aisle_id', 'product_id', 'prob']]
    aisle_to_product = (
        aisle_to_product.groupby(['aisle_id'])
        .apply(lambda df: (df['product_id'].values, df['prob'].values))
        .to_dict()
    )

    def flatten_prob(group):
        idx = group.name
        choices = aisle_to_product[idx][0]
        probs = aisle_to_product[idx][1]
        
        probs = MRF.tools.random_round(probs, group.shape[0])
        vals = MRF.tools.expand_int_prob(probs)
        vals = np.array([choices[i] for i in vals])

        group['product_id'] = vals
        return group

    syn_od_df = syn_od_df.groupby(['aisle_id']).apply(flatten_prob)


    print('Time elapsed:', time.time() - start_time)
    print('pos_list:', pos_list)
    print(syn_od_df.iloc[pos_list])
    

    return syn_od_df

if __name__ == '__main__':
    print('localtime:', MRF.tools.get_time())
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--exp_prefix', type=str, help='Experiment prefix')
    parser.add_argument('--epsilon', type=float, help='Epsilon value')
    parser.add_argument('--data_name', type=str, help='Data name')
    parser.add_argument('--tuple_num', type=int, help='Number of permutation tuples')
    parser.add_argument('--process_num', type=int, help='Number of processes', default=4)

    args = parser.parse_args()
    print(args)
    
    g_epsilon = args.epsilon
    g_exp_name = f"{args.exp_prefix}_{g_epsilon:.2f}"
    g_data_name = args.data_name
    g_tuple_num = args.tuple_num
    g_process_num = args.process_num

    data = InstacartData()
    data.load_data(f'data/{g_data_name}')
    # data.load_data(f'data/{g_data_name}', frac=0.01)
    
    g_budget = MRF.tools.get_privacy_budget(g_epsilon, delta=1/len(data.od_df))
    # g_budget = 0.36568
    # g_budget = 3656.8
    print('budget:', g_budget)
    o_u_budget = 0.5 * g_budget
    od_o_budget = 0.5 * g_budget

    o_u_max_group_size = 30
    # o_u_sample_size = 45
    o_u_sample_size = 60
    od_o_max_group_size = 20
    # od_o_sample_size = 26
    od_o_sample_size = 30

    o_u_exp_name = g_exp_name+'_o_u'
    od_o_exp_name = g_exp_name+'_od_o'

    # order - user
    o_u_data, not_valid_u_df = data.get_o_u_data()
    print('not valid u:')
    print(not_valid_u_df)

    o_u_config = {
        'exp_name':     o_u_exp_name,
        'data_name':    g_data_name,
        'budget':       o_u_budget,
        # 'size_bins':    list(range(int(o_u_max_group_size/2), o_u_max_group_size+1)),
        'size_bins':    list(range(7, o_u_sample_size+1)),
        'calculate_edge':   True,

        'theta':                        1,
        'existing_marginal_query_num':  6,
        'marginal_query_num':           1,
        'query_iter_num':               len(o_u_data.i_domain) * g_tuple_num,

        'sensitivity':                  1,
        'group_size_sensitivity':       1,

        'max_clique_size':              3e6,
        'max_parameter_size':           1e7,
        'PrivMRF_clique_size':          3e6,
        'PrivMRF_max_parameter_size':   1e7,

        'max_attr_num':                 3,
        'clique_attr_num':              20,

        'R_score_budget':               0.05,
        'group_size_budget':            0.05,
        'init_budget':                  0.7,
        'refine_budget':                0.2
    }
    if g_epsilon < 0.21:
        o_u_config['theta'] = 0.15
        o_u_config['size_bins'] = list(range(5, o_u_sample_size+1))
    elif g_epsilon < 0.41:
        o_u_config['theta'] = 0.5
    o_u_data.get_group_data([-1,], o_u_sample_size)
    o_u_data.prepare_data(o_u_sample_size, o_u_max_group_size, g_tuple_num)

    model = PP.PrivPetal()
    syn_u_data, syn_o_data = model.run(o_u_data, o_u_config, process_num=g_process_num)
    syn_u_df = pd.DataFrame(syn_u_data, columns=o_u_data.h_df.columns)
    syn_u_df.to_csv(f'./temp/PrivPetal_'+g_exp_name+'_u.csv', index=False)
    syn_o_df = pd.DataFrame(syn_o_data, columns=o_u_data.i_df.columns)
    syn_o_df.to_csv(f'./temp/'+o_u_exp_name+'_o.csv', index=False)

    syn_u_df = pd.read_csv(f'./temp/PrivPetal_'+g_exp_name+'_u.csv')
    syn_o_df = pd.read_csv(f'./temp/'+o_u_exp_name+'_o.csv')
    syn_u_data = syn_u_df.to_numpy()
    syn_o_data = syn_o_df.to_numpy()
    
    
    # order_product - order
    print('order_product - order')
    od_o_data, not_valid_o_df = data.get_od_o_data()
    print('not valid o:')
    print(not_valid_o_df)
    
    od_o_config = {
        'exp_name':     od_o_exp_name,
        'data_name':    g_data_name,
        'budget':       od_o_budget,
        # 'size_bins':    list(range(int(od_o_max_group_size/2), od_o_max_group_size+1)),
        'size_bins':    list(range(7, od_o_sample_size+1)),
        'calculate_edge':   True,

        'theta':                        0.5,
        'existing_marginal_query_num':  6,
        'marginal_query_num':           1,
        'query_iter_num':               len(od_o_data.i_domain) * g_tuple_num,

        'sensitivity':                  o_u_max_group_size,
        'group_size_sensitivity':       o_u_max_group_size,

        'max_clique_size':              3e6,
        'max_parameter_size':           1e7,
        'PrivMRF_clique_size':          3e6,
        'PrivMRF_max_parameter_size':   1e7,

        'max_attr_num':                 3,
        'clique_attr_num':              20,
        'learn_MRF':                    True,

        'R_score_budget':               0.05,
        'group_size_budget':            0.05,
        'init_budget':                  0.7,
        'refine_budget':                0.2
    }
    od_o_data.get_group_data([-1,], od_o_sample_size)
    od_o_data.prepare_data(od_o_sample_size, od_o_max_group_size, g_tuple_num)

    model = PP.PrivPetal()
    _, syn_od_data = model.run(od_o_data, od_o_config, syn_o_data[:, :-1], process_num=g_process_num)
    syn_od_df = pd.DataFrame(syn_od_data, columns=od_o_data.i_df.columns)
    syn_od_df.to_csv(f'./temp/'+od_o_config['exp_name']+'_syn_od.csv', index=False)

    # post processing
    print('post processing')
    syn_o_df = process_o_data_back(syn_o_df)
    syn_o_df.to_csv(f'./temp/PrivPetal_'+g_exp_name+'_o.csv', index=False)

    aisle_depart_product_df = pd.read_csv(f'data/{g_data_name}/aisle_depart_product.csv')
    syn_od_df = pd.read_csv(f'./temp/'+od_o_config['exp_name']+'_syn_od.csv')
    syn_od_df = generate_depart_product_by_aisle(aisle_depart_product_df, syn_od_df)
    syn_od_df = syn_od_df[[col for col in syn_od_df.columns if col != 'order_id'] + ['order_id']]

    tools.write_large_df(syn_od_df, f'./temp/PrivPetal_'+g_exp_name+'_od.csv')
