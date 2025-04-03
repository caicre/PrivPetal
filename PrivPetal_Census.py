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
import PrivPetal
from PrivPetal import Data
import argparse

def none_or_int(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' must be an integer or 'none'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrivPetal Census Data')
    parser.add_argument('--exp_prefix', help='Experiment name prefix', default="exp0")
    parser.add_argument('--epsilon', type=float, help='Privacy parameter epsilon', default=3.20)
    parser.add_argument('--data_name', help='Name of the dataset', default="California")
    parser.add_argument('--max_group_size', type=int, help='Maximum size of groups', default=8)
    parser.add_argument('--tuple_num', type=int, help='Number of tuples', default=3)
    parser.add_argument('--fact_threshold', type=none_or_int, help='Factorization threshold (integer or none)', default=20) 
    parser.add_argument('--fact_size', type=none_or_int, help='Factorization size (integer or none)', default=10) # Use None for fact_threshold and fact_size if you are not sure whether to factorize
    parser.add_argument('--process_num', type=int, help='Number of processes', default=4)
    
    multiprocessing.set_start_method('spawn', force=True)
    args = parser.parse_args()
    print(args)

    epsilon         = args.epsilon
    exp_name        = args.exp_prefix + '_{:.2f}'.format(epsilon)
    data_name       = args.data_name
    max_group_size  = args.max_group_size
    tuple_num       = args.tuple_num
    fact_size       = args.fact_size
    fact_threshold  = args.fact_threshold
    
    data = Data.load_data('./data/'+data_name)

    theta = 6
    if  epsilon < 0.41:
        theta = 3
    budget = MRF.tools.get_privacy_budget(epsilon, delta=1/len(data.i_df))
    print('total budget: {:.8f}'.format(budget))
    config = {
        'exp_name':     exp_name,
        'data_name':    data_name,
        'budget':       budget,
        'theta':        6,

        'init_budget':      0.6,
        'refine_budget':    0.3,
        'size_bins':        list(range(6, max_group_size+1)),
        'query_iter_num':   len(data.i_domain) * tuple_num,

        'max_clique_size':              3e6,
        'max_parameter_size':           1e7,

        'PrivMRF_clique_size':          1e7,
        'PrivMRF_max_parameter_size':   3e7,
    }
    if epsilon < 0.41:
        config['existing_structure_learning_it'] = 5
        config['theta'] = 3
        config['size_bins'] = list(range(5, max_group_size+1))

    if fact_threshold is not None and fact_size is not None:
        h_fact_attr_list = [attr for attr in data.h_domain.attr_list if data.h_domain.dict[attr]['size'] > fact_threshold]
        i_fact_attr_list = [attr for attr in data.i_domain.attr_list if data.i_domain.dict[attr]['size'] > fact_threshold]

        fact_data, h_attr_to_new_attr, i_attr_to_new_attr = data.factorize(h_fact_attr_list, fact_size, i_fact_attr_list, fact_size)

        fact_data.get_group_data([-1,], max_group_size)
        fact_data.prepare_data(max_group_size, max_group_size, tuple_num)

        model = PrivPetal.PrivPetal()
        syn_h_data, syn_i_data = model.run(fact_data, config, process_num=args.process_num)

        o_syn_i_data = utils.tools.factorize_back(syn_i_data[:, 1:-1], i_attr_to_new_attr, fact_size, data.i_domain)
        syn_i_data = np.concatenate([syn_i_data[:, [0,]], o_syn_i_data, syn_i_data[:, [-1,]]], axis=1)

        o_syn_h_data = utils.tools.factorize_back(syn_h_data[:, 1:], h_attr_to_new_attr, fact_size, data.h_domain)
        syn_h_data = np.concatenate([syn_h_data[:, [0,]], o_syn_h_data], axis=1)
    
    else:
        model = PrivPetal.PrivPetal()
        syn_h_data, syn_i_data = model.run(data, config, process_num=args.process_num)

    syn_h_path = './temp/PrivPetal_'+exp_name+'_'+data_name+'_household.csv'
    syn_i_path = './temp/PrivPetal_'+exp_name+'_'+data_name+'_individual.csv'

    syn_h_df = pd.DataFrame(syn_h_data, columns=data.h_df.columns)
    print('write:', syn_h_path)
    syn_h_df.to_csv(syn_h_path, index=False)

    syn_i_df = pd.DataFrame(syn_i_data, columns=data.i_df.columns)
    print('write:', syn_i_path)
    syn_i_df.to_csv(syn_i_path, index=False)

    print ("local_time:", MRF.tools.get_time())
