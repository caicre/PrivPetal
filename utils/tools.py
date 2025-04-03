import psycopg2
import pandas as pd
import csv
import numpy as np
import os
from io import StringIO
import re
import time
import MRF
import json
from tqdm import tqdm
import itertools
from sqlalchemy import create_engine
import random

def create_database(db_name):
    print('creating db '+db_name)
    conn = psycopg2.connect(database="postgres", user='postgres', \
        password='123456', host='127.0.0.1', port= '5432'
    )
    conn.autocommit = True
    cursor = conn.cursor()

    cursor.execute("SELECT datname FROM pg_database;")
    list_database = cursor.fetchall()
    # print(list_database)

    if (db_name, ) in list_database:
        print('warning: database '+db_name+' already exists. Remove and recreate.')
        sql = 'SELECT pg_terminate_backend(pg_stat_activity.pid)\n'
        sql +='FROM pg_stat_activity\n'
        sql += "WHERE datname='"+ db_name+"' AND pid<>pg_backend_pid();\n"
        cursor.execute(sql)

        sql = '''DROP database '''+db_name+''';'''
        cursor.execute(sql)
        conn.commit()

    sql = '''CREATE database '''+db_name+''';'''
    cursor.execute(sql)

def expand_group_data(group_data, max_group_size):
    res = -np.ones(\
        shape=(len(group_data), max_group_size*group_data[0].shape[1]), \
        dtype=int)
    for i in range(len(group_data)):
        group = group_data[i]
        assert(len(group) <= max_group_size)
        res[i, :group.size] = group.flatten().reshape(1, -1)
    return res

def psql_insert_copy(table, conn, keys, data_iter):
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)

def insert_table_tbl(db_name, df: pd.DataFrame, tbl_name):
    print('inserting ', df.shape, 'as', tbl_name, 'to', db_name)
    engine = create_engine('postgresql://postgres:123456@127.0.0.1:5432/' + db_name)

    col_list = list(df.columns)
    for col in col_list:
        if col != col.lower():
            print(col)
            raise
    df.to_sql(tbl_name, engine, method=psql_insert_copy)

def factorize(target_attrs, max_size, data, domain: MRF.Domain):
    for attr in target_attrs:
        assert(domain.dict[attr]['size'] > max_size)
    # print(data.shape)
    # print(domain.attr_list)
    assert(list(range(data.shape[1])) == domain.attr_list)
    new_var = len(domain)

    attr_to_new_attr = {attr: [attr,] for attr in target_attrs}

    data = data.copy()
    new_data = []
    new_domain = domain.copy()
    for attr in target_attrs:
        quotient, remainder = np.divmod(data[:,attr], max_size)
        data[:, attr] = quotient
        new_data.append(remainder.reshape((-1, 1)))

        attr_to_new_attr[attr].append(new_var)
        new_domain.dict[attr]['size'] = int(max(quotient)+1)
        new_domain.add_variable(new_var, max_size)
        new_var += 1

    if len(target_attrs) > 0:
        new_domain.shape = [new_domain.dict[i]['size'] for i in new_domain.attr_list]
        new_data = np.concatenate(new_data, axis=1)
        new_data = np.concatenate([data, new_data], axis=1)

        return new_data, new_domain, attr_to_new_attr

    else:
        return data, new_domain, attr_to_new_attr

def factorize_back(new_data, attr_to_new_attr, max_size, domain):
    data = new_data[:, :len(domain)].copy()
    for attr, new_attr in attr_to_new_attr.items():
        assert(len(new_attr) == 2)
        col1, col2 = new_attr
        assert(attr == col1)
        col_data = new_data[:, col1]*max_size + new_data[:, col2]
        data[:, attr] = col_data
    return data


def downsample(i_data, max_group_size, i_col, drop=False, h_col=None, h_data=None):
    i_group_data = MRF.tools.get_group_data(i_data, [i_col,])
    length = np.array([len(group) for group in i_group_data])
    hist, _ = np.histogram(length, bins=list(range(20)))
    print('group size hist:')
    print(hist)
    i_data, i_group_data = MRF.tools.down_sample(i_group_data, max_group_size, drop=drop)

    if drop:
        FK_set = set(i_data[:, i_col])
        mask = [FK in FK_set for FK in h_data[:, h_col]]
        h_data = h_data[mask]


    return i_data, i_group_data, h_data

def read_data(data_name):
    # read data
    h_df = pd.read_csv('./data/'+data_name+'/household.csv')
    i_df = pd.read_csv('./data/'+data_name+'/individual.csv')

    # # debug
    # h_df = h_df[['HOUSEHOLD', 'FARM', 'OWNERSHP', 'ACREHOUS', 'PROPINSR', 'ROOMS']]
    # i_df = i_df[['INDIVIDUAL', 'RELATE', 'SEX', 'AGE', 'MARST', 'INCTOT', 'HOUSEHOLD']]

    h_domain = json.load(open('./data/'+data_name+'/household_domain.json'))
    i_domain = json.load(open('./data/'+data_name+'/individual_domain.json'))
    h_domain = MRF.tools.get_domain_by_attrs(h_domain, h_df.columns[1:])
    i_domain = MRF.tools.get_domain_by_attrs(i_domain, i_df.columns[1:-1])
    h_data = h_df.to_numpy()
    i_data = i_df.to_numpy()
    print(h_df)
    print(h_domain)
    print(i_df)
    print(i_domain)

    return h_df, i_df, h_data, i_data, h_domain, i_domain

def get_size_by_quantile(group_size_hist, quantile):
    total = sum([i*group_size_hist[i] for i in range(1, len(group_size_hist))])
    goal = total * quantile
    partial_sum = 0
    for i in range(1, len(group_size_hist)):
        partial_sum += i * group_size_hist[i]
        if partial_sum >= goal:
            return  i
    assert(0)

def are_disjoint(lists):
    total_length = sum(len(lst) for lst in lists)
    unique_elements = set(element for lst in lists for element in lst)
    return total_length == len(unique_elements)

def write_large_df(df, path):
    if os.path.exists(path):
        os.remove(path)
    chunk_size = 10**6 
    with tqdm(total=len(df), desc="Writing chunks", unit="rows") as pbar:
        for i in range(0, len(df), chunk_size):
            df.iloc[i:i + chunk_size].to_csv(path, mode='a', header=(i == 0), index=False)
            pbar.update(chunk_size)

def check_FK(i_group_data, h_data):
    FK_array = np.array([group[0, -1] for group in i_group_data])
    PK_set = set(h_data[:, 0])
    assert all([FK in PK_set for FK in FK_array])

def check_FK_with_order(i_group_data, h_data):
    assert len(i_group_data) == len(h_data)
    FK_array = np.array([group[0, -1] if len(group) > 0 else None for group in i_group_data])
    PK_array = h_data[:, 0]
    res = [FK == PK or FK is None for FK, PK in zip(FK_array, PK_array)]
    assert all(res)

def join_h_and_group_data(group_data, h_data):
    length = sum([len(group) for group in group_data])
    join = np.full((length, group_data[0].shape[1] + h_data.shape[1]), -1, dtype=int)
    idx = 0

    for i in tqdm(range(len(group_data)), desc="Joining Data", unit="groups"):
        group = group_data[i]
        group_size = len(group)
        join[idx:idx+group_size, :group.shape[1]] = group
        join[idx:idx+group_size, group.shape[1]:] = h_data[i].reshape(1, -1)  # broadcast
        idx += group_size

    assert idx == length
    return join

def approximated_self_join(group_data, max_comb_length = 100):
    group_data = [group for group in group_data if len(group) > 1]
    group_length = [len(group) for group in group_data]

    total_length = [size * (size-1) // 2 for size in group_length]
    exceed_num = np.sum(np.array(total_length) > max_comb_length)
    print(f'{exceed_num} groups exceed max_comb_length {max_comb_length}')
    total_length = [min(size, max_comb_length) for size in total_length]
    total_length = sum(total_length)
    print('max_comb_length:', max_comb_length)
    
    join = np.full((total_length, group_data[0].shape[1]*2), -1, dtype=int)

    idx = 0
    weight = []
    for i in tqdm(range(len(group_data)), desc="Joining Data", unit="groups"):
        group = group_data[i]
        group_size = len(group)

        if group_size * (group_size-1) // 2 > max_comb_length:
            comb_list = []
            for i in range(max_comb_length):
                comb_list.append(random.sample(range(group_size), 2))
        else:
            comb_list = list(itertools.combinations(range(group_size), 2))

        for j, k in comb_list:
            join[idx, :group.shape[1]] = group[j]
            join[idx, group.shape[1]:] = group[k]
            idx += 1
        weight.extend([group_size/len(comb_list)] * len(comb_list))

    
    assert idx == total_length
    assert len(weight) == total_length

    return join, weight

def get_marginal_across_two_tables(attr_list, one_table_attr_num, num_way_marginal):

    marginal_list = list(itertools.combinations(attr_list, num_way_marginal))
    marginal_list = [
        marginal for marginal in marginal_list
        if any(attr < one_table_attr_num for attr in marginal) and any(attr >= one_table_attr_num for attr in marginal)
    ]
    marginal_list = [tuple(sorted(marg)) for marg in marginal_list]
    random.shuffle(marginal_list)
    marginal_list = marginal_list[:100]
    
    return marginal_list

def add_empty_group(i_group_data, h_data):
    if len(i_group_data) == len(h_data):
        check_FK_with_order(i_group_data, h_data)
        return i_group_data, h_data
    res = []
    group_idx = 0
    for i in range(len(h_data)):
        if group_idx == len(i_group_data):
            res.append(np.zeros((0, i_group_data[0].shape[1]), dtype=int))
            continue
        if i_group_data[group_idx][0, -1] == h_data[i, 0]:
            res.append(i_group_data[group_idx])
            group_idx += 1
        else:
            res.append(np.zeros((0, i_group_data[0].shape[1]), dtype=int))
        
    res = np.array(res, dtype=object)
    check_FK_with_order(res, h_data)
    return res, h_data
