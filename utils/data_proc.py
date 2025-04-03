import csv
import json
import sys
import os

sys.path.insert(1, os.path.realpath(os.path.curdir))

import pandas as pd
from MRF.domain import Domain
# from domain import Domain
import numpy as np
import os
from MRF.tools import get_group_data

def read_csv(path):
    reader = csv.reader(open(path, 'r'))
    header = next(reader)
    data_list = []
    for row in reader:
        data_list.append(row)
    
    return header, data_list


def compress_household_data(household_data, id_attrs=[0,]):
    res_household_data = []
    res_household_data.append(household_data[0])

    group_length = []

    h_id = household_data[0][id_attrs]
    i = 0
    temp_len = 0
    data_len = len(household_data)
    while i < data_len:
        row = household_data[i]
        if (row[id_attrs] != h_id).any():
            res_household_data.append(row)
            h_id = row[id_attrs]
            group_length.append(temp_len)
            temp_len = 0
        temp_len += 1
        i += 1
    group_length.append(temp_len)

    return np.array(res_household_data, dtype=int), group_length

# Get data with/without a foreign key, sorted in the order of the FK.
# Do not modify the order of the rows, data and group_data must be consistent
# The 0-th col of data is the id_col, and the 1-st col of data is for attr 0.
# The last col of data is for FK.
# Domain is used to identify attributes. Columns not in the domain are ignored.
def read_table(table_path, domain_path, id_col=None, FK_col=None, ratio=1.0, all_attrs=False):
    df = pd.read_csv(table_path)
    # print(df.columns)
    domain_dict = json.load(open(domain_path))

    # print(df.loc[:5])

    attrs = list(domain_dict.keys())

    if id_col is None:
        sorted_cols = []
    else:
        sorted_cols = [id_col,]

    if all_attrs:
        temp_attrs = list(df.columns)
        # print(temp_attrs)
        if not id_col is None:
            temp_attrs.remove(id_col)
        sorted_cols.extend(temp_attrs)
    else:
        sorted_cols.extend(attrs)
        if not FK_col is None:
            sorted_cols.append(FK_col)
    # print(sorted_cols)

    # sort columns in the order of domain keys
    # also let id_col be the first col and FK_col be the last col
    df = df[sorted_cols]

    domain_dict = {i: domain_dict[attrs[i]] for i in range(len(attrs))}
    domain = Domain(domain_dict, list(range(len(attrs))))
    data = df.to_numpy()

    # if id_col is None:
    #     start = 0
    # else:
    #     start = 1
    # for col in range(len(domain)):
    #     print(col, col+start, domain.dict[col], attrs[col])
    #     print(np.sum(data[:, col+start] >= domain.dict[col]['size']))
    #     assert((data[:, col+start] < domain.dict[col]['size']).all())

    if not id_col is None:
        assert(len(np.unique(data[:, 0])) == len(data))
    if FK_col is None:

        if not id_col is None:
            data = data[np.argsort(data[:,sorted_cols.index(id_col)])]

        data_num = int(len(data)*ratio)
        data = data[:data_num]
        if not id_col is None:
            sorted_cols.remove(id_col)

        return data, domain, sorted_cols

    else:
        FK_data = data[:,sorted_cols.index(FK_col)]
        K_data = data[:, 0]

        data = data[np.lexsort((K_data, FK_data))]

        group_data = get_group_data(data, group_id_attrs=[sorted_cols.index(FK_col),])

        group_num = int(len(group_data) * ratio)
        data_num = sum([len(group_data[i]) for i in range(group_num)])

        data = data[:data_num]
        group_data = group_data[:group_num]

        sorted_cols.remove(id_col)
        sorted_cols.remove(FK_col)

        return data, group_data, domain, sorted_cols
        

# attr domain of csv data should be [0, some_int]
# read synthetic data or ground truth
def read_data(data_path, h_domain_path, i_domain_path):
    header, data_list = read_csv(data_path)

    # drop the additional cols of group types of the synthetic data
    df = pd.DataFrame(np.array(data_list, dtype=int), columns=header)
    for col in df.columns:
        if col.find('group_type') != -1:
            df.drop(columns=col, inplace=True)
    header = df.columns

    # group_id/serial, attr1, attr2, ...., attrn
    # print(df.columns)
    # print(df.shape)
    np_data = df.to_numpy()

    h_domain_dict = json.load(open(h_domain_path))
    i_domain_dict = json.load(open(i_domain_path))

    h_attrs = list(h_domain_dict.keys())
    i_attrs = list(i_domain_dict.keys())

    
    i_cols = [i for i in range(len(header)) if header[i] in i_domain_dict]
    individual_data = np_data[:, i_cols]
    i_domain_dict = {attr: {'size': len(i_domain_dict[attr])} for attr in i_domain_dict}
    i_attrs_in_order = [header[i] for i in range(len(header)) if header[i] in i_domain_dict]
    i_domain_dict = {i: i_domain_dict[i_attrs_in_order[i]] for i in range(len(i_attrs_in_order))}

    cols = [0,]
    h_cols = [i for i in range(len(header)) if header[i] in h_domain_dict]
    cols.extend(h_cols)
    household_data = np_data[:, cols]
    household_data, _ = compress_household_data(household_data)
    household_data = household_data[:, 1:]

    h_domain_dict = {attr: {'size': len(h_domain_dict[attr])} for attr in h_domain_dict}
    h_attrs_in_order = [header[i] for i in range(len(header)) if header[i] in h_domain_dict]
    h_domain_dict = {i: h_domain_dict[h_attrs_in_order[i]] for i in range(len(h_attrs_in_order))}

    cols = [0,]
    cols.extend(i_cols)
    individual_data_with_group_id = np_data[:, cols]
    group_data = get_group_data(individual_data_with_group_id)

    i_domain = Domain(i_domain_dict, list(range(len(i_domain_dict))))
    h_domain = Domain(h_domain_dict, list(range(len(h_domain_dict))))
    # print(domain)

    return individual_data, household_data, group_data, i_domain, h_domain, i_attrs, h_attrs

def get_domain(output_attrs, output_attrs_data):
    domain_dict = {}
    for idx in range(len(output_attrs)):
        attr = output_attrs[idx]
        val = output_attrs_data[:, idx]

        domain_dict[attr] = {'size': int(np.max(val)+1)}

        if attr in ['AGE', 'INCTOT', 'PROPINSR', 'COSTELEC', 'VALUEH']:
            domain_dict[attr]['type'] = 'continuous'
        else:
            domain_dict[attr]['type'] = 'discrete'
    return domain_dict
