import itertools
import json
from utils import data_proc
import pandas as pd
import random
import numpy as np
import time
from multiprocessing import Pool
import sys
import time
import pickle

from utils.evaluate_2table import cut_group_data, check_FK, get_group_query_error

class IntraGroupQuery:
    def __init__(self):
        self.i_contain_req = []
        self.h_contain_req = []
        self.size_req = None

    def __str__(self):
        str1 = ''

        temp_str = [str(attr) + ', '+ str(domain_req) for (attr, domain_req) in self.i_contain_req ]
        temp_str = '; '.join(temp_str) + ' '

        str1 += temp_str

        str2 = [str(attr) + ', '+ str(domain_req) for (attr, domain_req) in self.h_contain_req ]
        str2 = '; '.join(str2)

        return 'i_req: ' + str1 + '; h_req: '+ str2 + '; size:'+ json.dumps(self.size_req)

    def i_contain_init(self, domain, domain_ratio, attr_num):

        attr_list = np.random.choice(domain.attr_list, size=attr_num, replace=False)

        for attr in attr_list:
            domain_size = domain.dict[attr]['size']
            
            if 'type' in domain.dict[attr] and domain.dict[attr]['type'] == 'continuous':
                size = int(domain_size * domain_ratio)
                size = max(1, size)
                start = int(np.random.random() * (domain_size - size))
                domain_req = set(list(range(start, start+size)))
            else:
                domain_req = list(range(domain_size))
                random.shuffle(domain_req)

                size = int(domain_size * domain_ratio)
                size = max(1, size)
                domain_req = set(domain_req[:size])

            self.i_contain_req.append((attr, domain_req))

    # select an additional household attr randomly
    def h_contain_init(self, domain, domain_ratio, attr_num):
        attr_num = min(attr_num, len(domain))
        if attr_num <= 0:
            return
        attr_list = np.random.choice(domain.attr_list, size=attr_num, replace=False)

        for attr in attr_list:
            domain_size = domain.dict[attr]['size']

            if 'type' in domain.dict[attr] and domain.dict[attr]['type'] == 'continuous':
                size = int(domain_size * domain_ratio)
                size = max(1, size)
                start = int(np.random.random() * (domain_size - size))
                domain_req = set(list(range(start, start+size)))
            else:
                domain_req = list(range(domain_size))
                random.shuffle(domain_req)

                size = int(domain_size * domain_ratio)
                size = max(1, size)
                domain_req = set(domain_req[:size])

            self.h_contain_req.append((attr, domain_req))

    def size_init(self, size_ratio, max_size):
        size = int(max_size * size_ratio)
        start = int(max_size * random.random()) - size + 2
        end = start + size
        self.size_req = (start, end)

    def check_sat(self, group, group_size, h):

        if len(group) == 0:
            return False

        if self.size_req != None:
            if group_size < self.size_req[0] or group_size >= self.size_req[1]:
                return False          

        for attr, req in self.h_contain_req:            
            if h[attr] not in req:
                return False

        find_record = False
        for i in range(group.shape[0]):
            record = group[i]
            req_sat = True
            for attr, domain_req in self.i_contain_req:                
                if not record[attr] in domain_req:
                    req_sat = False
                    break
            
            if req_sat:
                find_record = True

        if find_record:
            return True

        return False

    def check_sat_array(self, group_data, group_size_data, h_data):
        h_data_bool = np.ones(h_data.shape[0], dtype=bool)
        for attr, req in self.h_contain_req:
            h_data_bool = np.logical_and(h_data_bool, np.isin(h_data[:, attr], list(req)))

        if self.size_req != None:
            group_size_bool = np.logical_and(group_size_data >= self.size_req[0], group_size_data < self.size_req[1])
        else:
            group_size_bool = np.ones(group_size_data.shape[0], dtype=bool)

        try:
            for i, group in enumerate(group_data):
                if len(group.shape) != 2:
                    print('bug:')
                    print(group)
                    print(self)
        except Exception as e:
            print('exception bug:')
            print(i)
            print(group)
            print(e)
            print(self)
            print(group_data)
        i_data = np.concatenate(group_data, axis=0)

        def get_i_data_bool(req):
            i_data_bool = np.ones(i_data.shape[0], dtype=bool)
            for attr, domain_req in req:
                i_data_bool = np.logical_and(i_data_bool, np.isin(i_data[:, attr], list(domain_req)))
            return i_data_bool
        i_data_bool = get_i_data_bool(self.i_contain_req)
        

        idx_array = np.cumsum(group_size_data)
        i_group_bool = np.zeros(h_data.shape[0], dtype=bool)

        start_idx = 0
        for i in range(h_data.shape[0]):
            end_idx = idx_array[i]
            i_bool = i_data_bool[start_idx:end_idx]
            if np.sum(i_bool) > 0:
                i_group_bool[i] = True
            start_idx = end_idx

        # if random.random() < 0.01:
        #     print('h_data_bool', np.sum(h_data_bool))
        #     print('group_size_bool', np.sum(group_size_bool))
        #     print('i_group_bool', np.sum(i_group_bool))

        return np.logical_and(np.logical_and(h_data_bool, group_size_bool), i_group_bool)


def evaluate_2table_1ind(gt_i_group, test_i_group, gt_h_data, test_h_data, h_domain, i_domain, attr_num, size_ratio, max_size, base, process_num=10, query_num=10000, selectivity=0.2):
    print('evaluate_2table_1ind')
    print('attr_num:', attr_num)
    print('size_ratio:', size_ratio)
    print('max_size:', max_size)
    print('base:', base)
    print('selectivity:', selectivity)  
    h_attr_num, i_attr_num = attr_num, attr_num
    domain_ratio = selectivity ** ( 1 / (h_attr_num + i_attr_num ))
    i_domain_ratio, h_domain_ratio = domain_ratio, domain_ratio

    def get_query():
        q  = IntraGroupQuery()
        q.i_contain_init(i_domain, i_domain_ratio, i_attr_num)
        q.h_contain_init(h_domain, h_domain_ratio, h_attr_num)
        if not size_ratio is None:
            q.size_init(size_ratio, max_size)
        return q

    query_list = []
    for _ in range(query_num):
        query_list.append(get_query())


    error = get_group_query_error(\
        gt_i_group, test_i_group, i_domain, gt_h_data, test_h_data, h_domain, query_list, \
        process_num=process_num, file_name=str(i_attr_num)+'and'+str(h_attr_num), base=base)
    
    return error

if __name__ == '__main__':

    i_path          = sys.argv[1]
    i_domain_path   = sys.argv[2]
    h_path          = sys.argv[3]
    h_domain_path   = sys.argv[4]
    test_i_path     = sys.argv[5]
    test_h_path     = sys.argv[6]
    base            = float(sys.argv[7])

    print('evaluating', test_i_path)
    print('evaluating', test_h_path)
    localtime = time.asctime(time.localtime(time.time()))
    start_time = time.time()
    print ("time:", localtime)

    evaluate_num = int(2e5) # for evaluation efficiency
    # evaluate_num = int(1e6)
    # evaluate_num = int(1e5)
    # evaluate_num = 3

    # read ground truth data and test data
    _, gt_i_group, i_domain, i_attrs = data_proc.read_table(\
        i_path, i_domain_path, id_col='INDIVIDUAL', FK_col='HOUSEHOLD')
    gt_h_data, h_domain, h_attrs = data_proc.read_table(\
        h_path, h_domain_path, id_col='HOUSEHOLD')

    gt_i_group, gt_h_data = cut_group_data(gt_i_group, gt_h_data, evaluate_num)
    assert(check_FK(gt_i_group, gt_h_data))
    gt_i_group = [group[:, 1:-1] for group in gt_i_group]
    gt_h_data = gt_h_data[:, 1:]
    print('gt household table:', gt_h_data.shape)

    _, test_i_group, i_domain, i_attrs = data_proc.read_table(\
        test_i_path, i_domain_path, id_col='INDIVIDUAL', FK_col='HOUSEHOLD')
    test_h_data, h_domain, h_attrs = data_proc.read_table(\
        test_h_path, h_domain_path, id_col='HOUSEHOLD')

    test_i_group, test_h_data = cut_group_data(test_i_group, test_h_data, evaluate_num)
    # print(test_i_group.shape)
    # print(test_i_group)
    assert(check_FK(test_i_group, test_h_data))

    res_i_group = []
    for group in test_i_group:
        if len(group) == 0:
            res_i_group.append([])
        else:
            res_i_group.append(group[:, 1:-1])
    test_i_group = np.array(res_i_group, dtype=object)
    # test_i_group = [group[:, 1:-1] for group in test_i_group]
    test_h_data = test_h_data[:, 1:]
    print('test household table:', test_h_data.shape)

    size_ratio=0.2
    max_size=7
    print('max_size: {}, size_ratio: {:.2f}'.format(max_size, size_ratio))

    for attr_num in range(1, 3):
        error = evaluate_2table_1ind(gt_i_group, test_i_group, gt_h_data, test_h_data, h_domain, i_domain, attr_num, size_ratio, max_size, base, process_num=50, query_num=10000, selectivity=0.2)
        print('    error: {:.6f}'.format(error))


    print('time cost: {:.4f}'.format(time.time()-start_time))
