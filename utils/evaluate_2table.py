import itertools
from MRF import tools
import json
from . import data_proc
# import data_proc
import pandas as pd
import random
import numpy as np
import time
from multiprocessing import Pool
import sys
import time

class  GroupQuery:
    def __init__(self):
        self.i_contain_req = {}
        self.h_contain_req = {}
        self.size_req = None

    def from_str(self, string):
        idx = string.find('}')
        i_contain_req = json.loads(string[:idx+1])
        self.i_contain_req = {int(key): tuple(i_contain_req[key]) for key in i_contain_req}

        idx = string.find('size:', idx) + 5
        self.size_req = tuple(json.loads(string[idx: -1]))

        assert(0)

    def __str__(self):
        str1 = [str(key) + ', '+ str(domain_req) + ', '+ str(record_num) for key, (domain_req, record_num) in self.i_contain_req.items() ]
        str1 = '; '.join(str1)

        str2 = [str(key) + ', '+ str(domain_req) for key, domain_req in self.h_contain_req.items() ]
        str2 = '; '.join(str2)

        return 'i req: ' + str1 + '; h req: '+ str2 + '; size:'+ json.dumps(self.size_req)

    def i_contain_init(self, domain, domain_ratio, record_num, i_attr=None):

        if i_attr == None:
            attr = random.choice(domain.attr_list)
        else:
            attr = random.choice(i_attr)

        domain_size = domain.dict[attr]['size']
        domain_req = list(range(domain_size))
        random.shuffle(domain_req)

        if 'type' in domain.dict[attr] and domain.dict[attr]['type'] == 'continuous':
            size = int(domain_size * domain_ratio)
            size = max(1, size)
            start = int(np.random.random() * (domain_size - size))
            domain_req = set(list(range(start, start+size)))
        else:
            size = int(domain_size * domain_ratio)
            size = max(1, size)
            domain_req = set(domain_req[:size])

        self.i_contain_req[attr] = (domain_req, record_num)

    # select an additional household attr randomly
    def h_contain_init(self, domain, domain_ratio, h_attr=None):

        if h_attr == None:
            attrs = list(set(domain.attr_list) - self.h_contain_req.keys())
        else:
            attrs = list(set(h_attr) - self.h_contain_req.keys())

        if len(attrs) != 0:
            attr = random.choice(attrs)
        else:
            assert(0)

        domain_size = domain.dict[attr]['size']

        if 'type' in domain.dict[attr] and domain.dict[attr]['type'] == 'continuous':
            size = int(domain_size * domain_ratio)
            size = max(1, size)
            start = int(np.random.random() * (domain_size - size))
            domain_req = set(list(range(start, start+size)))
        else:
            domain_req = list(range(domain_size))
            random.shuffle(domain_req)

            size = max(int(domain_size * domain_ratio), 1)
            domain_req = set(domain_req[:size])

        self.h_contain_req[attr] = domain_req

    def size_init(self, size_ratio, max_size):
        size = int(max_size * size_ratio)
        # at least size 1
        # start = int(max_size * random.random()) + 1
        start = int(max_size * random.random()) - size + 1
        end = start + size
        self.size_req = (start, end)

    def check_sat(self, group, group_size, h):
        if len(group) == 0:
            return False

        if self.size_req != None:
            if group_size < self.size_req[0] or group_size >= self.size_req[1]:
                return False          

        for attr, req in self.h_contain_req.items():            
            if h[attr] not in req:
                return False  

        for attr, req in self.i_contain_req.items():
            domain_req, record_num = req
            req_sat = False

            for record in group:
                if record[attr] in domain_req:
                    record_num -= 1
                if record_num <= 0:
                    req_sat = True
                    break

            if not req_sat:
                return False

        return True

def check_FK(group_data, h_data, FK_col=-1):
    FK_list = []
    for group in group_data:
        if len(group) == 0:
            FK_list.append(-1)
        else:
            FK_list.append(group[0][FK_col])
    FK_list = np.array(FK_list)

    invalid_cnt = np.sum(FK_list == -1)
    print('invalid_cnt:', invalid_cnt)
    mask = FK_list != -1

    # print(FK_list.shape)
    # print(h_data[:, 0].shape)
    return np.equal(FK_list[mask], h_data[:, 0].flatten()[mask]).all()

def cut_group_data(i_group_data, h_data, num):
    # print(num)
    if len(h_data) > num:
        idx = np.arange(len(h_data))
        np.random.shuffle(idx)
        idx = idx[: num]
        idx = np.sort(idx)

        h_data = h_data[idx]

    res_list = []
    length = len(i_group_data)
    idx = 0
    for h_id in h_data[:, 0]:
        while idx < length:
            if i_group_data[idx][0, -1] < h_id:
                idx += 1
            elif i_group_data[idx][0, -1] == h_id:
                res_list.append(i_group_data[idx]) 
                idx += 1
                break
            else:
                res_list.append([])
                break

    i_group_data = np.array(res_list, dtype=object)

    return i_group_data, h_data

def get_group_query_list_cnt(group_data, group_size_data, h_data, query_list):
    query_cnt = []
    for query in query_list:
        query_cnt.append(
            get_group_query_cnt(group_data, group_size_data, h_data, query)
        )
    return query_cnt

def get_group_query_cnt(group_data, group_size_data, h_data, query):
    # cnt = 0
    # start_time = time.time()
    # for i in range(len(group_data)):
    #     group = group_data[i]
    #     h = h_data[i]
    #     size = group_size_data[i]
    #     if query.check_sat(group, size, h):
    #         cnt += 1
    # time1 = time.time() - start_time
    # start_time = time.time()
    cnt2 = query.check_sat_array(group_data, group_size_data, h_data)
    cnt2 = int(np.sum(cnt2))
    # print(query)
    # time2 = time.time() - start_time
    # print('cnt:', cnt, cnt2)
    # print('time:', time1, time2)
    # print()
    return cnt2



def get_group_query_error(group_data1, group_data2, i_domain, h_data1, h_data2, h_domain, \
    query_list, process_num=30, file_name='', base=1):
    # print(domain)

    assert(len(group_data1) == len(h_data1))
    assert(len(group_data2) == len(h_data2))

    query_num = len(query_list)


    group_size_data1 = np.array([len(group) for group in group_data1], dtype=int)
    group_size_data2 = np.array([len(group) for group in group_data2], dtype=int)


    with open('./temp/query_list.txt', 'w') as out_file:
        for query in query_list:
            out_file.write(str(query) + '\n')

    acc_error = 0
    with Pool(processes=process_num) as pool:

        query_block_size = 80

        query_cnt_list1 = []
        query_cnt_list2 = []
        for i in range(0, query_num, query_block_size):
            query_cnt_list1.append(pool.apply_async(
                get_group_query_list_cnt, \
                (group_data1, group_size_data1, h_data1, query_list[i: i+query_block_size])
            ))
            query_cnt_list2.append(pool.apply_async(
                get_group_query_list_cnt, \
                (group_data2, group_size_data2, h_data2, query_list[i: i+query_block_size])
            ))

        query_cnt_list1 = [res.get() for res in query_cnt_list1]
        query_cnt_list2 = [res.get() for res in query_cnt_list2]
        # print('get cnt time cost: {:.4f}'.format(time.time()-start_time))

        query_cnt1 = []
        for item in query_cnt_list1:
            query_cnt1.extend(item)
        query_cnt2 = []
        for item in query_cnt_list2:
            query_cnt2.extend(item)
        
        json.dump(query_cnt1, open('./temp/query_cnt_'+file_name+'_1.json', 'w'))
        json.dump(query_cnt2, open('./temp/query_cnt_'+file_name+'_2.json', 'w'))

        group_num1 = len(group_data1)
        group_num2 = len(group_data2)

        # # debug
        # length = np.array([len(group) for group in group_data1], dtype=int)
        # group_num1 = np.sum(length > 1)
        # length = np.array([len(group) for group in group_data2], dtype=int)
        # group_num2 = np.sum(length > 1)

        # error_base = group_num1 / 200
        # error_base = group_num1 / 1000
        # error_base = group_num1 / 2000
        error_base = group_num1 * base
        # error_base = 300
        error_list = []

        for i in range(len(query_list)):
            query = query_list[i]
            cnt1 = query_cnt1[i]
            # cnt2 = query_cnt2[i] * group_num1 / group_num2 / 1.0198
            cnt2 = query_cnt2[i] * group_num1 / group_num2
            error = abs(cnt1 - cnt2)/max(error_base, cnt1)
            # print(str(query) + ' error: {:.4f}'.format(error))
            # if cnt1 > 20000 and cnt2 < 10:
            #     print(i, cnt1, cnt2, query)
            error_list.append(error)
            acc_error += abs(error)

        json.dump(error_list, open('./temp/error_list.json', 'w'))
        acc_error /= query_num

    # # debug
    # print('group_num1:', group_num1)
    # print('group_num2:', group_num2)
    return acc_error

def remove_PK_FK(i_group_data, h_data):
    res_group_data = []
    for group in i_group_data:
        if len(group) == 0:
            res_group_data.append([])
        else:
            res_group_data.append(group[:, 1:-1])
    res_group_data = np.array(res_group_data, dtype=object)
    h_data = h_data[:, 1:]
    return res_group_data, h_data


   
