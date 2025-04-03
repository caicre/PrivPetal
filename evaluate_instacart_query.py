from utils.evaluate_2table import remove_PK_FK, cut_group_data, check_FK, get_group_query_error
from evaluate_2table_2individual_relative import evaluate_2table_2ind
from evaluate_2table_1individual_relative import evaluate_2table_1ind
import argparse
import time
import json
import os
import pandas as pd
import MRF
import PrivPetal as PP

class InstacartTestData:
    def load_test_data(self, path, exp_name, gt_path, frac=1):
        od_df = pd.read_csv(os.path.join(path, exp_name+'_od.csv'))
        o_df = pd.read_csv(os.path.join(path, exp_name+'_o.csv'))
        u_df = pd.read_csv(os.path.join(path, exp_name+'_u.csv'))
        p_df = pd.read_csv(os.path.join(gt_path, 'gt', 'aisle_depart_product.csv'))

        # od_df = od_df[['od_id', 'reordered', 'product_id', 'order_id']]
        od_df = od_df[['od_id', 'reordered', 'department_id', 'aisle_id', 'product_id', 'order_id']]
        o_df = o_df[['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'user_id']]
        u_df = u_df[['user_id']]

        # randomly sample percentile of users
        if frac < 1:
            u_df = u_df.sample(frac=frac)

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
        print(p_df.columns, p_df.shape)
        print(p_df.head())

        od_domain = json.load(open(os.path.join(gt_path, 'gt', 'order_products_domain.json')))
        o_domain = json.load(open(os.path.join(gt_path, 'gt', 'orders_domain.json')))
        u_domain = json.load(open(os.path.join(gt_path, 'gt', 'users_domain.json')))


        od_domain = MRF.tools.get_domain_by_attrs(od_domain, od_df.columns[1:-2])
        o_domain = MRF.tools.get_domain_by_attrs(o_domain, o_df.columns[1:-1])
        u_domain = MRF.tools.get_domain_by_attrs(u_domain, u_df.columns[1:])
        p_domain = MRF.tools.get_adaptive_domain(p_df.to_numpy()[:, 1:])

        print(od_domain)
        print(o_domain)
        print(u_domain)
        print(p_domain)

        self.od_df, self.o_df, self.u_df, self.p_df = od_df, o_df, u_df, p_df
        self.od_domain, self.o_domain, self.u_domain, self.p_domain = od_domain, o_domain, u_domain, p_domain

    def load_gt_data(self, path, frac=1):
        od_df = pd.read_csv(os.path.join(path, 'gt', 'order_products.csv'))
        o_df = pd.read_csv(os.path.join(path, 'gt', 'orders.csv'))
        u_df = pd.read_csv(os.path.join(path, 'gt', 'users.csv'))
        p_df = pd.read_csv(os.path.join(path, 'gt', 'aisle_depart_product.csv'))

        print(od_df.columns, od_df.shape)
        print(od_df.head())
        print(o_df.columns, o_df.shape)
        print(o_df.head())
        print(u_df.columns, u_df.shape)
        print(u_df.head())
        print(p_df.columns, p_df.shape)
        print(p_df.head())

        od_domain = json.load(open(os.path.join(path, 'gt', 'order_products_domain.json')))
        o_domain = json.load(open(os.path.join(path, 'gt', 'orders_domain.json')))
        u_domain = json.load(open(os.path.join(path, 'gt', 'users_domain.json')))

        od_domain = MRF.tools.get_domain_by_attrs(od_domain, od_df.columns[1:-2])
        o_domain = MRF.tools.get_domain_by_attrs(o_domain, o_df.columns[1:-1])
        u_domain = MRF.tools.get_domain_by_attrs(u_domain, u_df.columns[1:])
        p_domain = MRF.tools.get_adaptive_domain(p_df.to_numpy()[:, 1:])

        print(od_domain)
        print(o_domain)
        print(u_domain)
        print(p_domain)

        self.od_df, self.o_df, self.u_df, self.p_df = od_df, o_df, u_df, p_df
        self.od_domain, self.o_domain, self.u_domain, self.p_domain = od_domain, o_domain, u_domain, p_domain

    def get_o_u_data(self):
        o_df = self.o_df
        u_df = self.u_df

        used_u_set = set(o_df.iloc[:, -1])
        idx = u_df.iloc[:, 0].apply(lambda x: x in used_u_set)
        used_u_df = u_df[idx]
        not_used_u_df = u_df[~idx]
        print('used_u_df:', used_u_df.shape)
        print('not_used_u_df:', not_used_u_df.shape)

        u_id = used_u_df.to_numpy()[:, 0]
        print(max(u_id), min(u_id))

        u_id = o_df.to_numpy()[:, -1]
        print(max(u_id), min(u_id))
        return PP.Data(used_u_df, self.u_domain, o_df, self.o_domain), not_used_u_df
    
    def get_od_o_data(self):
        od_df = self.od_df[['od_id', 'reordered', 'department_id', 'aisle_id', 'order_id']]
        o_df = self.o_df[['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]

        used_o_set = set(od_df.iloc[:, -1])
        idx = o_df.iloc[:, 0].apply(lambda x: x in used_o_set)
        used_o_df = o_df[idx]
        not_used_o_df = o_df[~idx]
        print('used_o_df:', used_o_df.shape)
        print('not_used_o_df:', not_used_o_df.shape)

        return PP.Data(used_o_df, self.o_domain, od_df, self.od_domain), not_used_o_df
    
    def get_od_p_data(self):
        od_df = self.od_df[['od_id', 'reordered', 'product_id']]
        p_df = self.p_df

        used_p_set = set(od_df.iloc[:, -1])
        idx = p_df.iloc[:, 0].apply(lambda x: x in used_p_set)
        used_p_df = p_df[idx]
        not_used_p_df = p_df[~idx]
        print('used_p_df:', used_p_df.shape)
        print('not_used_p_df:', not_used_p_df.shape)

        return PP.Data(used_p_df, self.p_domain, od_df, self.od_domain), not_used_p_df


def evaluate_data(gt_data, test_data, ind_num, attr_num, max_group_size, base, size_ratio, selectivity):
    gt_data.get_group_data([-1,], max_group_size+1)
    test_data.get_group_data([-1,], max_group_size+1)

    gt_i_group_data, gt_h_data = cut_group_data(gt_data.i_group_data, gt_data.h_data, int(2e5))
    assert(check_FK(gt_i_group_data, gt_h_data))
    gt_i_group_data, gt_h_data = remove_PK_FK(gt_i_group_data, gt_h_data)
    test_i_group_data, test_h_data = cut_group_data(test_data.i_group_data, test_data.h_data, int(2e5))
    assert(check_FK(test_i_group_data, test_h_data))
    test_i_group_data, test_h_data = remove_PK_FK(test_i_group_data, test_h_data)

    if ind_num == 2:
        error = evaluate_2table_2ind(gt_i_group_data, test_i_group_data, gt_h_data, test_h_data, gt_data.h_domain, gt_data.i_domain, attr_num, size_ratio, max_group_size, base, query_num=10000, process_num=25, selectivity=selectivity)
    elif ind_num == 1:
        error = evaluate_2table_1ind(gt_i_group_data, test_i_group_data, gt_h_data, test_h_data, gt_data.h_domain, gt_data.i_domain, attr_num, size_ratio, max_group_size, base, query_num=10000, process_num=25, selectivity=selectivity)
    else:
        raise ValueError('ind_num must be 1 or 2')
    print('error:', error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--data_name', type=str, default='instacart')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--base', type=float, default=0.01)
    parser.add_argument('--size_ratio', type=float, default=0.2)
    parser.add_argument('--selectivity', type=float, default=0.01)
    parser.add_argument('--ind_num', type=int, default=2)
    args = parser.parse_args()
    print('time:', time.asctime(time.localtime(time.time())))
    print(args)

    data_name = args.data_name

    o_u_max_group_size = 60
    od_o_max_group_size = 30
    od_p_max_group_size = 3000

    gt_data = InstacartTestData()
    gt_data.load_gt_data(f'data/{data_name}')

    test_data = InstacartTestData()
    test_data.load_test_data('temp/', args.exp_name, f'data/{data_name}')
    
    o_u_gt_data, gt_not_used_u_df = gt_data.get_o_u_data()
    o_u_test_data, test_not_used_u_df = test_data.get_o_u_data()

    od_o_gt_data, gt_not_used_o_df = gt_data.get_od_o_data()
    od_o_test_data, test_not_used_o_df = test_data.get_od_o_data()

    print('evaluate individual num:', args.ind_num)
    for attr_num in [1, 2]:
        print(f'evaluate order_product - order FK, attr_num: {attr_num}')
        evaluate_data(od_o_gt_data, od_o_test_data, args.ind_num, attr_num, o_u_max_group_size, args.base, args.size_ratio, args.selectivity)
        print('-' * 20 + '\n')

        print(f'evaluate order - user FK, attr_num: {attr_num}')
        evaluate_data(o_u_gt_data, o_u_test_data, args.ind_num, attr_num, o_u_max_group_size, args.base, args.size_ratio, args.selectivity)
        print('-' * 20 + '\n')
