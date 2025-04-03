import argparse
import pandas as pd
import json
import os
import numpy as np

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Process input and output directories.')
    parser.add_argument('--input_dir', type=str, help='Input directory containing CSV files')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving results')
    return parser.parse_args()

def get_department_to_aisle_dict(aisle_depart_product_df):

    department_to_aisle_dict = aisle_depart_product_df.groupby('department_id')['aisle_id'].apply(list).to_dict()

    # from department_id and idx to aisle_id
    department_idx_to_aisle_dict = {k: sorted(set(v)) for k, v in department_to_aisle_dict.items()}
    # from aisle_id to idx
    aisle_to_idx_dict = {}
    for dept_id, aisle_ids in department_idx_to_aisle_dict.items():
        for idx, aisle_id in enumerate(aisle_ids):
            aisle_to_idx_dict[aisle_id] = idx

    return department_idx_to_aisle_dict, aisle_to_idx_dict

# department_id - 1
# aisle_id to aisle_id_idx
# order_hour_of_day, days_since_prior_order to part1 and part2

# Main processing function
def process_data(input_dir, output_dir):
    # Load CSV files
    order_products_prior_df = pd.read_csv(os.path.join(input_dir, 'order_products_prior.csv'))
    orders_df = pd.read_csv(os.path.join(input_dir, 'orders.csv'))
    products_df = pd.read_csv(os.path.join(input_dir, 'products.csv'))

    # Map product_id in order_products_prior.csv to aisle_id and department_id
    order_products_prior_df = order_products_prior_df.merge(products_df[['product_id', 'aisle_id', 'department_id']], on='product_id', how='left')

    # assert each aisle_id belong to the same department_id
    grouped = order_products_prior_df.groupby('aisle_id')['department_id'].nunique()
    assert (grouped == 1).all()

    # assert each product_id belong to the same aisle_id
    grouped = order_products_prior_df.groupby('product_id')['aisle_id'].nunique()
    assert (grouped == 1).all()

    ########################################################
    # process aisle_depart_product
    print('processing aisle_depart_product')
    aisle_depart_product_df = order_products_prior_df[['aisle_id', 'department_id', 'product_id']]
    aisle_depart_product_df = aisle_depart_product_df.groupby(['aisle_id', 'department_id', 'product_id']).size().reset_index(name='count')
    aisle_depart_product_gt_df = aisle_depart_product_df[['product_id', 'aisle_id', 'department_id']]

    print(aisle_depart_product_df)
    print(aisle_depart_product_gt_df)
    aisle_depart_product_df.to_csv(os.path.join(output_dir, 'aisle_depart_product.csv'), index=False)
    aisle_depart_product_gt_df.to_csv(os.path.join(output_dir, 'gt', 'aisle_depart_product.csv'), index=False)

    department_idx_to_aisle_dict, aisle_to_idx_dict = get_department_to_aisle_dict(aisle_depart_product_gt_df)
    print(department_idx_to_aisle_dict)
    print(aisle_to_idx_dict)
    print()

    ########################################################
    # process order_products
    print('processing order_products')
    # move order_id to the last column and add od_id
    order_products_prior_df = order_products_prior_df[[col for col in order_products_prior_df.columns if col != 'order_id'] + ['order_id']]
    order_products_prior_df.insert(0, 'od_id', range(len(order_products_prior_df)))
    order_products_prior_df['aisle_id_idx'] = order_products_prior_df['aisle_id'].map(aisle_to_idx_dict)

    order_products_gt_df = order_products_prior_df[['od_id', 'reordered', 'department_id', 'aisle_id', 'product_id', 'order_id']]
    order_products_gt_df.to_csv(os.path.join(output_dir, 'gt', 'order_products.csv'), index=False)
    order_products_prior_df = order_products_prior_df[['od_id', 'reordered', 'department_id', 'aisle_id_idx', 'order_id']]
    order_products_prior_df['department_id'] = order_products_prior_df['department_id'] - 1

    assert (order_products_prior_df >= 0).all().all()
    print(order_products_prior_df)
    print(order_products_gt_df)
    order_products_prior_df.to_csv(os.path.join(output_dir, 'order_products.csv'), index=False)
    order_products_gt_df.to_csv(os.path.join(output_dir, 'gt', 'order_products.csv'), index=False)
    print()

    ########################################################
    # process orders
    print('processing orders')
    orders_df = orders_df[['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'user_id']]

    # factorize order_hour_of_day and days_since_prior_order    
    orders_df['order_hour_of_day_part1'] = orders_df['order_hour_of_day'] // 6
    orders_df['order_hour_of_day_part2'] = orders_df['order_hour_of_day'] % 6

    orders_df['days_since_prior_order'] = orders_df['days_since_prior_order'].fillna(-1).astype(int)
    assert (orders_df['days_since_prior_order'] < 40).all()
    orders_df['days_since_prior_order_part1'] = np.where(
        orders_df['days_since_prior_order'] != -1,
        orders_df['days_since_prior_order'] // 10,
        4
    )
    orders_df['days_since_prior_order_part2'] = np.where(
        orders_df['days_since_prior_order'] != -1,
        orders_df['days_since_prior_order'] % 10,
        0
    )
    orders_gt_df = orders_df[['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'user_id']].copy()
    orders_df = orders_df[['order_id', 'order_dow', 'order_hour_of_day_part1', 'order_hour_of_day_part2', 'days_since_prior_order_part1',  'days_since_prior_order_part2', 'user_id']]
    assert (orders_df >= 0).all().all()
    # replace -1 (NA) with max + 1
    orders_gt_df['days_since_prior_order'].replace(-1, orders_gt_df['days_since_prior_order'].max() + 1, inplace=True)

    print(orders_df)
    print(orders_gt_df)
    orders_df.to_csv(os.path.join(output_dir, 'orders.csv'), index=False)
    orders_gt_df.to_csv(os.path.join(output_dir, 'gt', 'orders.csv'), index=False)
    print()

    ########################################################
    # process users
    print('processing users')
    # Save user_ids to users.csv
    users_df = orders_df[['user_id']].drop_duplicates()
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
    users_df.to_csv(os.path.join(output_dir, 'gt', 'users.csv'), index=False)
    print()

    ########################################################
    # process domains
    print('processing domains')
    # generate a dict mapping each attr to its max_value + 1
    print('order_products')
    domain_dict = {}
    for col in order_products_prior_df.columns:
        if  col != 'order_id' and col != 'od_id' and col != 'product_id':
            domain = (min(order_products_prior_df[col]), max(order_products_prior_df[col]))
            print(col, domain)
            domain_dict[col] = {'size': domain[1] + 1, 'type': 'discrete'}
    json.dump(domain_dict, open(os.path.join(output_dir, 'order_products_domain.json'), 'w'))

    domain_dict = {}
    for col in order_products_gt_df.columns:
        if  col != 'order_id' and col != 'od_id' and col != 'product_id':
            domain = (min(order_products_gt_df[col]), max(order_products_gt_df[col]))
            print(col, domain)
            domain_dict[col] = {'size': domain[1] + 1, 'type': 'discrete'}
    json.dump(domain_dict, open(os.path.join(output_dir, 'gt', 'order_products_domain.json'), 'w'))

    print('\norders')
    domain_dict = {}
    for col in orders_df.columns:
        if  col != 'order_id' and col != 'user_id':
            domain = (min(orders_df[col]), max(orders_df[col]))
            print(col, domain)
            domain_dict[col] = {'size': domain[1] + 1, 'type': 'discrete'}
    json.dump(domain_dict, open(os.path.join(output_dir, 'orders_domain.json'), 'w'))

    domain_dict = {}
    for col in orders_gt_df.columns:
        if  col != 'order_id' and col != 'user_id':
            domain = (min(orders_gt_df[col]), max(orders_gt_df[col]))
            print(col, domain)
            domain_dict[col] = {'size': domain[1] + 1, 'type': 'discrete'}
    json.dump(domain_dict, open(os.path.join(output_dir, 'gt', 'orders_domain.json'), 'w'))

    json.dump({}, open(os.path.join(output_dir, 'users_domain.json'), 'w'))
    json.dump({}, open(os.path.join(output_dir, 'gt', 'users_domain.json'), 'w'))

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Process the data
    process_data(args.input_dir, args.output_dir)
