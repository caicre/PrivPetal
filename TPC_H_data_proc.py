import numpy as np
import csv
import pandas as pd
import json
import datetime
import MRF
import time

std_date = datetime.date(1992, 1, 1)
L_SHIPDATE_proc_c = 14
L_COMMITDATE_proc_c = 14
L_COMMITDATE_proc_b = 91
L_RECEIPTDATE_proc_c = 6

def get_ymd(date):
    val_list = [int(item) for item in date.split('-')]
    return val_list

def get_date_year(date):
    val_list = [int(item) for item in date.split('-')]
    return val_list[0]

def get_data_val(date):
    val_list = [int(item) for item in date.split('-')]
    try:
        date = datetime.date(val_list[0], val_list[1], val_list[2])
    except ValueError:
        # print(val_list)
        date = datetime.date(val_list[0], val_list[1], 28)
    return (date - std_date).days

def get_date(val):
    val =  int(val)
    date = std_date + datetime.timedelta(days=val)
    return date

def get_str_from_date(date):
    return '{:d}-{:0>2d}-{:0>2d}'.format(date.year, date.month, date.day)

def get_domain(output_attrs, output_attrs_data):
    domain_dict = {}
    for idx in range(len(output_attrs)):
        attr = output_attrs[idx]

        if attr.find('KEY') == -1 or attr.find('NATIONKEY') != -1:
            val = output_attrs_data[:, idx]

            domain_dict[attr] = {'size': int(np.max(val)+1)}
    return domain_dict

def proc_order_back(df, gt_data):
    columns = [
        'O_ORDERKEY', 'O_ORDERSTATUS', 'O_ORDERDATE1',\
        'O_ORDERDATE2', 'O_ORDERDATE3', 'O_ORDERPRIORITY', 'O_CUSTKEY'
    ]
    df = df[columns]
    data = df.to_numpy()
    output_attrs_data = []

    gt_columns = ['O_ORDERKEY', 'O_CUSTKEY', 'O_ORDERSTATUS', 'O_TOTALPRICE',\
        'O_ORDERDATE', 'O_ORDERPRIORITY', 'O_CLERK', 'O_SHIPPRIORITY', \
        'O_COMMENT']

    for col in range(len(columns)):
        attr = columns[col]
        col_data = data[:, col]

        if attr == 'O_ORDERKEY':

            output_attrs_data.append(col_data)

        elif attr == 'O_ORDERSTATUS':

            gt_col_data = gt_data[:, gt_columns.index(attr)]
            val = np.unique(gt_col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }
            re_val_map = {val: key for key, val in val_map.items()}

            output_data = np.array([re_val_map[item] for item in col_data])

            output_attrs_data.append(output_data)

        elif attr == 'O_ORDERDATE1':

            col_data2 = data[:, col+1]
            col_data3 = data[:, col+2]

            output_data = np.array(\
                ['{}-{:0>2d}-{:0>2d}'.format(col_data[i] + 1992, col_data2[i]+1, col_data3[i]+1) for i in range(len(col_data))])
            output_data = np.array([get_date(get_data_val(item)) for item in output_data])
            output_data = np.array(\
                ['{}-{:0>2d}-{:0>2d}'.format(item.year, item.month, item.day) for item in output_data])
            output_attrs_data.append(output_data)

        elif attr == 'O_ORDERPRIORITY':

            gt_col_data = gt_data[:, gt_columns.index(attr)]
            val = np.unique(gt_col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }
            re_val_map = {val: key for key, val in val_map.items()}

            output_data = np.array([re_val_map[item] for item in col_data])

            output_attrs_data.append(output_data)
        

        elif attr == 'O_CUSTKEY':

            output_attrs_data.append(col_data)
        
    
            # print('pass', attr)

    output_attrs = [
        'O_ORDERKEY', 'O_ORDERSTATUS', 'O_ORDERDATE',\
        'O_ORDERPRIORITY', 'O_CUSTKEY'
    ]


    output_attrs_data = [output_data.reshape((-1, 1)) for output_data in output_attrs_data]
    output_attrs_data = np.concatenate(output_attrs_data, axis=1)

    df = pd.DataFrame(output_attrs_data, columns=output_attrs)

    return df


def proc_lineitem(reader, order_df, year_data, part_to_ratio, order_to_pri):
    attrs = ['L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY', 'L_LINENUMBER', \
        'L_QUANTITY', 'L_EXTENDEDPRICE', 'L_DISCOUNT', 'L_TAX', \
        'L_RETURNFLAG', 'L_LINESTATUS', 'L_SHIPDATE', 'L_COMMITDATE', \
        'L_RECEIPTDATE', 'L_SHIPINSTRUCT', 'L_SHIPMODE', 'L_COMMENT']
    
    line_list = []
    for line in reader:
        line_list.append(line[:-2])

    assert(len(line_list[0])+1 == len(attrs))

    # print(line_list[0])

    input_data = np.array(line_list)
    print('record number:', len(input_data))

    orginal_order_df = pd.read_csv('./data/input_data/orders.tbl', delimiter='|', header=None)
    orginal_order_df.columns = ['O_ORDERKEY', 'O_CUSTKEY', 'O_ORDERSTATUS', 'O_TOTALPRICE',\
        'O_ORDERDATE', 'O_ORDERPRIORITY', 'O_CLERK', 'O_SHIPPRIORITY', \
        'O_COMMENT', 'xxx']
    print(orginal_order_df)
    orginal_order = orginal_order_df.to_numpy()
    order_date_col = list(orginal_order_df.columns).index('O_ORDERDATE')
    order_to_date = {row[0]: row[order_date_col] for row in orginal_order}

    output_attrs_data = []
    output_attrs = ['L_QUANTITY', 'L_DISCOUNT', \
        'L_SHIPDATE', 'L_COMMITDATE', \
        'L_RECEIPTDATE', 'L_SHIPINSTRUCT', 'L_SHIPMODE',\
        'L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY']

    for attr in output_attrs:
        col = attrs.index(attr)
        print('   ', attr)

        if attr == 'L_QUANTITY':

            col_data = input_data[:, col].astype(int)

            # change l_quantity according to part type and brand

            part_col = attrs.index('L_PARTKEY')
            ratio_data = np.array([part_to_ratio[int(line[part_col])] for line in input_data])
            col_data = ratio_data * col_data

            print(np.min(col_data), np.max(col_data))
            col_data1, col_data2 = np.divmod(col_data, 10)

            output_attrs_data.append(col_data1.astype(int))
            output_attrs_data.append(col_data2.astype(int))
        
        elif attr == 'L_DISCOUNT':

            col_data = input_data[:, col].copy()
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))
            col_data = col_data.astype(float)
            # print(np.min(col_data), np.max(col_data))

            col_data[col_data<0] = 0.0
            col_data[col_data>=1.0] = 0.999
            col_data = (col_data * 100).astype(int)

            # print(input_data[:, col])
            # print(col_data)
            output_attrs_data.append(col_data)

        elif attr == 'L_SHIPDATE':
            col_data = input_data[:, col].copy()

            order_date = [order_to_date[int(order)] for order in input_data[:, 0]]
            order_day_num = [get_data_val(date) for date in order_date]
            ship_date_num = [get_data_val(date) for date in col_data]
            diff = np.array(ship_date_num) - np.array(order_day_num) - 1
            col1, col2 = np.divmod(diff, 11)

            output_attrs_data.append(col1)
            output_attrs_data.append(col2)

        elif attr == 'L_COMMITDATE':
            col_data = input_data[:, col].copy()

            commit_date_num = [get_data_val(date) for date in col_data]
            diff = np.array(commit_date_num) - np.array(order_day_num) - 30
            col1, col2 = np.divmod(diff, 10)

            output_attrs_data.append(col1)
            output_attrs_data.append(col2)
        
        elif attr == 'L_RECEIPTDATE':
            col_data = input_data[:, col].copy()

            receipt_date_num = [get_data_val(date) for date in col_data]
            diff = np.array(receipt_date_num) - np.array(ship_date_num) - 1
            col1, col2 = np.divmod(diff, 10)

            output_attrs_data.append(col1)
            output_attrs_data.append(col2)

        elif attr == 'L_SHIPINSTRUCT':
            col_data = input_data[:, col].copy()

            val = np.unique(col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }
            col_data = np.array([val_map[item] for item in col_data], dtype=int)

            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
        elif attr == 'L_SHIPMODE':
            col_data = input_data[:, col].copy()

            val = np.unique(col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }
            print(val_map)
            col_data = np.array([val_map[item] for item in col_data], dtype=int)

            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
        elif attr == 'L_ORDERKEY':
            col_data = input_data[:, col].copy()
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)

            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
        elif attr == 'L_PARTKEY':
            col_data = input_data[:, col].copy()
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)

            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
        elif attr == 'L_SUPPKEY':
            col_data = input_data[:, col].copy()
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)

            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
        else:
            print(attr)
            assert(0)

    output_attrs = ['LINEITEMKEY', 'L_QUANTITY1', 'L_QUANTITY2',\
        'L_DISCOUNT', 'L_SHIPDATE1', 'L_SHIPDATE2', 'L_COMMITDATE1', 'L_COMMITDATE2',\
        'L_RECEIPTDATE1', 'L_RECEIPTDATE2',
        'L_SHIPINSTRUCT', 'L_SHIPMODE', 'L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY']

    output_attrs_data = [col_data.reshape((-1, 1)) for col_data in output_attrs_data]
    output_attrs_data = np.concatenate(output_attrs_data, axis=1)
    output_attrs_data = np.concatenate([np.arange(len(output_attrs_data)).reshape((-1, 1)), output_attrs_data], axis=1)
    output_attrs_data = output_attrs_data[np.argsort(output_attrs_data[:, -3])]
    

    
    output_group_data = MRF.tools.get_group_data(output_attrs_data, [-3,])
    output_group_data_list = []
    print('output_group_data:')
    print(len(output_group_data))

    order_data = order_df.to_numpy()
    order_to_year = {order_data[i, 0]: year_data[i] for i in range(len(order_data))}
    random_array = np.random.random(size=len(output_group_data))
    order_data = order_data[np.argsort(order_data[:, 0])]
    idx_list = []
    for i in range(len(output_group_data)):
        group = output_group_data[i]
        r = random_array[i]
        order_id = group[0, -3]
        year = order_to_year[order_id]

        if len(group) == year - 1992 + 1:
            output_group_data_list.append(group)
            idx_list.append(i)
        else:
            if r < 0.25:
                output_group_data_list.append(group)
                idx_list.append(i)

    order_data = order_data[idx_list]
    order_df = pd.DataFrame(order_data, columns=order_df.columns)
    order_df.to_csv('./data/TPC-H/orders.csv', index=False)
    
    output_attrs_data = np.concatenate(output_group_data_list, axis=0)

    print('output_group_data:')
    print(len(output_group_data_list))

    # change shipmode by orderpriority
    random_array = np.random.random(size=len(output_attrs_data))
    temp_map = {
        0: 1,
        1: 2,
        2: 3,
        3: 5,
        4: 6
    }
    # print(order_to_pri)
    for i in range(len(output_attrs_data)):

        r = random_array[i]
        order = output_attrs_data[i, -3]

        if order_to_pri[order] == 0 or order_to_pri[order] == 1:
            if r < 0.25:
                output_attrs_data[i, -4] = 0 # AIR
            elif r < 0.5:
                output_attrs_data[i, -4] = 4 # REG AIR
            else:
                output_attrs_data[i, -4] = temp_map[int((r - 0.5) / 0.1)]
        else:
            if r < 0.05:
                output_attrs_data[i, -4] = 0 # AIR
            elif r < 0.10:
                output_attrs_data[i, -4] = 4 # REG AIR
            else:
                output_attrs_data[i, -4] = temp_map[int((r - 0.10) / 0.180001)]

    col = []
    col.extend(output_attrs)
    print('final lineitem data:')
    print(output_attrs_data.shape)
    # print(output_attrs_data[:20])
    print(np.min(output_attrs_data[:, -3]), np.max(output_attrs_data[:, -3]))
    print(len(col), col)
    df = pd.DataFrame(output_attrs_data, columns=col)

    output_attrs = ['LINEITEMKEY', 'L_SHIPMODE',\
        'L_QUANTITY1', 'L_QUANTITY2', 'L_DISCOUNT', 'L_SHIPDATE1', 'L_SHIPDATE2', \
        'L_COMMITDATE1', 'L_COMMITDATE2', 'L_RECEIPTDATE1', 'L_RECEIPTDATE2',
        'L_SHIPINSTRUCT', 'L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY']
    df = df[output_attrs]

    domain_dict = get_domain(output_attrs, df.to_numpy())
    json.dump(domain_dict, open('./data/TPC-H/lineitem_domain.json', 'w'))

    df.to_csv('./data/TPC-H/lineitem.csv', index=False)


def proc_lineitem_back(df, order_df, gt_o_tbl, gt_p_tbl):
    columns = ['L_QUANTITY1', 'L_QUANTITY2',\
        'L_DISCOUNT', 'L_SHIPDATE1', 'L_SHIPDATE2', 'L_COMMITDATE1', 'L_COMMITDATE2',\
        'L_RECEIPTDATE1', 'L_RECEIPTDATE2', 'L_SHIPINSTRUCT',\
        'L_SHIPMODE', 'L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY']
    df = df[columns]
    data = df.to_numpy()
    output_attrs_data = []

    order = order_df.to_numpy()
    order_date_col = list(order_df.columns).index('O_ORDERDATE')
    order_to_date = {int(row[0]): row[order_date_col] for row in order}
    # print(order_to_date)

    p_to_price = {line[0]: line[-1] for line in gt_p_tbl}

    converted_columns = ['L_QUANTITY',\
        'L_DISCOUNT', 'L_SHIPDATE', \
        'L_COMMITDATE', 'L_RECEIPTDATE', 'L_SHIPINSTRUCT', \
        'L_SHIPMODE', 'L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY']

    gt_columns = ['L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY', 'L_LINENUMBER', \
        'L_QUANTITY', 'L_EXTENDEDPRICE', 'L_DISCOUNT', 'L_TAX', \
        'L_RETURNFLAG', 'L_LINESTATUS', 'L_SHIPDATE', 'L_COMMITDATE', \
        'L_RECEIPTDATE', 'L_SHIPINSTRUCT', 'L_SHIPMODE', 'L_COMMENT']

    for col in range(len(columns)):
        attr = columns[col]
        col_data = data[:, col]

        if attr == 'L_QUANTITY1':

            output_data = col_data
            col_data2 = data[:, col+1]
            output_data = col_data * 10 + col_data2

            output_attrs_data.append(output_data)


        elif attr == 'L_DISCOUNT':

            output_data = col_data / 100

            output_attrs_data.append(output_data)

        elif attr == 'L_SHIPDATE1':

            order_date = [order_to_date[int(order)] for order in data[:, -3]]
            order_day_num = np.array([get_data_val(date) for date in order_date])

            col_data2 = data[:, col+1]

            ship_date_num = col_data * 11 + col_data2 + 1 + order_day_num
            
            output_data = np.array([get_date(num) for num in ship_date_num])
            output_attrs_data.append(output_data)

        elif attr == 'L_COMMITDATE1':

            col_data2 = data[:, col+1]

            commit_date_num = col_data * 10 + col_data2 + 30 + order_day_num

            output_data = np.array([get_date(num) for num in commit_date_num])
            output_attrs_data.append(output_data)

        elif attr == 'L_RECEIPTDATE1':

            col_data2 = data[:, col+1]

            receipt_date_num = col_data * 10 + col_data2 + 1 + ship_date_num

            output_data = np.array([get_date(num) for num in receipt_date_num])
            output_attrs_data.append(output_data)


        elif attr == 'L_SHIPINSTRUCT':

            gt_col_data = gt_o_tbl[:, gt_columns.index(attr)]
            val = np.unique(gt_col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }
            re_val_map = {val: key for key, val in val_map.items()}

            output_data = np.array([re_val_map[item] for item in col_data])

            output_attrs_data.append(output_data)

        elif attr == 'L_SHIPMODE':

            gt_col_data = gt_o_tbl[:, gt_columns.index(attr)]
            val = np.unique(gt_col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }
            re_val_map = {val: key for key, val in val_map.items()}

            output_data = np.array([re_val_map[item] for item in col_data])

            output_attrs_data.append(output_data)

        elif attr == 'L_ORDERKEY':

            output_attrs_data.append(col_data)
        
        elif attr == 'L_PARTKEY':

            output_attrs_data.append(col_data)
        
        elif attr == 'L_SUPPKEY':

            output_attrs_data.append(col_data)

        else:
            pass

    output_attrs_data = [output_data.reshape((-1, 1)) for output_data in output_attrs_data]
    output_attrs_data = np.concatenate(output_attrs_data, axis=1)
    lineitem = output_attrs_data[np.argsort(output_attrs_data[:, -3])]

    df = pd.DataFrame(lineitem, columns=converted_columns)

    df['price'] = [p_to_price[part_key] for part_key in df['L_PARTKEY']]
    df['L_EXTENDEDPRICE'] = df['L_QUANTITY'] * df['price']
    # print(df)
    # print(df.shape)
    # print(df.columns)

    final_columns = converted_columns.copy()
    final_columns.insert(1, 'L_EXTENDEDPRICE')
    df = df[final_columns]

    return df

def proc_customer(reader):
    attrs = ['C_CUSTKEY', 'C_NAME', 'C_ADDRESS', 'C_NATIONKEY',\
        'C_PHONE', 'C_ACCTBAL', 'C_MKTSEGMENT', 'C_COMMENT']

    line_list = []
    for line in reader:
        line_list.append(line[:-2])

    assert(len(line_list[0])+1 == len(attrs))

    input_data = np.array(line_list)
    print('record number:', len(input_data))

    output_attrs_data = []
    output_attrs = [
        'C_CUSTKEY', 'C_NATIONKEY',
    ]

    for attr in output_attrs:
        print('   ', attr)
        col = attrs.index(attr)
        col_data = input_data[:, col].copy()

        if attr == 'C_CUSTKEY':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
        elif attr == 'C_NATIONKEY':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)
        else:
            print(attr)
            assert(0)

    output_attrs_data = [col_data.reshape((-1, 1)) for col_data in output_attrs_data]
    output_attrs_data = np.concatenate(output_attrs_data, axis=1)
    df = pd.DataFrame(output_attrs_data, columns=output_attrs)
    df.to_csv('./data/TPC-H/customer.csv', index=False)

    cust_to_nation = {line[0]: line[1] for line in output_attrs_data}

    domain_dict = get_domain(output_attrs, output_attrs_data)
    json.dump(domain_dict, open('./data/TPC-H/customer_domain.json', 'w'))

    return cust_to_nation

def proc_order(reader, cust_to_nation):
    attrs = ['O_ORDERKEY', 'O_CUSTKEY', 'O_ORDERSTATUS', 'O_TOTALPRICE',\
        'O_ORDERDATE', 'O_ORDERPRIORITY', 'O_CLERK', 'O_SHIPPRIORITY', \
        'O_COMMENT']

    line_list = []
    for line in reader:
        line_list.append(line[:-2])

    assert(len(line_list[0])+1 == len(attrs))

    input_data = np.array(line_list)
    print('record number:', len(input_data))

    output_attrs_data = []
    output_attrs = [
        'O_ORDERKEY', 'O_ORDERSTATUS', 'O_ORDERDATE',\
        'O_ORDERPRIORITY', 'O_CUSTKEY'
    ]

    for attr in output_attrs:
        print('   ', attr)
        col = attrs.index(attr)
        col_data = input_data[:, col].copy()

        if attr == 'O_ORDERKEY':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)
        elif attr == 'O_ORDERSTATUS':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            val = np.unique(col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }

            col_data = np.array([val_map[item] for item in col_data], dtype=int)

            output_attrs_data.append(col_data)
        elif attr == 'O_ORDERDATE':

            year_data = np.array([get_date_year(date) for date in col_data])

            col_data = np.array([np.array(get_ymd(date)) for date in col_data])
            print(col_data.shape)
            print(col_data)
            output_attrs_data.append(col_data[:, 0]-1992)
            output_attrs_data.append(col_data[:, 1]-1)
            output_attrs_data.append(col_data[:, 2]-1)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
            # print(np.min(col_data), np.max(col_data))
        elif attr == 'O_ORDERPRIORITY':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            val = np.unique(col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }

            print(val_map)
        
            col_data = np.array([val_map[item] for item in col_data], dtype=int)

            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
            # print(np.min(col_data), np.max(col_data))

        elif attr == 'O_CUSTKEY':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)
        else:
            print(attr)
            assert(0)

    output_attrs = [
        'O_ORDERKEY', 'O_ORDERSTATUS', 'O_ORDERDATE1',\
        'O_ORDERDATE2', 'O_ORDERDATE3', 'O_ORDERPRIORITY', 'O_CUSTKEY'
    ]

    output_attrs_data = [col_data.reshape((-1, 1)) for col_data in output_attrs_data]
    output_attrs_data = np.concatenate(output_attrs_data, axis=1)
    

    domain_dict = get_domain(output_attrs, output_attrs_data)
    json.dump(domain_dict, open('./data/TPC-H/orders_domain.json', 'w'))

    order_to_pri = {line[0]: line[-2] for line in output_attrs_data}

    # change O_CUSTKEY by customer nation and order year
    year_col = output_attrs.index('O_ORDERDATE1')
    cust_col = output_attrs.index('O_CUSTKEY')

    valid_cust = set(output_attrs_data[:, cust_col])
    print('valid_cust:', len(valid_cust))
    cust_to_nation = {c: n for c, n in cust_to_nation.items() if c in valid_cust}
    print(len(cust_to_nation))
    nation_to_cust = {n: [] for n in cust_to_nation.values()} 
    for c, n in cust_to_nation.items():
        nation_to_cust[n].append(c)
    # for n in nation_to_cust:
    #     print(n, len(nation_to_cust[n]))
    
    prob_array = np.random.random(size=len(output_attrs_data))
    random_array = (np.random.random(size=len(output_attrs_data)) * 4).astype(int)
    nation_array = (np.random.random(size=len(output_attrs_data)) * 25).astype(int)
    nation_to_cust_idx = {n: 0 for n in nation_to_cust}
    nation_to_cust_size = {n: len(nation_to_cust[n]) for n in nation_to_cust}
    for i in range(len(output_attrs_data)):
        row = output_attrs_data[i]
        if prob_array[i] < 0.40:
            target_nation = row[year_col] + random_array[i] * 7
            if target_nation >= 25:
                target_nation = row[year_col] + int(np.random.random()*3) * 7
        else:
            target_nation = nation_array[i]

        row[cust_col] = nation_to_cust[target_nation][nation_to_cust_idx[target_nation]]
        nation_to_cust_idx[target_nation] += 1
        nation_to_cust_idx[target_nation] %= nation_to_cust_size[target_nation]


    df = pd.DataFrame(output_attrs_data, columns=output_attrs)

    return df, year_data, order_to_pri

def proc_part(reader):
    attrs = ['P_PARTKEY', 'P_NAME', 'P_MFGR', 'P_BRAND', 'P_TYPE',\
        'P_SIZE', 'P_CONTAINER', 'P_RETAILPRICE', 'P_COMMENT']

    line_list = []
    for line in reader:
        line_list.append(line[:-2])

    assert(len(line_list[0])+1 == len(attrs))

    input_data = np.array(line_list)
    print('record number:', len(input_data))

    output_attrs_data = []
    output_attrs = [
        'P_PARTKEY', 'P_MFGR', 'P_BRAND', 'P_TYPE',\
        'P_SIZE', 'P_CONTAINER', 'P_RETAILPRICE'
    ]

    for attr in output_attrs:
        print('   ', attr)
        col = attrs.index(attr)
        col_data = input_data[:, col].copy()

        if attr == 'P_PARTKEY':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
        elif attr == 'P_MFGR':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            val = np.unique(col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }

            col_data = np.array([val_map[item] for item in col_data], dtype=int)

            output_attrs_data.append(col_data)
        elif attr == 'P_BRAND':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            val = np.unique(col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }
            print(val_map)
            brand_list = sorted(list(set(col_data)))
            temp_ratio = 1/len(brand_list) * 1.8
            brand_to_ratio = {brand_list[i]: 0.2 + (i+1) * temp_ratio for i in range(len(brand_list))}
            print('brand_to_ratio:')
            print(brand_to_ratio)
            brand_to_ratio = {val_map[item]: brand_to_ratio[item] for item in brand_to_ratio}
            print(brand_to_ratio)

            col_data = np.array([val_map[item] for item in col_data], dtype=int)

            output_attrs_data.append(col_data)

        elif attr == 'P_TYPE':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            val = np.unique(col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }

            type_list = sorted(list(set(col_data)))

            type_to_ratio = {}
            for i in range(len(type_list)):
                type_str = type_list[i]
                if i < 25: # ECONOMY [0.4, 0.6)
                    type_to_ratio[type_str] = 0.4 + i/25 * 0.2
                elif i < 50: # LARGE [0.2, 0.4)
                    type_to_ratio[type_str] = 0.2 + (i-25)/25 * 0.2
                elif i < 75: # MEDIUM [0.4, 0.6)
                    type_to_ratio[type_str] = 0.4 + (i-50)/25 * 0.2
                elif i < 100: # PROMO [0.8, 1.0)
                    type_to_ratio[type_str] = 0.8 + (i-75)/25 * 0.2
                elif i < 125: # SMALL [0.8, 1.0)
                    type_to_ratio[type_str] = 0.8 + (i-100)/25 * 0.2
                else: # STANDARD [0.4, 0.6)
                    type_to_ratio[type_str] = 0.4 + (i-125)/25 * 0.2

            print(type_to_ratio)

            type_to_ratio = {val_map[item]: type_to_ratio[item] for item in type_to_ratio}

            print(type_to_ratio)


            col_data = np.array([val_map[item] for item in col_data], dtype=int)

            output_attrs_data.append(col_data)

        elif attr == 'P_SIZE':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)

            output_attrs_data.append(col_data)
        elif attr == 'P_CONTAINER':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            val = np.unique(col_data)
            val_map = {
                val[i]: i for i in range(len(val))
            }

            col_data = np.array([val_map[item] for item in col_data], dtype=int)

            output_attrs_data.append(col_data)
        elif attr == 'P_RETAILPRICE':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(float)
            # print(np.min(col_data), np.max(col_data))

            col_data[col_data < 0] = 0
            col_data[col_data >= 2000] = 1999

            col_data = (col_data/20).astype(int)

            output_attrs_data.append(col_data)

            # print(input_data[:, col][:20])
            # print(col_data[:20])
            # print(np.min(col_data), np.max(col_data))
        else:
            print(attr)
            assert(0)

    output_attrs_data = [col_data.reshape((-1, 1)) for col_data in output_attrs_data]
    output_attrs_data = np.concatenate(output_attrs_data, axis=1)

    brand_col = output_attrs.index('P_BRAND')
    type_col = output_attrs.index('P_TYPE')

    part_to_ratio = {line[0]: brand_to_ratio[line[brand_col]] * type_to_ratio[line[type_col]] \
        for line in output_attrs_data}
    # keys = list(part_to_ratio.keys())
    # print(min(keys), max(keys), len(keys))
    # print(list(keys)[:20])
    # print(part_to_ratio)

    df = pd.DataFrame(output_attrs_data, columns=output_attrs)
    df.to_csv('./data/TPC-H/part.csv', index=False)

    domain_dict = get_domain(output_attrs, output_attrs_data)
    json.dump(domain_dict, open('./data/TPC-H/part_domain.json', 'w'))

    return part_to_ratio

def proc_supp(reader):
    attrs = ['S_SUPPKEY', 'S_NAME', 'S_ADDRESS', 'S_NATIONKEY',\
        'S_PHONE', 'S_ACCTBAL', 'S_COMMENT']

    line_list = []
    for line in reader:
        line_list.append(line[:-2])

    assert(len(line_list[0])+1 == len(attrs))

    input_data = np.array(line_list)
    print('record number:', len(input_data))

    output_attrs_data = []
    output_attrs = [
        'S_SUPPKEY', 'S_NATIONKEY', 'S_ACCTBAL'
    ]

    for attr in output_attrs:
        print('   ', attr)
        col = attrs.index(attr)
        col_data = input_data[:, col].copy()

        if attr == 'S_SUPPKEY':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)

        elif attr == 'S_NATIONKEY':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)
        elif attr == 'S_ACCTBAL':
            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(float)
            # print(np.min(col_data), np.max(col_data))

            col_data[col_data < -1000] = 0
            col_data[col_data >= 10000] = 9999

            col_data = ((col_data+1000)/100).astype(int)

            output_attrs_data.append(col_data)

    output_attrs_data = [col_data.reshape((-1, 1)) for col_data in output_attrs_data]
    output_attrs_data = np.concatenate(output_attrs_data, axis=1)
    df = pd.DataFrame(output_attrs_data, columns=output_attrs)
    df.to_csv('./data/TPC-H/supplier.csv', index=False)

    domain_dict = get_domain(output_attrs, output_attrs_data)
    json.dump(domain_dict, open('./data/TPC-H/supplier_domain.json', 'w'))

def proc_partsupp(reader):
    attrs = ['PS_PARTKEY', 'PS_SUPPKEY', 'PS_AVAILQTY', 'PS_SUPPLYCOST', 'PS_COMMENT']

    line_list = []
    for line in reader:
        line_list.append(line[:-2])

    assert(len(line_list[0])+1 == len(attrs))

    input_data = np.array(line_list)
    print('record number:', len(input_data))

    output_attrs_data = []
    output_attrs = [
        'PS_PARTKEY', 'PS_SUPPKEY', 'PS_AVAILQTY', 'PS_SUPPLYCOST'
    ]

    for attr in output_attrs:
        print('   ', attr)
        col = attrs.index(attr)
        col_data = input_data[:, col].copy()

        if attr == 'PS_PARTKEY':

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)

        elif attr == 'PS_SUPPKEY':

            col_data = col_data.astype(int)
            output_attrs_data.append(col_data)

        elif attr == 'PS_AVAILQTY':

            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(int)

            col_data = (col_data/100).astype(int)

            output_attrs_data.append(col_data)

        elif attr == 'PS_SUPPLYCOST':

            # val, cnts = np.unique(col_data, return_counts=True)
            # print(val)
            # print(cnts)
            # print('unique number:', len(val))

            col_data = col_data.astype(float)
            # print(np.min(col_data), np.max(col_data))

            col_data[col_data < 0] = 0
            col_data[col_data > 1000] = 1000

            col_data = (col_data/10).astype(int)

            output_attrs_data.append(col_data)

    output_attrs_data = [col_data.reshape((-1, 1)) for col_data in output_attrs_data]
    output_attrs_data = np.concatenate(output_attrs_data, axis=1)
    df = pd.DataFrame(output_attrs_data, columns=output_attrs)
    print(df.loc[:5])

    part_df = pd.read_csv('./data/TPC-H/part.csv')
    part_df = part_df.rename(columns={'P_PARTKEY': 'PS_PARTKEY'})
    part_df = part_df.set_index('PS_PARTKEY')

    supp_df = pd.read_csv('./data/TPC-H/supplier.csv')
    supp_df = supp_df.rename(columns={'S_SUPPKEY': 'PS_SUPPKEY'})
    supp_df = supp_df.set_index('PS_SUPPKEY')

    df = df.join(part_df, on='PS_PARTKEY')
    df = df.join(supp_df, on='PS_SUPPKEY')

    print(df.loc[:5])

    df = df[["PS_PARTKEY", "PS_SUPPKEY", "P_BRAND", "P_TYPE"]]

    df.to_csv('./data/TPC-H/partsupp.csv', index=False)
    output_attrs_data = df.to_numpy()
    output_attrs = ["PS_PARTKEY", "PS_SUPPKEY", "P_BRAND", "P_TYPE1", "P_TYPE2"]

    df = pd.read_csv('./data/TPC-H/partsupp.csv')
    data = df.to_numpy()
    col_data1, col_data2 = np.divmod(data[:, 3], 25)
    data = np.concatenate([data[:, :3], col_data1.reshape((-1, 1)), col_data2.reshape((-1, 1))], axis=1)
    df = pd.DataFrame(data, columns=output_attrs)
    df.to_csv('./data/TPC-H/partsupp.csv', index=False)

    domain_dict = get_domain(output_attrs, data)
    json.dump(domain_dict, open('./data/TPC-H/partsupp_domain.json', 'w'))

if __name__ == '__main__':
    np.random.seed(int(time.time()))

    print('./data/input_data/part.tbl')
    reader = csv.reader(open('./data/input_data/part.tbl', 'r'), delimiter='|')
    part_to_ratio = proc_part(reader)

    print('./data/input_data/customer.tbl')
    reader = csv.reader(open('./data/input_data/customer.tbl', 'r'), delimiter='|')
    cust_to_nation = proc_customer(reader)

    print('./data/input_data/orders.tbl')
    reader = csv.reader(open('./data/input_data/orders.tbl', 'r'), delimiter='|')
    order_df, year_data, order_to_pri = proc_order(reader, cust_to_nation)

    print('./data/input_data/lineitem.tbl')
    reader = csv.reader(open('./data/input_data/lineitem.tbl', 'r'), delimiter='|')
    proc_lineitem(reader, order_df, year_data, part_to_ratio, order_to_pri)

    print('./data/input_data/supplier.tbl')
    reader = csv.reader(open('./data/input_data/supplier.tbl', 'r'), delimiter='|')
    proc_supp(reader)

    print('./data/input_data/partsupp.tbl')
    reader = csv.reader(open('./data/input_data/partsupp.tbl', 'r'), delimiter='|')
    proc_partsupp(reader)