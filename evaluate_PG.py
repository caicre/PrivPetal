import psycopg2
import TPC_H_data_proc
import pandas as pd
import csv
import numpy as np
import os
import csv
from io import StringIO
import re
import time

from sqlalchemy import create_engine

def get_lowercase_list(data):
    return [item.lower() for item in data]

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

def legalize_tbl(df):
    for col in df.columns:
        if 'key' in col:
            df[col] = df[col].astype('int')

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
    
        time.sleep(1)

    sql = '''CREATE database '''+db_name+''';'''
    cursor.execute(sql)

    insert_table_tbl(db_name)

def insert_table_tbl(db_name):
    print('inserting ground truth tables...')
    engine = create_engine('postgresql://postgres:123456@127.0.0.1:5432/' + db_name)

    origin_o_df = pd.read_csv(os.path.join('./data/input_data/TPC-H/orders.tbl'), delimiter='|', header=None)
    origin_o_df = origin_o_df[list(origin_o_df.columns)[:-1]]
    origin_l_df = pd.read_csv(os.path.join('./data/input_data/TPC-H/lineitem.tbl'), delimiter='|', header=None)
    origin_l_df = origin_l_df[list(origin_l_df.columns)[:-1]]
    origin_p_df = pd.read_csv(os.path.join('./data/input_data/TPC-H/part.tbl'), delimiter='|', header=None)
    origin_p_df = origin_p_df[list(origin_p_df.columns)[:-2]]

    order_df = pd.read_csv('./data/TPC-H/orders.csv')
    order_df = TPC_H_data_proc.proc_order_back(order_df, origin_o_df.to_numpy())

    lineitem_df = pd.read_csv('./data/TPC-H/lineitem.csv')
    lineitem_df = TPC_H_data_proc.proc_lineitem_back(
        lineitem_df, order_df, origin_l_df.to_numpy(), origin_p_df.to_numpy())

    order_df.columns = get_lowercase_list(order_df.columns)
    legalize_tbl(order_df)
    order_df.to_sql('orders', engine, method=psql_insert_copy, index=False)

    lineitem_df.columns = get_lowercase_list(lineitem_df.columns)
    legalize_tbl(lineitem_df)
    lineitem_df.to_sql('lineitem', engine, method=psql_insert_copy)

    # customer
    df = pd.read_csv('./data/input_data/TPC-H/customer.tbl', header=None, delimiter='|')
    df = df.iloc[:, :-1]
    df.columns = get_lowercase_list(['C_CUSTKEY', 'C_NAME', 'C_ADDRESS', 'C_NATIONKEY',\
        'C_PHONE', 'C_ACCTBAL', 'C_MKTSEGMENT', 'C_COMMENT'])
    legalize_tbl(df)
    # print(df.dtypes)
    df.to_sql('customer', engine, method=psql_insert_copy)

    # part
    df = pd.read_csv('./data/input_data/TPC-H/part.tbl', header=None, delimiter='|')
    df = df.iloc[:, :-1]
    df.columns = get_lowercase_list(['P_PARTKEY', 'P_NAME', 'P_MFGR', 'P_BRAND', 'P_TYPE',\
        'P_SIZE', 'P_CONTAINER', 'P_RETAILPRICE', 'P_COMMENT'])
    legalize_tbl(df)
    # print(df.dtypes)
    df.to_sql('part', engine, method=psql_insert_copy)

    # # part
    # df = pd.read_csv('./data/TPC-H/part.csv')
    # df.columns = get_lowercase_list(df.columns)
    # legalize_tbl(df)
    # df.to_sql('part', engine, method=psql_insert_copy)

    # supplier
    df = pd.read_csv('./data/input_data/TPC-H/supplier.tbl', header=None, delimiter='|')
    df = df.iloc[:, :-1]
    df.columns = get_lowercase_list(['S_SUPPKEY', 'S_NAME', 'S_ADDRESS', 'S_NATIONKEY',\
        'S_PHONE', 'S_ACCTBAL', 'S_COMMENT'])
    legalize_tbl(df)
    # print(df.dtypes)
    df.to_sql('supplier', engine, method=psql_insert_copy)

    # nation
    df = pd.read_csv('./data/input_data/TPC-H/nation.tbl', header=None, delimiter='|')
    df = df.iloc[:, :-1]
    df.columns = get_lowercase_list(['N_NATIONKEY', 'N_NAME', 'N_REGIONKEY', 'N_COMMENT'])
    legalize_tbl(df)
    # print(df.dtypes)
    df.to_sql('nation', engine, method=psql_insert_copy)

    # region
    df = pd.read_csv('./data/input_data/TPC-H/region.tbl', header=None, delimiter='|')
    df = df.iloc[:, :-1]
    df.columns = get_lowercase_list(['R_REGIONKEY', 'R_NAME', 'R_COMMENT'])
    legalize_tbl(df)
    # print(df.dtypes)
    df.to_sql('region', engine, method=psql_insert_copy)

    # partsupp
    df = pd.read_csv('./data/input_data/TPC-H/partsupp.tbl', header=None, delimiter='|')
    df = df.iloc[:, :-1]
    df.columns = get_lowercase_list(['PS_PARTKEY', 'PS_SUPPKEY', 'PS_AVAILQTY', 'PS_SUPPLYCOST', 'PS_COMMENT'])
    legalize_tbl(df)
    # print(df.dtypes)
    df.to_sql('partsupp', engine, method=psql_insert_copy)


def update_table(df_path, tab_name, db_name):
    print('update table', tab_name, df_path)
    engine = create_engine('postgresql://postgres:123456@127.0.0.1:5432/' + db_name)
    conn = engine.raw_connection()
    cur = conn.cursor()

    cur.execute("DROP TABLE {};".format(tab_name))
    conn.commit()

    df = pd.read_csv(df_path)
    df.columns = get_lowercase_list(df.columns)
    legalize_tbl(df)
    # print(df.dtypes)
    df.to_sql(tab_name, engine, index=False)

def check_db(db_name):
    print('checking', db_name)
    engine = create_engine('postgresql://postgres:123456@127.0.0.1:5432/' + db_name)

    for tab in ['lineitem', 'orders', 'customer', 'part', 'supplier', \
        'partsupp', 'nation', 'region']:
        df = pd.read_sql('SELECT count(*) FROM '+tab, engine)
        print(tab)
        print(df)

        

def test_query(db_name, output_path, query_path, q_list=None):
    print('testing', db_name)
    engine = create_engine('postgresql://postgres:123456@127.0.0.1:5432/' + db_name)

    if not os.path.exists(output_path):
        os.system('mkdir '+ output_path)

    if q_list is None:
        q_list = [4, 5, 7, 9, 12, 14, 17, 19]
    
    # for q in [4, 5, 7, 9, 12, 14]:
    for q in q_list:
    # for q in [4, 5, 7, 9, 12, 14, '17.6', 19]:
    # for q in [4, '4.1', '4.2']:
    # for q in ['19.2']:
    # for q in ['1c']:
        with open(query_path+str(q)+'.sql') as q_file:
            query = ''.join(q_file.readlines())
            print(q)
            # print(query)

            df = pd.read_sql(query, engine)
            df.to_csv(output_path+str(q)+'.csv', index=False)

def get_error(res1_path, res2_path, output_path, q_list=None):
    output_file = open(output_path, 'w')

    if q_list is None:
        q_list = [4, 5, 7, 9, 12, 14, 17, 19]

    for q in q_list:
        df1 = pd.read_csv(res1_path+str(q)+'.csv')
        df2 = pd.read_csv(res2_path+str(q)+'.csv')

        if q == 12:
            res1 = df1.to_numpy()[:, 1:].flatten()
            res1 = {i: res1[i] for i in range(len(res1))}
            res2 = df2.to_numpy()[:, 1:].flatten()
            res2 = {i: res2[i] for i in range(len(res2))}
        else:
            res1 = {
                    ','.join(list(row[:-1])): float(row[-1])
                    for row in df1.to_numpy().astype(str)
                }
            res2 = {
                    ','.join(list(row[:-1])): float(row[-1])
                    for row in df2.to_numpy().astype(str)
                }
        error = 0
        for item in res1:
            if item in res2:
                error += abs(res1[item] - res2[item]) / (res1[item] + 1e-6)
            else:
                error += 1

        error =  error / len(res1)

        output = 'q'+str(q)+ ' {:.4f}\n'.format(error)
        print(output, end='')
        output_file.write(output)

def evaluate_exp(method_exp_name, eps, query_path, answer_output_path, q_list=None):
    db_name = method_exp_name.lower()
    create_database(db_name)

    update_table('./temp/'+method_exp_name+'_'+eps+'_syn_l.csv', 'lineitem', db_name)
    update_table('./temp/'+method_exp_name+'_'+eps+'_syn_o.csv', 'orders', db_name)

    test_query(db_name, answer_output_path, query_path, q_list=q_list)

if __name__ == '__main__':

    # create_database("gt_db")
    # check_db("gt_db")
    # test_query('gt_db', './data/input_data/answers/', './data/input_data/converted_queries/')
    test_query('gt_db', './data/input_data/answers/', './data/input_data/converted_queries/', q_list=[19,])
