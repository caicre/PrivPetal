import os
import time
import datetime
from evaluate_PG import evaluate_exp, get_error, create_database, test_query

if __name__ == '__main__':
    for dir_path in ['./log/', './result/', './temp/']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    # Create ground truth database and generate answers for evaluation
    create_database("gt_db")
    test_query('gt_db', './data/input_data/answers/', './data/input_data/converted_queries/')

    data_name = 'TPC-H'
    method = "PrivPetal"
    tuple_num = 2
    process_num = 4
    
    for exp_num in [0,]:
        exp_prefix = f'exp{exp_num}'
        for epsilon in ['3.20', '0.20']:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{exp_prefix}_{epsilon}"
            
            # Run main experiment
            cmd = f"python {method}_{data_name}.py --exp_prefix {method}_{exp_prefix} --epsilon {epsilon} --data_name {data_name} --tuple_num {tuple_num} --process_num {process_num} > ./log/{method}_{exp_prefix}_eps{epsilon}_data{data_name}_{timestamp}.txt"
            print(f"\n{cmd}")
            os.system(cmd)

            # Run evaluation
            try:
                answer_path = f'./temp/{method}_{exp_prefix}_{epsilon}/'
                evaluate_exp(f"{method}_{exp_prefix}", epsilon, './data/input_data/converted_queries/', answer_path)
                
                result_path = f'./result/evaluate_{method}_{exp_prefix}_{epsilon}.txt'
                get_error('./data/input_data/answers/', answer_path, result_path)
            except Exception as e:
                print(f"Error during evaluation: {e}")
