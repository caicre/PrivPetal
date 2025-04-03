import os
import datetime

if __name__ == '__main__':
    for dir_path in ['./log/', './result/', './temp/']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    tuple_num = 2
    process_num = 4

    for data_name in ['instacart']:
        for exp_num in [0, ]:
            exp_prefix = f'exp{exp_num}'
            for epsilon in ['3.20', '0.10']:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                exp_name = f"{exp_prefix}_{epsilon}"
                
                # Run main experiment
                cmd = f"python PrivPetal_instacart.py --exp_prefix {exp_prefix} --epsilon {epsilon} --data_name {data_name} --tuple_num {tuple_num} --process_num {process_num} > ./log/PrivPetal_instacart_{exp_prefix}_eps{epsilon}_data{data_name}_{timestamp}.txt"
                print(f"\n{cmd}")
                os.system(cmd)


                
                cmd = f'python evaluate_instacart_query.py --data_name=instacart --exp_name=PrivPetal_{exp_prefix}_{epsilon} --ind_num=1 > ./result/evaluate_instacart_query_PrivPetal_{exp_prefix}_{epsilon}_1ind.log'
                print(cmd)
                os.system(cmd)

                cmd = f'python evaluate_instacart_query.py --data_name=instacart --exp_name=PrivPetal_{exp_prefix}_{epsilon} --ind_num=2 > ./result/evaluate_instacart_query_PrivPetal_{exp_prefix}_{epsilon}_2ind.log'
                print(cmd)
                os.system(cmd)
