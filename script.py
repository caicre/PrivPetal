import os
import datetime

if __name__ == '__main__':
    for dir_path in ['./log/', './result/', './temp/']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    max_group_size      = 8
    tuple_num           = 3
    fact_threshold      = 20
    fact_size           = 10
    method              = "PrivPetal"

    for base in ['0.01']:
        for data_name in ['California']:
            for exp_num in [0, ]:
                exp_prefix = f'exp{exp_num}'
                for epsilon in ['3.20', '0.10']:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    exp_name = f"{exp_prefix}_{epsilon}"
                    
                    # Run main experiment
                    cmd = f"python PrivPetal_Census.py --exp_prefix {exp_prefix} --epsilon {epsilon} --data_name {data_name} --max_group_size {max_group_size} --tuple_num {tuple_num} --fact_threshold {fact_threshold} --fact_size {fact_size} > ./log/PrivPetal_Census_{exp_prefix}_eps{epsilon}_data{data_name}_{timestamp}.txt"
                    print(f"\n{cmd}")
                    os.system(cmd)

                    # Run evaluation
                    eval_cmd = f"python evaluate_2table_1individual_relative.py ./data/{data_name}/individual.csv ./data/{data_name}/individual_domain.json ./data/{data_name}/household.csv ./data/{data_name}/household_domain.json ./temp/{method}_{exp_name}_{data_name}_individual.csv ./temp/{method}_{exp_name}_{data_name}_household.csv {base} > ./result/{method}_{exp_name}_{data_name}_1ind_cnt{base}.txt"
                    print(f"\n{eval_cmd}")
                    os.system(eval_cmd)

                    eval_cmd = f"python evaluate_2table_2individual_relative.py ./data/{data_name}/individual.csv ./data/{data_name}/individual_domain.json ./data/{data_name}/household.csv ./data/{data_name}/household_domain.json ./temp/{method}_{exp_name}_{data_name}_individual.csv ./temp/{method}_{exp_name}_{data_name}_household.csv {base} > ./result/{method}_{exp_name}_{data_name}_2ind_cnt{base}.txt"
                    print(f"\n{eval_cmd}")
                    os.system(eval_cmd)



