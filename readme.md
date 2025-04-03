# PrivPetal

Implementation of the paper "PrivPetal: Relational Data Synthesis via Permutation Relations"

## Project Overview

PrivPetal is a method for synthesizing relational data while preserving privacy. It supports multiple types of relational datasets with foreign keys, including:
- Census dataset: An individual table with a foreign key referencing a household table
- TPC-H benchmark: Multiple tables and foreign keys, including one table with two foreign keys
- Instacart dataset: Multiple tables and foreign keys

For other datasets, you will need to write a custom script that:
- Invokes PrivPetal to process each pair of tables with a foreign key
- Generates the complete dataset following the method described in our paper

## Requirements

- Python 3.9
- CUDA 11.7
- GPU
- Required Python packages are listed in `environment.yml`

## Installation


1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1YI1aXsdhxPazHvVc8Ighg1jmx5ItpzrA/view?usp=sharing)
2. Extract the downloaded ZIP file into the project directory
3. Install Anaconda.
4. Use Conda to set up the environment:
```
conda env create -f environment.yml
conda activate privpetal
```

## Core Components

Synthesize one dataset:

- `PrivPetal.py`: Main implementation of PrivPetal
- `PrivPetal_Census.py`: Implementation for Census data synthesis
- `PrivPetal_TPC-H.py`: Implementation for TPC-H data synthesis
- `PrivPetal_instacart.py`: Implementation for Instacart data synthesis

Scripts for running multiple experiments and evaluating the synthesized data:

- `script.py`: Evaluation script for Census data
- `script_TPC-H.py`: Evaluation script for TPC-H data
- `script_instacart.py`: Evaluation script for Instacart data

By default, PrivPetal will use 2 gpus and 4 processes. You may modify this by set `os.environ["CUDA_VISIBLE_DEVICES"]` in PrivPetal.py and set `--process_num` when running the scripts.

## Other Datasets

PrivPetal can be extended to work with your own datasets.

### Data Requirements

- PrivPetal only supports discrete attributes. For continuous attributes, you must discretize them before processing.
- Each attribute should be encoded with values ranging from 0 to size-1 (inclusive).
- Store your data in CSV files and provide a domain definition in JSON format. See the `data/California` directory for examples.

### Implementation Steps

- You'll need to write a script that follows the approach described in our paper.
- Use `PrivPetal_Census.py` and `PrivPetal_instacart.py` as reference implementations.
- Invoke PrivPetal to process each pair of tables with foreign key.

### Performance Considerations:

- Domain Limit
    - Check the domain limits in the PrivPetal log to ensure it's sufficiently large.
    - For example, when synthesizing tables R_I (individual) and R_H (household):
        - If the largest attribute in R_I has domain size x
        - The largest attribute in R_H has domain size y
        - The maximum group size is s
    - Then the domain limit should be larger than x * y * s to properly capture inter-relational correlations
    - It should also be larger than x * x * s to capture intra-group correlations
    - Ideally, most marginals selected by PrivPetal should contain 3 attributes or more.

- If the domain size is insufficient, consider:
    - Factorizing attributes. If an attribute A's domain size is very large (e.g., 10,000), you can decompose it into two attributes A_1 and A_2, where 100 * A_1 + A_2 = A. By replacing A with A_1 and A_2, the maximum attribute domain size in the dataset becomes much smaller, and PrivPetal may synthesize it with better utility.
    - Adding more data. Small datasets may not provide enough information for synthesis.
    - Decreasing theta in the configuration. For example, set `theta = 3` in the config, as shown in `PrivPetal_Census.py`.
    - Merging group sizes. For example, set `size_bins=list(range(6, max_group_size+1))`, as shown in `PrivPetal_Census.py`.
        - Note: Merging group sizes may introduce bias but makes the synthesis more resilient to noise.
        - Once you merge group sizes, the groups become smaller than s, so you can decrease theta further, and the domain limit will be larger.

### Tables with Multiple Foreign Keys:

- For cases where a table contains multiple foreign keys, refer to `PrivPetal_TPC-H.py` for an implementation example.
- The key technique is using the `match_data` function to properly merge foreign keys.

