import itertools
import MRF.tools as tools
import numpy as np
import math

def get_edge_scores(data, domain, budget, sensitivity):
    attr_num = len(domain)
    R_score_num = attr_num * (attr_num - 1) / 2 + 1e-4
    R_sensitivity = 2* sensitivity
    R_score_noise = (R_sensitivity ** 2 / (budget  / R_score_num)) ** 0.5

    print('R_score_noise: {:.2f}'.format(R_score_noise))
    print('R_sensitivity:', R_sensitivity)
    edge_score_list = []
    for i, j in itertools.combinations(list(domain.attr_list), 2):
        edge = (i, j)
        edge_score = tools.get_unnormalized_R_score(data, domain, edge)
        noisy_edge_score = edge_score + np.random.normal(scale=R_score_noise)
        if noisy_edge_score < 0:
            noisy_edge_score = 1e-4
        edge_score_list.append((edge, noisy_edge_score, edge_score))

    return edge_score_list

# data: h_attr, group_size, individual 1 i_attr, individual 2 i_attr, ...
def get_f_data_edge_score(data, domain, h_attr_num, \
    i_attr_num, max_group_size, budget, sensitivity):
    attr_num = len(domain)
    R_score_num = attr_num * (attr_num - 1) / 2 + 1e-4
    R_sensitivity = 2* sensitivity
    R_score_noise = (R_sensitivity ** 2 / (budget  / R_score_num)) ** 0.5

    print('R_score_noise: {:.2f}'.format(R_score_noise))
    print('R_sensitivity:', R_sensitivity)
    edge_score_list = []
    for j in range(h_attr_num+1):
        for i in range(j):
            edge = (i, j)
            edge_score = tools.get_unnormalized_R_score(data, domain, edge)
            n_score = edge_score + np.random.normal(scale=R_score_noise)
            edge_score_list.append((edge, n_score, edge_score))

    for t in range(1, max_group_size+1):
        start_pos = h_attr_num + 1 + i_attr_num * (t-1)
        stop_pos = h_attr_num + 1 + i_attr_num * t
        temp_data = data[~(data[:, start_pos] == 0)]
        length = len(temp_data)
        print('size:', t, start_pos, stop_pos, length)
        if length == 0:
            print(f'WARNING: actual max group size is smaller than given max group size {max_group_size}')

        for j in range(start_pos, stop_pos):
            for i in range(j):
                edge = (i, j)
                if length == 0:
                    edge_score = 0
                else:
                    edge_score = tools.get_unnormalized_R_score(temp_data, domain, edge)
                n_score = edge_score + np.random.normal(scale=R_score_noise)
                edge_score_list.append((edge, n_score, edge_score))

                # print('edge {}, {}, score: {:.4f}'.format(i, j, edge_score))

    return edge_score_list

def get_Gaussian_noise(sensitivity, budget, query_num=1):
    noise = (sensitivity ** 2 * query_num / budget) ** 0.5
    return noise

# permutations: h_attr, group_size, individual 1 i_attr, individual 2 i_attr, ...
def get_permutation_edge_score(budget, permutations, weights, \
    f_domain, h_attr_num, i_attr_num, max_group_size, sensitivity=2):
    query_num = (h_attr_num + i_attr_num) * (h_attr_num + i_attr_num - 1) / 2 + i_attr_num * (i_attr_num + 1) / 2
    R_score_noise = get_Gaussian_noise(sensitivity, budget, query_num)
    print('R_score_noise: {:.4e}'.format(R_score_noise))
    edge_score_dict = {}
    cache_dict = {}

    total = np.sum(weights)
    for j in range(h_attr_num + i_attr_num):
        for i in range(j):
            edge = (i, j)
            edge_score = tools.get_unnormalized_R_score(permutations, f_domain, edge, weights=weights,
                                                        cache_dict=cache_dict)
            n_score = edge_score + np.random.normal(scale=R_score_noise)
            n_score, edge_score = n_score/total, edge_score/total
            edge_score_dict[edge] = (n_score, edge_score)
            
            print('edge {}, n_score: {:.4f}, score: {:.4f}'.format(edge, n_score, edge_score))

    # For intra-group correlation, should calculate R score only from size>1 groups.
    # Scores should be normalized so that they are comparable.
    print('intra-group correlation R-score:')
    pos = permutations[:, h_attr_num-1] <= 1
    large_size_weights = weights.copy()
    large_size_weights[pos] = 0
    large_size_total = np.sum(large_size_weights)

    print('large_size_total:', large_size_total)
    for j in range(i_attr_num):
        for i in range(j+1):
            edge = (h_attr_num+i, h_attr_num+i_attr_num+j)

            edge_score = tools.get_unnormalized_R_score(permutations, f_domain, edge, weights=large_size_weights, cache_dict=cache_dict)


            n_score = edge_score + np.random.normal(scale=R_score_noise)
            n_score, edge_score = n_score/large_size_total, edge_score/large_size_total
            edge_score_dict[edge] = (n_score, edge_score)

            print('edge {}, n_score: {:.4f}, score: {:.4f}'.format(edge, n_score, edge_score))


    def get_attr(attr):
        if attr >= h_attr_num:
            return h_attr_num + (attr - h_attr_num)%i_attr_num
        return attr
    
    def get_individual(attr):
        if attr >= h_attr_num:
            return int((attr - h_attr_num)/i_attr_num)
        return -1

    for j in range(h_attr_num+i_attr_num, h_attr_num+i_attr_num*max_group_size):
        for i in range(j):
            new_edge = (i, j)
            # j must be an individual attr
            if i < h_attr_num:
                edge = (get_attr(i), get_attr(j))
            else:
                if get_individual(i) == get_individual(j):
                    edge = (get_attr(i), get_attr(j))
                    edge = tuple(sorted(edge))
                else:
                    edge = sorted([get_attr(i), get_attr(j)])
                    edge = (edge[0], edge[1]+i_attr_num)
                
            # print(f'{edge} -> {new_edge}')
            edge_score_dict[new_edge] = edge_score_dict[edge]

    edge_score_list = []
    for edge, item in edge_score_dict.items():
        edge_score_list.append((edge, item[0], item[1]))

    return edge_score_list

def get_edge_scores_without_stops(data, domain, h_attr_num, \
    i_attr_num, max_group_size, budget, sensitivity):
    assert(data.shape[1] == h_attr_num + max_group_size * (i_attr_num + 1))

    attr_num = len(domain)
    R_score_num = attr_num * (attr_num - 1) / 2 + 1e-4
    R_sensitivity = 2* sensitivity
    R_score_noise = (R_sensitivity ** 2 / (budget  / R_score_num)) ** 0.5

    print('R_score_noise: {:.2f}, ratio: {:.4f}'.format(R_score_noise, R_score_noise/len(data)))
    print('R_sensitivity:', R_sensitivity)

    def inject_noise(score):
        n_score = score + np.random.normal(scale=R_score_noise)
        if n_score < 0:
            n_score = 1e-4
        return n_score

    edge_score_list = []
    for j in range(h_attr_num):
        for i in range(j):
            edge = (i, j)
            edge_score = tools.get_unnormalized_R_score(data, domain, edge)
            n_score = inject_noise(edge_score)
            edge_score_list.append((edge, n_score, edge_score))

    for t in range(1, max_group_size+1):
        start_pos = h_attr_num + (i_attr_num+1) * (t-1)
        stop_pos = h_attr_num + (i_attr_num+1) * t
        temp_data = data[~(data[:, stop_pos-1] == 0)]
        print(t, start_pos, stop_pos, len(temp_data))

        for j in range(start_pos, stop_pos):
            for i in range(j):
                edge = (i, j)
                edge_score = tools.get_unnormalized_R_score(temp_data, domain, edge)
                n_score = inject_noise(edge_score)
                edge_score_list.append((edge, n_score, edge_score))

    return edge_score_list

def get_noisy_data_num(data_num, sensitivity, data_num_budget):
    noise = (sensitivity ** 2 / data_num_budget) ** 0.5
    noisy_data_num = data_num + np.random.normal(scale=noise)
    return noisy_data_num

def get_dom_limit(data_num, marginal_num, marginal_budget, sensitivity, theta):
    normal_abs_dev_ratio = (2 / math.pi) ** 0.5
    marginal_noise = (sensitivity ** 2 / (marginal_budget  / marginal_num)) ** 0.5
    dom_limit = data_num / (marginal_noise * normal_abs_dev_ratio) / theta
    return dom_limit, marginal_noise

def get_h_noise(budget, sensitivity, config):
    h_num = config['structrue_learning_it'] * config['sample_candidate_num']
    h_noise = (h_num * sensitivity ** 2 / budget) ** 0.5
    return h_noise