from .cp_factor import Factor, Potential
import numpy as np
import cupy as cp
import networkx as nx
import math
import itertools
import time
import pickle
import random
import pandas as pd
from .domain import Domain
import json
from . import tools
import copy
from joblib import Parallel, delayed
import traceback

def get_select_marginal_schedule(marginal_num, iter_num):
    if marginal_num < iter_num:
        return [1] * marginal_num
    marginal_num_list = [int(marginal_num/iter_num)] * iter_num
    for i in range(marginal_num-sum(marginal_num_list)):
        marginal_num_list[i] += 1
    return marginal_num_list

class MarkovRandomField:
    def __init__(self, total: float, domain: Domain, graph, junction_tree, config: dict):
        '''
        total:
            the total to which the model marginals will be normalied  
        config:
            print_interval:
                interval for printing loss when estimating parameters
            structure_learning_it:
                iteration number of structure learning
            estimate_iter_num:
                max iteration number of parameter estimation
            convergence_ratio:
                threshold which loss/theoretic_loss will be smaller than
                in early stop when esimating parameters
            max_attr_num:
                max number of attrs in a marginal
            norm_query_number:
                max number of norms queried in a structure learning iteration
            theta:
                min signal-to-noise ratio of candidate marginals
        '''
        print('Markov random field...')

        # default_configs
        self.config = {
            'print_interval':           100,
            'structure_learning_it':    10,
            'estimate_iter_num':        2000,
            'convergence_ratio':        0.3,
            'max_attr_num':             6,
            'marginal_query_number':    8,
            'norm_query_number':        400,
            'theta':                    6,

            'weight_clip':              None,
        }

        self.total = total
        print('model_total: {:.4e}'.format(self.total))
        self.domain = domain.copy()
        self.graph = graph.copy()
        self.junction_tree = junction_tree.copy()

        self.marginal_noise = None

        for item in config:
            self.config[item] = copy.deepcopy(config[item])

        self.maximal_cliques = list(self.junction_tree.nodes)
        for clique in self.maximal_cliques:
            print(f'clique: {clique}')
            assert(clique == tuple(sorted(clique)))

        self.potential = Potential({
            clique: Factor.zeros(self.domain.project(clique), cp) \
            for clique in self.maximal_cliques})
        size = sum(self.domain.project(clique).size() for clique in self.maximal_cliques)
        print('model size: {:.4e}'.format(size))

        message_list = [(a, b) for a, b in self.junction_tree.edges()]\
            + [(b, a) for a, b in self.junction_tree.edges()] 
        message_edge = []
        for message1 in message_list:
            for message2 in message_list:
                if message1[0] == message2[1] and message2[0] != message1[1]:
                    message_edge.append((message2, message1))
        G = nx.DiGraph()
        G.add_nodes_from(message_list)
        G.add_edges_from(message_edge)
        self.message_order = list(nx.topological_sort(G))

        self.clique_to_marginal = {clique: [] for clique in self.maximal_cliques}
        self.marginal_to_clique = {}

        # cache of data marginals
        self.data_marginal = {}

    def __get_data_marginal(self, marginal, data, domain, weights=None):
        if marginal not in self.data_marginal:
            hist = tools.get_histogram(marginal, data, domain, weights)
            self.data_marginal[marginal] = hist

        return self.data_marginal[marginal]

    def __assign_marginal_list(self, marginal_list):
        for marginal in marginal_list:
            if marginal in self.marginal_to_clique:
                continue
            find_flag=False
            for clique in self.maximal_cliques:
                if set(marginal) <= set(clique):
                    self.clique_to_marginal[clique].append(marginal)
                    self.marginal_to_clique[marginal] = clique
                    find_flag=True
                    break
            if not find_flag:
                print(f'WARNING: marginal {marginal} can not be assigned to cliques')

    def belief_propagation(self, potential):
        belief = Potential({clique: potential[clique].copy() for clique in self.maximal_cliques})

        sent_message = dict()
        for clique1, clique2 in self.message_order:
            separator = set(clique1) & set(clique2)
            if (clique2, clique1) in sent_message:
                message = belief[clique1] - sent_message[(clique2, clique1)]
            else:
                message = belief[clique1]
            message = message.logsumexp(separator)
            belief[clique2] += message

            sent_message[(clique1, clique2)] = message

        partition_func = float(belief[self.maximal_cliques[0]].logsumexp())
        for clique in self.maximal_cliques:
            belief[clique] += np.log(self.total) - partition_func
            belief[clique] = belief[clique].exp()

        return belief, partition_func

    def cal_marginal_dict(self, potential, marginal_set, to_cpu=False):
        self.__assign_marginal_list(marginal_set)

        maximal_clique_marginal, partition_func = self.belief_propagation(potential)
        if to_cpu:
            for clique in maximal_clique_marginal:
                maximal_clique_marginal[clique] = maximal_clique_marginal[clique].to_cpu()

        marginal_dict = {}
        for marginal in marginal_set:
            clique_factor = maximal_clique_marginal[self.marginal_to_clique[marginal]]
            marginal_fact = clique_factor.project(marginal)
            marginal_dict[marginal] = marginal_fact
        return Potential(marginal_dict), partition_func
    
    def estimate_parameters(self, marginal_dict: dict, iter_num: int, terminate_loss=None, conv_ratio=0.99):
        print('estimating parameters...')
        if not self.marginal_noise is None:
            theoretic_loss = tools.get_theoretic_loss(self.domain, marginal_dict.keys(), self.marginal_noise)
            print('theoretic_loss: {:.4e}'.format(theoretic_loss))

        weight_clip = self.config['weight_clip']
        print('weight_clip: ', weight_clip)
        if terminate_loss:
            print('terminate_loss: {:.4e}'.format(terminate_loss))
        potential = self.potential
        marginal_list = list(marginal_dict.keys())
        
        mu = None
        alpha = 1.0 /self.total ** 2
        stepsize = lambda t: 2.0*alpha

        mu, partition_func = self.cal_marginal_dict(potential, marginal_list)
        ans = Potential.l2_marginal_loss(mu, marginal_dict)

        last_print_loss = None
        for it in range(iter_num):
            start_time = time.time()
            omega, nu = potential, mu
            curr_loss, gradient = ans
            alpha = stepsize(it)
            expanded_gradient = self.__get_expanded_gradient(gradient)
            for i in range(25):
                potential = omega - alpha * expanded_gradient
                if not weight_clip is None:
                    potential.clip(weight_clip) # clip to avoid float overflow

                mu, partition_func = self.cal_marginal_dict(potential, marginal_list)
                ans = Potential.l2_marginal_loss(mu, marginal_dict)
                if curr_loss - ans[0] >= 0.5*alpha*gradient.dot(nu-mu):
                    break
                alpha *= 0.5
            
            if it % self.config['print_interval'] == 0 or it == iter_num-1:
                print('    it: {}/{} loss: {:.4e} lr: {:.4e} time: {:.2f}, partition_func: {:.4e}'.format(\
                    it, iter_num, curr_loss, alpha, time.time()-start_time, partition_func))
                if terminate_loss and curr_loss < terminate_loss:
                    break
                if it > 0.19 * iter_num and not last_print_loss is None:
                    if curr_loss/last_print_loss > conv_ratio:
                        break
                last_print_loss = curr_loss
        
        model_total = float(cp.sum(mu[marginal_list[0]].values))

        for marginal in marginal_list:
            data_total = float(cp.sum(marginal_dict[marginal].values))
            rel_error = float(cp.abs(data_total - model_total)/data_total)
            if rel_error > 0.02:
                print('WARNING: large relative error: {:.4e} between {} data_total: {} and model_total: {}'.format(
                    rel_error, marginal, data_total, model_total))
            tvd = float(cp.sum(cp.abs(marginal_dict[marginal].values - mu[marginal].values))/data_total/2)
            print('marginal: {}, total: {:.4f}, {:.4f}, learn tvd: {:.4f}'.format(
                marginal, data_total, model_total, tvd))
        print('')

        self.potential = potential
        self.mu = mu
        self.mu.to_cpu()

    def __get_expanded_gradient(self, gradient):
        expanded_gradient = Potential({clique: Factor.zeros(\
            self.domain.project(clique), cp) for clique in self.maximal_cliques})
        for marginal in gradient:
            clique = self.marginal_to_clique[marginal]
            expanded_gradient[clique] += gradient[marginal]
        return expanded_gradient

    def unordered_structure_learning(self, data, size_attr, weights, norm_budget=None, marginal_budget=None, \
        sensitivity=1, init_marginal_dict=Potential({})):
        '''
        data:
            np.array, tabular data without keys
        norm_budget:
            privacy budget for querying norms to select marginals. None for non-private
        marginal_budget:
            privacy budget for querying selected marginals. None for non-private
        sensitivity:
            sensitivity of a norm or a marginal
        init_marginal_dict:
            Potential dict storing init marginals and their hists
        '''
        structure_learning_it = self.config['structure_learning_it']
        estimate_iter_num = self.config['estimate_iter_num']
        max_attr_num = self.config['max_attr_num']
        marginal_dict = init_marginal_dict.copy()
        marginal_dict.to_gpu()
        marginal_query_num = self.config['marginal_query_num']
        print('structure learning...')
        print('init marginal:')
        print('norm_budget: {:.4f}, marginal_budget: {:.4f}'.format(norm_budget, marginal_budget))
        print('marginal_query_num:', marginal_query_num)
        for marginal in marginal_dict:
            hist = self.__get_data_marginal(marginal, data, self.domain, weights)
            tvd = tools.get_TVD(hist, cp.asnumpy(marginal_dict[marginal].values))
            print('init marginal: {}, total: {:.4e}, tvd: {:.4f}'.format(marginal, float(cp.sum(marginal_dict[marginal].values)), tvd))

        if marginal_budget:
            marginal_noise = tools.get_Gaussian_noise(sensitivity, marginal_budget, query_num=marginal_query_num)
        else:
            marginal_noise = 0
        print('marginal_noise: {:.4f}, ratio: {:.4f}'.format(marginal_noise, marginal_noise/len(data)))
        self.marginal_noise = marginal_noise

        print('init_marginal_list:')
        print(list(init_marginal_dict.keys()))
        # get candidate marginals
        dom_limit = self.total / (marginal_noise * (2/math.pi) ** 0.5 + 1e-8) / self.config['theta']
        dom_limit = min(dom_limit, self.config['max_dom_limit'])
        print('max_attr_num:', max_attr_num)
        print('max dom_limit: {:.4f}, theta: {:.2f}'.format(dom_limit, self.config['theta']))
        candidate_list = self.get_candidate_marginal_list(dom_limit, max_attr_num)
        candidate_list = [(marginal, weight) for marginal, weight in candidate_list if marginal not in marginal_dict]

        # every candidate should contain size_attr
        candidate_list = [item for item in candidate_list if size_attr in item[0]]

        # estimate parameters with init marginal dict
        if len(marginal_dict) > 0:
            # terminate_loss = tools.get_theoretic_loss(self.domain, marginal_dict.keys(), marginal_noise) \
            #     * self.config['convergence_ratio']
            self.estimate_parameters(marginal_dict, estimate_iter_num)

        # select new marginals with norms and estimate parameters
        marginal_num_list = get_select_marginal_schedule(marginal_query_num, structure_learning_it)
        print('marginal_num_list:', len(marginal_num_list), marginal_num_list)
        if norm_budget:
            norm_budget = norm_budget / len(marginal_num_list)
        for i in range(len(marginal_num_list)):
            print(f'structure learnint it: {i}/{len(marginal_num_list)}')
            print(f'select {marginal_num_list[i]} from {len(candidate_list)}')
            select_list, candidate_list = self.select_marginal(
                data, marginal_num_list[i], candidate_list, norm_budget, sensitivity, weights=weights)
            
            for marginal in select_list:
                hist = self.__get_data_marginal(marginal, data, self.domain, weights)
                noisy_hist = hist + np.random.normal(scale=marginal_noise, size=hist.shape)
                tvd = tools.get_TVD(hist, noisy_hist)
                marginal_dict[marginal] = Factor(self.domain.project(marginal), noisy_hist)
                print('select marginal: {}, total: {:.4e}, query tvd: {:.4f}'.format(marginal, float(np.sum(noisy_hist)), tvd))

            terminate_loss = tools.get_theoretic_loss(self.domain, marginal_dict.keys(), marginal_noise) \
                * self.config['convergence_ratio']
            self.estimate_parameters(marginal_dict, estimate_iter_num, terminate_loss)

        return marginal_dict

    def structure_learning(self, data, weights=None, norm_budget=None, marginal_budget=None, \
        sensitivity=1, init_marginal_dict=Potential({})):
        '''
        data:
            np.array, tabular data without keys
        norm_budget:
            privacy budget for querying norms to select marginals. None for non-private
        marginal_budget:
            privacy budget for querying selected marginals. None for non-private
        sensitivity:
            sensitivity of a norm or a marginal
        init_marginal_dict:
            Potential dict storing init marginals and their hists
        '''
        structure_learning_it = self.config['structure_learning_it']
        estimate_iter_num = self.config['estimate_iter_num']
        max_attr_num = self.config['max_attr_num']
        marginal_dict = init_marginal_dict.copy()
        marginal_dict.to_gpu()
        marginal_query_num = self.config['marginal_query_num']
        print('structure learning...')
        print('localtime:', tools.get_time())
        print('init marginal:')
        print(f'norm_budget: {norm_budget}, marginal_budget: {marginal_budget}')
        print('marginal_query_num:', marginal_query_num)
        for marginal in marginal_dict:
            hist = self.__get_data_marginal(marginal, data, self.domain, weights)
            tvd = tools.get_TVD(hist, cp.asnumpy(marginal_dict[marginal].values))
            print('init marginal: {}, total: {:.4e} tvd: {:.4f}'.format(marginal, float(cp.sum(marginal_dict[marginal].values)), tvd))

        if marginal_budget:
            marginal_noise = tools.get_Gaussian_noise(sensitivity, marginal_budget, query_num=marginal_query_num)
        else:
            marginal_noise = 0
        print('marginal_noise: {:.4f}, ratio: {:.4f}'.format(marginal_noise, marginal_noise/len(data)))
        self.marginal_noise = marginal_noise

        print('init_marginal_list:')
        print(list(init_marginal_dict.keys()))
        # get candidate marginals
        dom_limit = self.total / (marginal_noise * (2/math.pi) ** 0.5 + 1e-8) / self.config['theta']
        print('max_attr_num:', max_attr_num)
        print('max dom_limit: {:.4f}, theta: {:.2f}'.format(dom_limit, self.config['theta']))
        candidate_list = self.get_candidate_marginal_list(dom_limit, max_attr_num)
        candidate_list = [(marginal, weight) for marginal, weight in candidate_list if marginal not in marginal_dict]

        # estimate parameters with init marginal dict
        if len(marginal_dict) > 0:
            terminate_loss = tools.get_theoretic_loss(self.domain, marginal_dict.keys(), marginal_noise) \
                * self.config['convergence_ratio']
            self.estimate_parameters(marginal_dict, estimate_iter_num, terminate_loss)

        # select new marginals with norms and estimate parameters
        marginal_num_list = get_select_marginal_schedule(marginal_query_num, structure_learning_it)
        print('marginal_num_list:', len(marginal_num_list), marginal_num_list)
        if norm_budget:
            norm_budget = norm_budget / len(marginal_num_list)
        for i in range(len(marginal_num_list)):
            print(f'structure learnint it: {i}/{len(marginal_num_list)}')
            print(f'select {marginal_num_list[i]} from {len(candidate_list)}')
            select_list, candidate_list = self.select_marginal(
                data, marginal_num_list[i], candidate_list, norm_budget, sensitivity)
            
            for marginal in select_list:
                hist = self.__get_data_marginal(marginal, data, self.domain, weights)
                noisy_hist = hist + np.random.normal(scale=marginal_noise, size=hist.shape)
                tvd = tools.get_TVD(hist, noisy_hist)
                marginal_dict[marginal] = Factor(self.domain.project(marginal), noisy_hist)
                print('select marginal: {}, domain: {:.4e}, total: {:.4e}, tvd: {:.4f}'.\
                    format(marginal, self.domain.project(marginal), float(np.sum(noisy_hist)), tvd))

            terminate_loss = tools.get_theoretic_loss(self.domain, marginal_dict.keys(), marginal_noise) \
                * self.config['convergence_ratio']
            self.estimate_parameters(marginal_dict, estimate_iter_num, terminate_loss)
        print('localtime:', tools.get_time())

        return marginal_dict

    def get_candidate_marginal_list(self, dom_limit, max_attr_num):
        clique_to_marginal = {}
        for clique in self.maximal_cliques:
            clique_to_marginal[clique] = []
            for attr_num in range(1, max_attr_num+1):
                for marginal in itertools.combinations(clique, attr_num):
                    # print(marginal, self.domain.project(marginal).size())
                    if self.domain.project(marginal).size() < dom_limit:
                        clique_to_marginal[clique].append(marginal)

        # normalize marginal weights such that will not always sample marginals in large cliques.
        marginal_list = []
        for clique in clique_to_marginal:
            if len(clique_to_marginal[clique]) == 0:
                continue
            weight = len(clique)**2/len(clique_to_marginal[clique])
            for marginal in clique_to_marginal[clique]:
                marginal_list.append(tuple([marginal, weight]))

        marginal_dict = {}
        for marginal, weight in marginal_list:
            if marginal in marginal_dict:
                marginal_dict[marginal] += weight
            else:
                marginal_dict[marginal] = weight

        marginal_list = list(marginal_dict.items())
        return marginal_list
    
    def select_marginal(self, data, select_num, candidate_list: list, norm_budget, sensitivity, weights=None):
        # get norm_query_number candidates
        norm_query_number = self.config['norm_query_number']
        print('norm_query_number:', norm_query_number)
        if len(candidate_list) > norm_query_number:
            p_weights = np.array([item[1] for item in candidate_list])
            p_weights /= np.sum(p_weights)
            query_marginal_list = np.random.choice(len(candidate_list), size=norm_query_number, p=p_weights, replace=False)
            query_marginal_list = [candidate_list[idx][0] for idx in query_marginal_list]
        else:
            query_marginal_list = [item[0] for item in candidate_list]

        # get norm noise
        if norm_budget:
            norm_noise = tools.get_Gaussian_noise(sensitivity, norm_budget, len(query_marginal_list))
        else:
            norm_noise = 0
        print('norm_noise: {:.4f}, ratio: {:.4f}'.format(norm_noise, norm_noise/len(data)/2))

        # query norm
        result_list = []
        model_marginal, _ = self.cal_marginal_dict(self.potential, query_marginal_list, to_cpu=True)
        for marginal in query_marginal_list:
            data_hist = self.__get_data_marginal(marginal, data, self.domain, weights)

            norm = np.sum(np.abs(data_hist - model_marginal[marginal].values))
            noisy_norm = norm + np.random.normal(scale=norm_noise)
            noisy_norm, norm = noisy_norm/self.total/2, norm/self.total/2

            result_list.append([marginal, noisy_norm, norm])

            # print('query norm:', marginal, data_hist.sum(), model_marginal[marginal].values.sum())

        # select select_num marginals and remove them from candidate_list
        result_list.sort(key=lambda x: x[1], reverse=True)
        for marginal, noisy_norm, norm in result_list[:5]:
            print('    query marginal norm: {}, {:.4f}, {:.4f}'.format(marginal, noisy_norm, norm))
        result_list = [item[0] for item in result_list[:select_num]]
        candidate_list = [(marginal, weight) for marginal, weight in candidate_list if marginal not in result_list]

        return result_list, candidate_list
    
    def generate_synthetic_data_by_substitution(self, data, target_attr, max_workers=20) -> np.ndarray:
        print('localtime:', tools.get_time())
        print('generate_synthetic_data_by_substitution:', target_attr)
        
        columns_to_groupby = list(range(data.shape[1]))
        columns_to_groupby.remove(target_attr)
        
        potential = self.potential.mov_to_cpu()
        clique_marginal_dict, partition_func = self.belief_propagation(self.potential)

        data = data[np.lexsort(data[:, columns_to_groupby].T)]
        chunk_boundaries = np.linspace(0, len(data), max_workers + 1).astype(int)

        chunk_list = []
        lower_bound = 0

        for i in range(max_workers):
            upper_bound = chunk_boundaries[i + 1]
            if upper_bound <= lower_bound:
                continue

            # Adjusting the upper bound so that rows with the same value stay in the same chunk
            while upper_bound < len(data) - 1 \
                and (data[upper_bound][columns_to_groupby] == data[upper_bound + 1][columns_to_groupby]).all():
                upper_bound += 1

            chunk = data[lower_bound:upper_bound]
            chunk_list.append(chunk)

            lower_bound = upper_bound

        def apply_df(chunk_data):
            chunk_data = chunk_data.copy()
            unique_groups, group_indices = np.unique(chunk_data[:, columns_to_groupby], axis=0, return_inverse=True)
            target_values = np.empty(chunk_data.shape[0], dtype=int)
            
            for idx, group in enumerate(unique_groups):
                
                x_idx = list(group)
                x_idx.insert(target_attr, slice(None))
                prob = tools.get_log_prob(potential, x_idx, partition_func)
                prob = np.exp(prob)

                group_rows = np.where(group_indices == idx)[0]
                vals = tools.generate_column_data(prob, len(group_rows))
                target_values[group_rows] = vals

            chunk_data[:, target_attr] = target_values
            return chunk_data

        try:
            chunk_list = Parallel(n_jobs=max_workers)(delayed(apply_df)(chunk_data) for chunk_data in chunk_list)
        except Exception as e:
            print("An error occurred during parallel processing:", e)
            traceback.print_exc()
            raise
        data = np.concatenate(chunk_list, axis=0)


        return data

    # generate synthetic data for self.domain.attr_list, other attributes will be set -1.
    def generate_synthetic_data(self, data=None, existing_attrs=[], ignore_error=False):
        if data is None:
            data = -np.ones((int(self.total), max(self.domain.attr_list)+1), dtype=int)
        existing_attrs = existing_attrs.copy()

        clique_marginal_dict, partition_func = self.belief_propagation(self.potential)
        clique_marginal_dict.to_cpu()

        start_clique = None
        for clique in self.maximal_cliques:
            if set(existing_attrs) <= set(clique):
                start_clique = clique
                break
        
        if not start_clique:
            if ignore_error:
                start_clique = self.maximal_cliques[0]
            else:
                print('ERROR: can not find a clique containing all existing attrs')
                assert(0)

        self.generate_synthetic_clique_data(data, clique_marginal_dict[start_clique], start_clique, existing_attrs)
        existing_attrs = sorted(set(existing_attrs + list(start_clique)))
        
        for _, clique in nx.dfs_edges(self.junction_tree, source=start_clique):
            self.generate_synthetic_clique_data(data, clique_marginal_dict[clique], clique, existing_attrs)
            existing_attrs = sorted(set(existing_attrs + list(clique)))

        return data

    # Generate synthetic data in place
    def generate_synthetic_clique_data(self, data, clique_factor, clique, existing_attrs=[]):
        existing_attrs = existing_attrs.copy()
        if existing_attrs:
            existing_attrs = sorted(set(existing_attrs).intersection(clique))
            
        print(f'sampling clique {clique}, existing_attrs {existing_attrs}')

        for attr in clique:
            if attr in existing_attrs:
                continue
            self.generate_synthetic_column_data(data, clique_factor, existing_attrs, attr)
            existing_attrs.append(attr)
            existing_attrs = sorted(existing_attrs)

    # Generate synthetic data in place
    def generate_synthetic_column_data(self, data, clique_factor: Factor, cond, target, value0_invalid=False):
        '''
        value0_invalid:
            Will not sample 0 if it is True
        '''
        print(f'    sampling attr {target} conditioned on {cond}')
        assert(target not in cond)
        assert(tuple(cond) == tuple(sorted(cond)))
        assert(clique_factor.domain.attr_list == sorted(clique_factor.domain.attr_list))

        if len(cond) == 0:
            prob = clique_factor.project(target).values
            if value0_invalid:
                prob[0] = 0
            data[:, target] = tools.generate_column_data(prob, len(data))
        else:
            marginal_value = clique_factor.project(cond + [target])
            marginal_value = marginal_value.moveaxis(cond + [target]).values
            if value0_invalid:
                marginal_value[..., 0] = 0
            
            unique_groups, group_indices = np.unique(data[:, cond], axis=0, return_inverse=True)
            target_values = np.empty(data.shape[0], dtype=int)  # Adjust dtype as needed
            
            for idx, group_idx in enumerate(unique_groups):
                group_rows = np.where(group_indices == idx)[0]
                vals = tools.generate_column_data(marginal_value[tuple(group_idx)], len(group_rows))
                # print('debug:')
                # print(idx)
                # print(group_idx)
                # print(marginal_value[idx].shape)
                # print(marginal_value[idx])
                # print(group_rows.shape)
                # print(group_rows)
                # print(vals.shape)
                # print(vals)
                # print()
                target_values[group_rows] = vals

            data[:, target] = target_values


    def save_model(self, path):
        with open(path, 'wb') as out_file:
            print('save model:', path)
            pickle.dump(self, out_file)

    @staticmethod
    def load_model(path) -> 'MarkovRandomField':
        with open(path, 'rb') as out_file:
            print('load model:', path)
            return pickle.load(out_file)