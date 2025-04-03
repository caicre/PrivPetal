# algorithms for building running graphs containing latent variables
import itertools
import networkx as nx
from . import tools
import json
import numpy as np
import itertools

def score_clique_limit(graph, edge_dict, \
    max_clique_size, max_parameter_size, domain):
    min_score = -1e7
    
    graph = graph.copy()
    if not nx.is_chordal(graph):
        graph = tools.triangulate(graph)

    sum_size = 0
    clique_list = [tuple(sorted(clique)) for clique in nx.find_cliques(graph)]
    for clique in clique_list:
        clique_size = domain.project(clique).size()
        if clique_size > max_clique_size:
            return min_score, "exceed max clique size" + str(clique_size), -1

        sum_size += clique_size
        if sum_size > max_parameter_size:
            return min_score, "exceed max parameter size" + str(sum_size), -1

    score = 0
    for edge in graph.edges:
        edge = tuple(sorted(edge))
        score += edge_dict[edge]

    return score, "succeed", sum_size

def score_additional_ratio(graph, last_score, last_size, edge_score, edge_dict, \
    max_clique_size, max_parameter_size, domain):
    min_score = -1e7

    graph = graph.copy()
    if not nx.is_chordal(graph):
        graph = tools.triangulate(graph)

    sum_size = 0
    clique_list = [tuple(sorted(clique)) for clique in nx.find_cliques(graph)]
    for clique in clique_list:
        clique_size = domain.project(clique).size()
        if clique_size > max_clique_size:
            return min_score, "exceed max clique size " + str(clique_size), -1, -1

        sum_size += clique_size
        if sum_size > max_parameter_size:
            return min_score, "exceed max parameter size " + str(sum_size), -1, -1

    score = 0
    for edge in graph.edges:
        score += edge_dict[edge]

    if sum_size > last_size:
        de = sum_size - last_size
    else:
        de = np.exp(sum_size - last_size)
    # May add edge in the triangulated graph, which incurs no incremental score and size. 
    # Choose large edge score in this case.
    marginal_ratio = (score - last_score + edge_score/1e5)/de

    return score, "succeed", sum_size, marginal_ratio

class AttributeGraph:
    def __init__(self, config, domain):
        """ config:
        max_clique_size:    max size for each clique of the junction tree
        max_parameter_size: max size for all the cliques
        exp_name:           identifier for temp files
        """
        self.config = config
        self.domain = domain
        self.attr_list = domain.attr_list
                
        self.attr_num = len(self.domain)

    def local_search_by_R(self, edge_list, data=None, init_MST=False, init_graph=None):
        attr_set = set(self.domain.attr_list)
        edge_list = [item for item in edge_list if set(item[0]) <= attr_set]

        print('attribute graph local search')
        if not data is None:
            data_entropy = tools.get_entropy({}, self.domain.attr_list, data, self.domain)
            print('data entropy: {:.4f}'.format(data_entropy))

        edge_list.sort(reverse=True, key=lambda x: x[1])
        edge_dict = {edge: noisy_score for edge, noisy_score, true_score in edge_list}

        # json.dump(edge_list, open('./temp/'+self.config['exp_name']+'_edge_list.json', 'w'))

        score_func = score_clique_limit
        max_clique_size = self.config['max_clique_size']
        max_parameter_size = self.config['max_parameter_size']

        if not init_graph is None:
            last_graph = init_graph.copy()
        elif init_MST:
            full_graph = nx.Graph()
            full_graph.add_nodes_from(self.domain.attr_list)
            for edge, n_score, score in edge_list:
                full_graph.add_edge(*edge, weight=n_score)
            MST = nx.maximum_spanning_tree(full_graph, weight='weight')
            tools.print_graph(MST, './temp/MST_'+self.config['exp_name']+'.png')

            last_graph = MST.copy()
            for edge in MST.edges:
                for i in range(len(edge_list)):
                    if edge_list[i][0] == edge:
                        del edge_list[i]
                        break
        else:
            last_graph = nx.Graph()
            last_graph.add_nodes_from(self.domain.attr_list)

        last_score, msg, last_size = score_func(last_graph, edge_dict,
            max_clique_size, max_parameter_size, self.domain)
        print('init score: {:.4e} size: {:.4e}'.format(last_score, last_size))
        add_flag = True
        it = 0
        while add_flag:
            add_flag = False
            
            for i in range(len(edge_list)):

                temp_edge, noisy_score, true_score = edge_list[i]
                temp_graph = last_graph.copy()
                temp_graph.add_edge(*temp_edge)
                temp_score, msg, temp_size = score_func(temp_graph, edge_dict,
                    max_clique_size, max_parameter_size, self.domain)
                
                # print(i, temp_edge, temp_score, temp_size)
                if temp_score > 0: # add edge successfully
                    last_graph = temp_graph
                    last_score = temp_score
                    last_size = temp_size
                    add_flag = True
                    del edge_list[i]

                    print('it: {:d}, num: {:d}, score: {:.2e}, edge: {}, size: {:.2e}, search: {}'\
                        .format(it, nx.classes.function.number_of_edges(last_graph), \
                        last_score, temp_edge, last_size, i))
                    break
            
            it += 1


        if not nx.is_chordal(last_graph):
            last_graph = tools.triangulate(last_graph)
        tools.print_graph(last_graph, './temp/graph_'+self.config['exp_name']+'.png')

        junction_tree = tools.get_junction_tree(last_graph)
        if not data is None:
            model_entropy = tools.get_junction_tree_entropy(junction_tree, data, self.domain)
            print('model_entropy: {:.4f}'.format(model_entropy))
        return last_graph, junction_tree


    def local_search(self, data, edge_list, max_it=1e6, choice_num=1000, search_num=100):
        print('attribute graph local search')
        data_entropy = tools.get_entropy({}, list(range(self.attr_num)), data, self.domain)
        print('data entropy: {:.4f}'.format(data_entropy))

        edge_list.sort(reverse=True, key=lambda x: x[1])
        edge_dict = {edge: noisy_score for edge, noisy_score, true_score in edge_list}

        json.dump(edge_list, open('./temp/'+self.config['exp_name']+'_edge_list.json', 'w'))

        # score_func = score_clique_limit
        score_func = score_additional_ratio
        max_clique_size = self.config['max_clique_size']
        max_parameter_size = self.config['max_parameter_size']

        full_graph = nx.Graph()
        full_graph.add_nodes_from(self.domain.attr_list)
        for edge, n_score, score in edge_list:
            full_graph.add_edge(*edge, weight=n_score)
        tools.print_graph(full_graph, './temp/full_graph_'+self.config['exp_name']+'.png')
        MST = nx.maximum_spanning_tree(full_graph, weight='weight')
        tools.print_graph(MST, './temp/MST_'+self.config['exp_name']+'.png')

        last_graph = MST.copy()
        for edge in MST.edges:
            for i in range(len(edge_list)):
                if edge_list[i][0] == edge:
                    del edge_list[i]
                    break

        # score, msg, size = score_func(graph, edge_dict, \
        #     max_clique_size, max_parameter_size, self.domain)
        last_score, msg, last_size, _ = score_func(last_graph, 0, 0, 0, edge_dict, \
            max_clique_size, max_parameter_size, self.domain)
        print('init score: {:.4e} size: {:.4e}'.format(last_score, last_size))
        add_flag = True
        it = 0
        while add_flag:
            add_flag = False
            marginal_ratio = 0

            search_i = 0
            if choice_num >= len(edge_list):
                choices = np.arange(len(edge_list))
            else:
                choices = np.random.choice(len(edge_list), size=choice_num, replace=False)
            for i in choices:
                temp_edge, noisy_score, true_score = edge_list[i]

                temp_graph = last_graph.copy()
                temp_graph.add_edge(*temp_edge)
                # temp_score, msg, temp_size = score_func(temp_graph, edge_dict,\
                #     max_clique_size, max_parameter_size, self.domain)
                temp_score, msg, temp_size, temp_marginal_ratio = score_func(temp_graph, last_score, last_size, noisy_score, edge_dict, \
                    max_clique_size, max_parameter_size, self.domain)
                # print('    edge: {}  score: {:.4e} edge_score: {:.4e}, size: {:.4e}, ratio: {:.4e}, msg: {}'.format(\
                #     temp_edge, temp_score, noisy_score, temp_size, temp_marginal_ratio, msg))
    
                if temp_marginal_ratio > 0:
                    search_i += 1
                if temp_marginal_ratio > marginal_ratio:
                    marginal_ratio = temp_marginal_ratio
                    edge = temp_edge
                    graph = temp_graph
                    score = temp_score
                    size = temp_size
                    idx = i
                if search_i > search_num:
                    break

            if marginal_ratio > 0:
                last_graph = graph
                last_score = score
                last_size = size

                del edge_list[idx]
                
                add_flag = True
                print('num: {:d}, score: {:.2e}, marginal_ratio: {:.2e}, edge: {}, size: {:.2e}, search: {}'\
                    .format(nx.classes.function.number_of_edges(graph), \
                    score, marginal_ratio, edge, size, search_i))
                it += 1

            if it > max_it:
                break
        print('num: {:d}, score: {:.2e}, marginal_ratio: {:.2e}, edge: {}, size: {:.2e}'\
                    .format(nx.classes.function.number_of_edges(graph), \
                    score, marginal_ratio, edge, size))

        # graph = nx.node_link_graph(json.load(open('./temp/exp1_3.20_graph.json', 'r')))
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        tools.print_graph(graph, './temp/graph_'+self.config['exp_name']+'.png')

        junction_tree = tools.get_junction_tree(graph)
        model_entropy = tools.get_junction_tree_entropy(junction_tree, data, self.domain)
        print('model_entropy: {:.4f}'.format(model_entropy))
        return graph, junction_tree
