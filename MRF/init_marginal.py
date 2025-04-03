import networkx as nx
from . import tools
import itertools
from .domain import Domain

def get_CFS_marginal(graph, edge_list: list, domain: Domain, dom_limit: float, attr_num_limit=6, target_attrs=None, check=None):
    """
    graph:
        networkx graph
    edge_list:
        list, each item is ((attr1, attr2), noisy_score, true_score), attr1 < attr2.
        scores should be normalized to (0, 1)
    target_attrs:
        get CFS marginal for all attrs if it is None. Otherwise, only for target_attrs. 
    """
    print('get CFS marginals')
    if target_attrs is None:
        target_attrs = set(domain.attr_list)
    else:
        target_attrs = set(target_attrs)
    # print(sorted(list(graph.nodes)))
    # print(domain.attr_list)
    assert(sorted(list(graph.nodes)) == domain.attr_list)

    edge_dict = {tuple(edge): noisy_score for edge, noisy_score, true_score in edge_list}
    maximal_cliques = [tuple(sorted(clique)) for clique in nx.find_cliques(graph)]
    # print(maximal_cliques)

    candidate_marginals = itertools.chain.from_iterable(\
        [tools.get_maximal_cliques(clique, domain, dom_limit, check=check) for clique in maximal_cliques])

    marginal_list = select_CFS_marginal(candidate_marginals, target_attrs, domain, edge_dict)

    return marginal_list


def select_CFS_marginal(candidate_marginals, target_attrs, edge_dict):
    if type(target_attrs) != set:
        target_attrs = set(target_attrs)

    attr_to_feature = {attr: (None, -1e6) for attr in target_attrs}

    candidate_marginals = [set(marginal) for marginal in candidate_marginals]
    candidate_marginals2 = []
    for marginal in candidate_marginals:
        max_flag = True
        for pa_marginal in candidate_marginals:
            if marginal < pa_marginal:
                max_flag = False
                break
        if max_flag:
            candidate_marginals2.append(tuple(sorted(marginal)))

    for marginal in candidate_marginals2:
        for target in target_attrs.intersection(marginal):
            score = tools.get_CFS_score(target, marginal, edge_dict)
            # print('   ', target, marginal, score)
            if score > attr_to_feature[target][1]:
                attr_to_feature[target] = (marginal, score)
    
    for attr, item in attr_to_feature.items():
        print('    attr_to_feature:', attr, item)

    marginal_list = [item[0] for attr, item in attr_to_feature.items() if attr in target_attrs]
    marginal_list = [item for item in marginal_list if not item is None]

    marginal_list = [tuple(sorted(marginal)) for marginal in marginal_list]
    marginal_list = set(marginal_list)
    marginal_list = [tuple(sorted(marginal)) for marginal in marginal_list]

    return marginal_list

