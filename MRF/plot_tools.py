from . import tools
import matplotlib.pyplot as plt
import json
import networkx as nx
import numpy as np
import itertools

def plot_attr(attr, data, domain, path=None):

    plt.rcParams['figure.figsize'] = (11.0, 2.5)
    plt.rcParams['savefig.dpi'] = 200
    plt.locator_params(nbins=10)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    marginal1 = tools.get_marginal(data, domain, (attr,))
    # print(attr, 'marginal', marginal1)

    # marginal1 = marginal1[20: 40]
    ax.plot(marginal1)
    ax.plot([0]*len(marginal1), 'r')

    if len(marginal1) < 30:
        ax.set_xticks(list(range(len(marginal1))))
    if path == None:
        path = './'+str(attr)+'.pdf'
    plt.savefig(path, bbox_inches='tight')

def plot_img(img, path):
    plt.rcParams['figure.figsize'] = (11.0, 2.5)
    plt.rcParams['savefig.dpi'] = 200
    plt.locator_params(nbins=10)
    fig = plt.figure()

    plt.imshow(img)
    plt.colorbar()

    plt.savefig(path, bbox_inches='tight')

def plot_2way_marginal(marginal, data, domain, path=None):

    marginal1 = tools.get_marginal(data, domain, marginal)

    if path == None:
        path = './'+str(marginal)+'.pdf'
    plot_img(marginal1, path)

def plot_list(plot_data, path, zero_line=False, size=None, yticks=None, xticks=None):

    if size is None:
        size = (11.0, 2.5)
    plt.rcParams['figure.figsize'] = size
    plt.rcParams['savefig.dpi'] = 200
    plt.locator_params(nbins=10)
    fig = plt.figure()

    plt.plot(plot_data)
    if yticks is not None:
        plt.yticks(yticks)
    if xticks is not None:
        plt.xticks(xticks)

    if zero_line:
        plt.plot([0]*len(plot_data), 'r')

    plt.savefig(path, bbox_inches='tight')

def plot_list_list(plot_data_list, path, zero_line=False):

    plt.rcParams['figure.figsize'] = (11.0, 2.5)
    plt.rcParams['savefig.dpi'] = 200
    plt.locator_params(nbins=10)
    fig = plt.figure()

    for plot_data in plot_data_list:
        plt.plot(plot_data)

    if zero_line:
        plt.plot([0]*len(plot_data), 'r')

    plt.savefig(path, bbox_inches='tight')

def plot_x_y(x, y, path, zero_line=False):

    plt.rcParams['figure.figsize'] = (11.0, 2.5)
    plt.rcParams['savefig.dpi'] = 200
    plt.locator_params(nbins=10)
    fig = plt.figure()

    plt.plot(x, y, 'o')

    if zero_line:
        plt.plot(x, [0]*len(x), 'r')

    plt.savefig(path, bbox_inches='tight')


def plot_correlation(data, domain, edge_path, path=None, min_edge=2, MI_threshold=0.20):

    attr_num = data.shape[1]

    edge_list = json.load(open(edge_path))
    G = nx.Graph()
    G.add_nodes_from(list(range(attr_num)))
    adj = np.zeros(shape=(attr_num, attr_num), dtype=float)

    for marginal, MI in edge_list[:1000]:
        for attr1, attr2 in itertools.combinations(marginal, 2):
            adj[attr1, attr2] += MI
            adj[attr2, attr1] += MI

    new_edge_list = []
    for attr1 in range(attr_num):
        for attr2 in range(attr1+1, attr_num):
            new_edge_list.append([[attr1, attr2], adj[attr1, attr2]])
    new_edge_list.sort(key=lambda x: x[1], reverse=True)
    json.dump(new_edge_list, open('./temp/edge_list_3way.json', 'w'))

    adj = adj/np.max(adj)
    new_adj = np.zeros(shape=(attr_num, attr_num), dtype=float)

    
    for attr1 in range(attr_num):
        weights = list(adj[attr1])
        weights.sort(reverse=True)
        # plot at least min_edge edges for each attr
        threshold = min(weights[min_edge], MI_threshold)

        for attr2 in range(attr_num):
            if adj[attr1, attr2] >= threshold - 1e-5:
                new_adj[attr1, attr2] = adj[attr1, attr2]
                new_adj[attr2, attr1] = adj[attr2, attr1]

                # print(attr1, attr2, adj[attr1, attr2], threshold)


    for attr1 in range(attr_num):
        for attr2 in range(attr1+1, attr_num):
            if new_adj[attr1, attr2] > 0:
                G.add_edge(attr1, attr2, weight=new_adj[attr1, attr2])

    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    pos = nx.spring_layout(G)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    edge_cmap = plt.cm.viridis
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='g', node_size=120, \
        edgelist=edges, edge_color=weights,\
        font_size=12, edge_cmap=edge_cmap, width=0.5, vmin=0, vmax=1)

    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['savefig.dpi'] = 600
    
    sm = plt.cm.ScalarMappable(cmap=edge_cmap)
    sm._A = []
    plt.colorbar(sm)
    plt.show()


    if path == None:
        path = './temp/graph_correlation.png'
    plt.savefig(path)

def plot_count_distribution(data, domain, marginal, path):
    hist = tools.get_marginal(data, domain, marginal)
    unique, cnt = np.unique(hist, return_counts=True)

    unique = np.log10(unique)
    cdf = 0
    for i in range(len(cnt)):
        cdf += cnt[i]
        cnt[i] = cdf

    plot_x_y(unique, cnt, path, zero_line=True)