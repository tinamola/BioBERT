import numpy as np
import itertools
import matplotlib as mpl
import networkx as nx
import matplotlib
matplotlib.use('Agg') # 为了不show plot
import matplotlib.pyplot as plt
import json
from itertools import compress

#install了 networkx 和 seaborn
rc={'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 10.0,
    'axes.titlesize': 32, 'xtick.labelsize': 20, 'ytick.labelsize': 16}
plt.rcParams.update(**rc)
mpl.rcParams['axes.linewidth'] = .5 #set the value globally
import seaborn as sns; sns.set()
sns.set_style("whitegrid")
mpl.rcParams['axes.linewidth'] = 0.0 #set the value globally

def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length)) #(13x14,13x14)
    labels_to_index = {} #{'0_[CLS]': 0, '1_Mar': 1, '2_##van': 2, '3_##o': 3, '4_was': 4, '5_born': 5, '6_in': 6, '7_[MASK]': 7, '8_.': 8, '9_[SEP]': 9, '10_[PAD]': 10, '11_[PAD]': 11, '12_[PAD]': 12, '13_[PAD]': 13}
    for k in np.arange(length):
        labels_to_index[str(k) + "_" + input_tokens[k]] = k

    for i in np.arange(1, n_layers + 1):
        for k_f in np.arange(length):
            index_from = (i) * length + k_f
            label = "L" + str(i) + "_" + str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i - 1) * length + k_t
                adj_mat[index_from][index_to] = mat[i - 1][k_f][k_t]  # [14][0]=[0][0][0]  [14][13]=[0][0][13]  [15][0]=[0][1][0]

    return adj_mat, labels_to_index  #adj_mat是182*182  对称轴上的13*14有东西，其余都是0


def draw_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i, j): A[i, j]}, 'capacity')

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers + 1):
        for k_f in np.arange(length):
            pos[i * length + k_f] = ((i + 0.5) * 2, length - k_f)
            label_pos[i * length + k_f] = (i * 2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    # plt.figure(1,figsize=(20,12))

    nx.draw_networkx_nodes(G, pos, node_color='green', node_size=50)
    nx.draw_networkx_labels(G, pos=label_pos, labels=index_to_labels, font_size=10)

    all_weights = []
    # 4 a. Iterate through the graph nodes to gather all the weights
    for (node1, node2, data) in G.edges(data=True):
        all_weights.append(data['weight'])  # we'll use this when determining edge thickness

    # 4 b. Get unique weights
    unique_weights = list(set(all_weights))

    # 4 c. Plot the edges - one by one!
    for weight in unique_weights:
        # 4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in G.edges(data=True) if
                          edge_attr['weight'] == weight]
        # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner

        w = weight  # (weight - min(all_weights))/(max(all_weights) - min(all_weights))
        width = w
        nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width, edge_color='darkblue')

    return G


def get_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i, j): A[i, j]}, 'capacity')

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers + 1):
        for k_f in np.arange(length):
            pos[i * length + k_f] = ((i + 0.5) * 2, length - k_f)
            label_pos[i * length + k_f] = (i * 2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    # plt.figure(1,figsize=(20,12))

    nx.draw_networkx_nodes(G, pos, node_color='green', node_size=50)
    nx.draw_networkx_labels(G, pos=label_pos, labels=index_to_labels, font_size=10)

    all_weights = []
    # 4 a. Iterate through the graph nodes to gather all the weights
    for (node1, node2, data) in G.edges(data=True):
        all_weights.append(data['weight'])  # we'll use this when determining edge thickness

    # 4 b. Get unique weights
    unique_weights = list(set(all_weights))

    # 4 c. Plot the edges - one by one!
    for weight in unique_weights:
        # 4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in G.edges(data=True) if
                          edge_attr['weight'] == weight]
        # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner

        w = weight  # (weight - min(all_weights))/(max(all_weights) - min(all_weights))
        width = w
        nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width, edge_color='darkblue')

    return G


def compute_flows(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values = np.zeros((number_of_nodes, number_of_nodes))
    for key in labels_to_index:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]  #u是从15一直到end node maybe 182*182
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G, u, v, flow_func=nx.algorithms.flow.edmonds_karp,capacity="weight")
                flow_values[u][pre_layer * length + v] = flow_value
            flow_values[u] /= flow_values[u].sum()

    return flow_values


def compute_node_flow(G, labels_to_index, input_nodes, output_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values = np.zeros((number_of_nodes, number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G, u, v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer * length + v] = flow_value
            flow_values[u] /= flow_values[u].sum()

    return flow_values


def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i - 1])

    return joint_attentions


def plot_attention_heatmap(att,token_list, s_position, t_positions, bio, dir,query_len=None):
    fig=plt.figure(figsize=(12, 12),dpi=1200)
    cls_att = att[:, s_position, t_positions]
    if bio=="_bm25":
        filtered_index=np.array([cls_att[11][i] >= 1/len(cls_att[1]) or i<query_len for i in range(len(cls_att[11]))])
        token_list=list(compress(token_list, filtered_index))
        cls_att=cls_att[:,filtered_index]
    xticklb = list(itertools.compress(token_list,[i in t_positions for i in np.arange(len(token_list))]))
    yticklb = [str(i) if i % 2 == 0 else '' for i in np.arange(0,att.shape[0], 1)]
    sns.set(font_scale=1.5)

    ax = sns.heatmap(cls_att, xticklabels=xticklb, yticklabels=yticklb, cmap="Blues", annot=True, annot_kws={"size": 14},vmin=0, vmax=0.7)
    temp = token_list[:10] if len(token_list) >= 10 else token_list
    temp=" ".join(temp)
    temp= temp.replace('"','\'')
    temp = temp.replace('?','Ques')
    temp = temp.replace(':', 'Col')
    plt.tight_layout()
    fig.savefig('{}.png'.format(dir+temp+bio),dpi=1200)   # to avoid '  someone said: "i'm done"  ' give error
    plt.close(fig)


def convert_adjmat_tomats(adjmat, n_layers, l):
    mats = np.zeros((n_layers, l, l))

    for i in np.arange(n_layers):
        mats[i] = adjmat[(i + 1) * l:(i + 2) * l, i * l:(i + 1) * l]

    return mats