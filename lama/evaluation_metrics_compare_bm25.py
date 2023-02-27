# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
import scipy
import io
import sys
from attention_drawer import *
import threading
##7/14 new
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


lock = threading.Lock()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

def __max_probs_values_indices(masked_indices, log_probs, topk=1000):

    # score only first mask
    masked_indices = masked_indices[:1]

    masked_index = masked_indices[0]
    log_probs = log_probs[masked_index]

    value_max_probs, index_max_probs = torch.topk(input=log_probs,k=topk,dim=0)
    index_max_probs = index_max_probs.numpy().astype(int)
    value_max_probs = value_max_probs.detach().numpy()

    return log_probs, index_max_probs, value_max_probs


def __print_top_k(value_max_probs, index_max_probs, vocab, mask_topk, index_list, max_printouts = 10):
    result = []
    msg = "\n| Top{} predictions\n".format(max_printouts)
    for i in range(mask_topk):
        filtered_idx = index_max_probs[i].item()  #这个应该就是[20000]里面max prob的那个的index

        if index_list is not None:
            # the softmax layer has been filtered using the vocab_subset
            # the original idx should be retrieved
            idx = index_list[filtered_idx]   #filtered_idx is the idx in common vocab, get the idx of the same token in bert!!!!!!!!!!!!!!!!
        else:
            idx = filtered_idx

        log_prob = value_max_probs[i].item()
        word_form = vocab[idx]

        if i < max_printouts:
            msg += "{:<8d}{:<20s}{:<12.3f}\n".format(
                i,
                word_form,
                log_prob
            )
        element = {'i' : i, 'token_idx': idx, 'log_prob': log_prob, 'token_word_form': word_form}
        result.append(element)
    return result, msg

def printAttention(vocab,_attentions_bert,_attentions_biobert,token_ids_list,token_ids1,dir):
    final_list_final=[]
    token_list_final = []
    for lst in [token_ids_list,token_ids1]:
        token_list, add_list, final_list = [], [], [0]
        for idxx, index in enumerate(lst):
            if vocab[index].startswith("##"):
                token_list[-1] = token_list[-1] + vocab[index][2:]
                add_list[-1].append((idxx, index))
            elif index == 0:  #[PAD]
                break
            else:
                token_list.append(vocab[index])
                add_list.append([(idxx, index)])  # [[101],[9751,5242,1186],[1108],[103]]
        masked_indices_list=[token_list.index("[MASK]")]
        if idxx==len(lst)-1:  #没有[pad]的sentence
            idxx+=1
        for ex in add_list:
            if len(ex) == 1 and ex[0][0] != final_list[-1]:  #最后一个不需要进parse list
                final_list.append(ex[0][0])
            elif len(ex) > 1:
                if ex[0][0] != final_list[-1]:
                    final_list.append(ex[0][0])
                final_list.append(ex[-1][0] + 1)
        final_list_final.append(final_list)
        token_list_final.append(token_list)
    final_list_final[1]=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    token_list_final[1]=['[CLS]', 'London', 'Stock', 'Exchange', 'was', 'founded', 'in', '[MASK]', '.', '[SEP]', '£$€', 'is', 'a', 'nickname', 'for', 'the', 'following', ':', 'School', 'of', 'Economics', ',', 'a', 'public', 'research', 'university', 'located', 'in', ',', 'England', 'Stock', 'Exchange', ',', 'a', 'stock', 'exchange', 'located', 'in', 'the', 'City', 'of', ',', 'England', '[SEP]']
    #concantenate ##
    attentions_mat = np.add.reduceat(np.asarray(_attentions_bert)[:, :, :idxx, :idxx], final_list_final[0],axis=3)
    attentions_mat = np.add.reduceat(attentions_mat, final_list_final[0], axis=2)  #确保还是eye
    attentions_mat_bio = np.add.reduceat(np.asarray(_attentions_biobert)[:, :, :idxx, :idxx], final_list_final[1], axis=3)
    attentions_mat_bio = np.add.reduceat(attentions_mat_bio, final_list_final[1], axis=2)


    plot_attention_heatmap(attentions_mat.sum(axis=1) / attentions_mat.shape[1], token_list_final[0], masked_indices_list,
                           t_positions=list(range(len(token_list_final[0]))),bio="",dir=dir)
    plot_attention_heatmap(attentions_mat_bio.sum(axis=1) / attentions_mat_bio.shape[1], token_list_final[1], masked_indices_list,
                           t_positions=list(range(len(token_list_final[1]))),bio="_bm25",dir=dir,query_len=len(token_list_final[0]))

    # res_att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]  #12,14,14  res_att_mat[x层][y token]的sum是1
    # res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None, ...]
    # res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[..., None]
    # res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=token_list)
    #
    # res_att_mat_bio = attentions_mat_bio.sum(axis=1) / attentions_mat_bio.shape[1]  # 12,14,14  res_att_mat[x层][y token]的sum是1
    # res_att_mat_bio = res_att_mat_bio + np.eye(res_att_mat_bio.shape[1])[None, ...]
    # res_att_mat_bio = res_att_mat_bio / res_att_mat_bio.sum(axis=-1)[..., None]
    # res_adj_mat_bio, res_labels_to_index_bio = get_adjmat(mat=res_att_mat_bio, input_tokens=token_list)
    #
    # joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)
    # joint_attentions_bio = compute_joint_attention(res_att_mat_bio, add_residual=False)
    # plot_attention_heatmap(joint_attentions, token_list, masked_indices_list, t_positions=list(range(len(token_list))),bio="_joint",dir=dir)
    # plot_attention_heatmap(joint_attentions_bio, token_list, masked_indices_list, t_positions=list(range(len(token_list))),bio="_joint_bio",dir=dir)
    #
    # A = res_adj_mat
    # res_G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    # B = res_adj_mat_bio
    # res_G_bio = nx.from_numpy_matrix(B, create_using=nx.DiGraph())
    # output_nodes = []
    # input_nodes = []
    # for key in res_labels_to_index:
    #     if 'L24' in key:
    #         output_nodes.append(key)
    #     if res_labels_to_index[key] < attentions_mat.shape[-1]:
    #         input_nodes.append(key)
    #
    # flow_values = compute_flows(res_G, res_labels_to_index, input_nodes, length=attentions_mat.shape[-1])
    # flow_att_mat = convert_adjmat_tomats(flow_values, n_layers=attentions_mat.shape[0], l=attentions_mat.shape[-1])
    # flow_values_bio = compute_flows(res_G_bio, res_labels_to_index_bio, input_nodes, length=attentions_mat.shape[-1])
    # flow_att_mat_bio = convert_adjmat_tomats(flow_values_bio, n_layers=attentions_mat.shape[0], l=attentions_mat.shape[-1])
    #
    # plot_attention_heatmap(flow_att_mat, token_list, masked_indices_list, t_positions=list(range(len(token_list))),bio="_flow",dir=dir)
    # plot_attention_heatmap(flow_att_mat_bio, token_list, masked_indices_list, t_positions=list(range(len(token_list))),bio="_flow_bio",dir=dir)

def get_ranking(log_probs, masked_indices, token_ids, vocab, log_probs1,token_ids1, sample,sample_bm25, bert_model=None, sentence=None,sentence_bm25=None,atten_bert=None, atten_bm25=None, label_index = None, index_list = None, topk = 1000, P_AT = 10, print_generation=True):

    experiment_result = {}

    log_probs, index_max_probs, value_max_probs = __max_probs_values_indices(masked_indices, log_probs, topk=topk)
    result_masked_topk, return_msg = __print_top_k(value_max_probs, index_max_probs, vocab, topk, index_list)
    experiment_result['topk'] = result_masked_topk

    log_probs1, index_max_probs1, value_max_probs1 = __max_probs_values_indices(masked_indices, log_probs1, topk=topk)
    result_masked_topk1, return_msg1 = __print_top_k(value_max_probs1, index_max_probs1, vocab, topk, index_list)

    if print_generation:
        print(return_msg)
        print(return_msg1)

    MRR = 0.
    P_AT_X = 0.
    P_AT_1 = 0.
    PERPLEXITY = None
    situation=-1
    if label_index is not None:

        # check if the label_index should be converted to the vocab subset
        if index_list is not None:
            label_index = index_list.index(label_index)

        query = torch.full(value_max_probs.shape, label_index, dtype=torch.long).numpy().astype(int)
        ranking_position = (index_max_probs==query).nonzero()
        query1 = torch.full(value_max_probs1.shape, label_index, dtype=torch.long).numpy().astype(int)
        if value_max_probs1.shape!=value_max_probs.shape:
            print("it couldn't happen!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!86")
        ranking_position1 = (index_max_probs1 == query1).nonzero()

        # LABEL PERPLEXITY
        ########## changed this to .long() in my own computer, otherwise it will "Expected object of scalar type Float but got scalar type Long
        tokens = torch.from_numpy(np.asarray(label_index)).long()
        label_perplexity = log_probs.gather(
            dim=0,
            index=tokens,
        )
        PERPLEXITY = label_perplexity.item()

        if len(ranking_position) >0 and ranking_position[0].shape[0] != 0 and len(ranking_position1) >0 and ranking_position1[0].shape[0] != 0:
            rank = ranking_position[0][0] + 1
            rank1 = ranking_position1[0][0] + 1
            # print("rank: {}".format(rank))

            if rank >= 0:
                MRR = (1/rank)
            if rank >= 0 and rank <= P_AT:
                P_AT_X = 1.
            if rank == 1:
                P_AT_1 = 1.
            if rank == 1 and rank1 == 1:
                # print(sample)
                # print("Both successfully predicts P@1")
                situation=0
            elif rank == 1 and rank1 > P_AT:
                # print(sample)
                # print("Bert successfully predict P@1 but BioBERT didn't even predict P@10")
                # lock.acquire()
                # printAttention(vocab, atten_bert, atten_bio, token_ids, "img/bert@1_bio@10+/")
                # lock.release()

                # lock.acquire()
                # text1,grad1 = bert_model.calculate_saliency([sentence], [sample])
                # text2,grad2= bert_model.calculate_saliency([sentence_bm25], [sample_bm25])
                # lock.release()
                #
                # printSaliencyFunc(grad1[0], text1, "img/SelfAttenBert@1_bio@10+/", "")
                # printSaliencyFunc(grad1[1], text1, "img/AttenBert@1_bio@10+/", "")
                # printSaliencyFunc(grad1[2], text1, "img/OutputBert@1_bio@10+/", "")
                # printSaliencyFunc(grad2[0], text1, "img/SelfAttenBert@1_bio@10+/", "_Bio")
                # printSaliencyFunc(grad2[1], text1, "img/AttenBert@1_bio@10+/", "_Bio")
                # printSaliencyFunc(grad2[2], text1, "img/OutputBert@1_bio@10+/", "_Bio")
                # situation = 1

                situation=1
            elif rank == 1 and rank1 > 1:
                # print(sample)
                # print("Bert successfully predict P@1 but BioBERT didn't")
                situation = 2
            elif rank1 == 1 and rank > P_AT:
                # print("start----------------------------")
                # print(sample)
                # print("context successfully predict P@1 but BERT didn't even predict P@10")
                # lock.acquire()
                # printAttention(vocab, atten_bert, atten_bio, token_ids, "img/bert@10+_bio@1/")
                # lock.release()

                lock.acquire()
                text1, grad1 = bert_model.calculate_saliency([sentence], [sample])
                text2, grad2 = bert_model.calculate_saliency([sentence_bm25], [sample_bm25])
                lock.release()
                # lock.acquire()
                # printAttention(vocab, atten_bert, atten_bm25, token_ids,token_ids1, "img/context/AttenMapbert@10+_context@1/")
                # lock.release()
                filtered_index=np.array([i in [10,11,12,31,45] or i<len(text1) for i in range(len(grad2[0][11]))])
                filtered_text2=list(compress(text2, filtered_index))
                max_0=max(torch.max(torch.abs(grad1[0])),torch.max(torch.abs(grad2[0])))
                max_1=max(torch.max(torch.abs(grad1[1])),torch.max(torch.abs(grad2[1])))
                max_2=max(torch.max(torch.abs(grad1[2])),torch.max(torch.abs(grad2[2])))
                printSaliencyFunc(grad1[0], text1, "img/context/SelfAtten/", "",max_0)
                printSaliencyFunc(grad1[1], text1, "img/context/AttenBert/", "",max_1)
                printSaliencyFunc(grad1[2], text1, "img/context/OutputBert/", "",max_2)
                printSaliencyFunc(grad2[0][:,filtered_index], filtered_text2, "img/context/SelfAtten/", "BM25_",max_0)
                printSaliencyFunc(grad2[1][:,filtered_index], filtered_text2, "img/context/AttenBert/", "BM25_",max_1)
                printSaliencyFunc(grad2[2][:,filtered_index], filtered_text2, "img/context/OutputBert/", "BM25_",max_2)  #12,104
                situation = 3
            elif rank1 == 1 and rank > 1:
                # print(sample)
                # print("BioBERT successfully predict P@1 but BERT didn't")
                situation = 4


    experiment_result["MRR"] = MRR
    experiment_result["P_AT_X"] = P_AT_X
    experiment_result["P_AT_1"] = P_AT_1
    experiment_result["PERPLEXITY"] = PERPLEXITY
    return MRR, P_AT_X, experiment_result, return_msg,situation

def printSaliencyFunc(grad, text1, dir, bm,maxval):
    grad = torch.abs(grad)

    if bm=="BM25_":
        replace=torch.unsqueeze(torch.mean(grad[:, (10, 11, 12)], axis=1),axis=1)
        grad_temp=grad[:,range(0,10)]
        grad = torch.cat((grad_temp, replace, grad[:,range(13,15)]), 1)
        text1=['[CLS]', 'London', 'Stock', 'Exchange', 'was', 'founded', 'in', '[MASK]', '.', '[SEP]', '£$€','England','[SEP]']
        # grad=
    fig=plt.figure(dpi=400)
    xticklb = list(text1)
    yticklb = [str(i) if i % 2 == 0 else '' for i in np.arange(0, grad.shape[0], 1)]
    # cmap = sns.diverging_palette(240, 240, as_cmap=True)
    ax = sns.heatmap(grad, xticklabels=xticklb, yticklabels=yticklb, cmap="Oranges", annot=True, annot_kws={"size": 12,"fontsize":16},vmax=maxval)
    temp = text1[:10] if len(text1) >= 10 else text1
    temp = " ".join(temp)
    temp = temp.replace('"', '\'')
    temp = temp.replace('?', 'Ques')
    temp = temp.replace(':', 'Col')
    plt.tight_layout()
    fig.savefig('{}.png'.format(dir + bm+ temp))  # to avoid '  someone said: "i'm done"  ' give error
    plt.close(fig)

def __overlap_negation(index_max_probs__negated, index_max_probs):
    # compares first ranked prediction of affirmative and negated statements
    # if true 1, else: 0
    return int(index_max_probs__negated == index_max_probs)


def get_negation_metric(log_probs, masked_indices, log_probs_negated,
                        masked_indices_negated, vocab, index_list=None,
                        topk = 1):

    return_msg = ""
    # if negated sentence present
    if len(masked_indices_negated) > 0:

        log_probs, index_max_probs, _ = \
            __max_probs_values_indices(masked_indices, log_probs, topk=topk)
        log_probs_negated, index_max_probs_negated, _ = \
            __max_probs_values_indices(masked_indices_negated,
                                       log_probs_negated, topk=topk)

        # overlap btw. affirmative and negated first ranked prediction: 0 or 1
        overlap = __overlap_negation(index_max_probs_negated[0],
                                     index_max_probs[0])
        # rank corrl. btw. affirmative and negated predicted log_probs
        spearman_rank_corr = scipy.stats.spearmanr(log_probs,
                                                   log_probs_negated)[0]

    else:
        overlap = np.nan
        spearman_rank_corr = np.nan

    return overlap, spearman_rank_corr, return_msg
