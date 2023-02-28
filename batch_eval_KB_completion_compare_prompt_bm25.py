from lama.modules import build_model_by_name
import lama.utils as utils
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
from tqdm import tqdm
from random import shuffle
import json
import lama.modules.base_connector as base
from pprint import pprint
import logging.config
import logging
import pickle
from multiprocessing.pool import ThreadPool
import multiprocessing
import lama.evaluation_metrics_compare_bm25 as metrics
import time, sys, numpy as np
import re
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
from datasets import load_dataset,concatenate_datasets
import elasticsearch
wiki = load_dataset("wikipedia", "20220301.en",split="train",cache_dir="D:\OneDrive\Desktop\WikipediaCorpus")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
bookcorpus = load_dataset("bookcorpus", split="train", cache_dir="D:\OneDrive\Desktop\BookCorpus")
bert_dataset = concatenate_datasets([bookcorpus, wiki])
es_client=elasticsearch.Elasticsearch('http://localhost:9200',http_auth=("elastic","your token here"),timeout=30)
bert_dataset.load_elasticsearch_index('text',es_client=es_client,es_index_name="bert_dataset")
num_samples=10

#installed networkx and seaborn

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def create_logdir_with_timestamp(base_logdir, modelname):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, modelname, timestr)
    os.makedirs(log_directory)

    path = "{}/last".format(base_logdir)
    try:
        os.unlink(path)
    except Exception:
        pass
    os.symlink(log_directory, path)
    return log_directory


def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]


def init_logging(log_directory):
    logger = logging.getLogger("LAMA")
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_directory, exist_ok=True)

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler
    ######################## add "w","utf-8" so that the console is neater
    fh = logging.FileHandler(str(log_directory) + "/info.log", "w",'utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


def batchify(data, data1, batch_size):
    msg = ""
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    list_samples_batches1 = []
    list_sentences_batches1 = []
    current_samples_batch1 = []
    current_sentences_batches1 = []
    c = 0
    list1, list2 = zip(*sorted(zip(data, data1),key=lambda x: len(" ".join(x[0]["masked_sentences"]).split())))
    # sort to group together sentences with similar length
    for sample,sample1 in zip(list1,list2):
        masked_sentences = sample["masked_sentences"]
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentences)
        masked_sentences1 = sample1["masked_sentences"]
        current_samples_batch1.append(sample1)
        current_sentences_batches1.append(masked_sentences1)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            list_samples_batches1.append(current_samples_batch1)
            list_sentences_batches1.append(current_sentences_batches1)
            current_samples_batch1 = []
            current_sentences_batches1 = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)
        list_samples_batches1.append(current_samples_batch1)
        list_sentences_batches1.append(current_sentences_batches1)

    return list_samples_batches, list_sentences_batches,list_samples_batches1, list_sentences_batches1, msg

def run_thread(arguments,arguments1):
    msg = ""
    # 1. attention heatmap
    sample_MRR, sample_P, experiment_result, return_msg, situation = metrics.get_ranking(
        arguments["filtered_log_probs"],
        arguments["masked_indices"],
        arguments["token_ids"],
        arguments["vocab"],    #这三个是新加的
        arguments1["filtered_log_probs"],
        arguments1["token_ids"],
        arguments["sample"],
        arguments1["sample"],
        label_index=arguments["label_index"],
        index_list=arguments["index_list"],
        print_generation=arguments["interactive"],
        bert_model=arguments["model"],  #这五个是新加的
        sentence=arguments["sentence"],
        sentence_bm25=arguments1["sentence"],
        atten_bert=arguments["attention"],
        atten_bm25=arguments1["attention"],
        topk=10000
    )
    msg += "\n" + return_msg

    sample_perplexity = 0.0
    if arguments["interactive"]:
        pprint(arguments["sample"])
        # THIS IS OPTIONAL - mainly used for debuggind reason
        # 2. compute perplexity and print predictions for the complete log_probs tensor
        sample_perplexity, return_msg = print_sentence_predictions(
            arguments["original_log_probs"],
            arguments["token_ids"],
            arguments["vocab"],
            masked_indices=arguments["masked_indices"],
            print_generation=arguments["interactive"],
        )
        input("press enter to continue...")
        msg += "\n" + return_msg

    return experiment_result, sample_MRR, sample_P, sample_perplexity, msg, situation


def lowercase_samples(samples, use_negated_probes=False):
    new_samples = []
    for sample in samples:
        sample["obj_label"] = sample["obj_label"].lower()
        sample["sub_label"] = sample["sub_label"].lower()
        lower_masked_sentences = []
        for sentence in sample["masked_sentences"]:
            sentence = sentence.lower()
            sentence = sentence.replace(base.MASK.lower(), base.MASK)
            lower_masked_sentences.append(sentence)
        sample["masked_sentences"] = lower_masked_sentences

        if "negated" in sample and use_negated_probes:
            for sentence in sample["negated"]:
                sentence = sentence.lower()
                sentence = sentence.replace(base.MASK.lower(), base.MASK)
                lower_masked_sentences.append(sentence)
            sample["negated"] = lower_masked_sentences

        new_samples.append(sample)
    return new_samples


def filter_samples(model, samples, vocab_subset, max_sentence_length, template):
    msg = ""
    new_samples = []
    samples_exluded = 0

    for sample in samples:
        excluded = False

        if "obj_label" in sample and "sub_label" in sample:

            obj_label_ids = model.get_id(sample["obj_label"])

            if obj_label_ids:
                recostructed_word = " ".join(
                    [model.vocab[x] for x in obj_label_ids]
                ).strip()
            else:
                recostructed_word = None

            excluded = False
            if not template or len(template) == 0:
                masked_sentences = sample["masked_sentences"]
                text = " ".join(masked_sentences)
                if len(text.split()) > max_sentence_length:
                    msg += "\tEXCLUDED for exeeding max sentence length: {}\n".format(
                        masked_sentences
                    )
                    samples_exluded += 1
                    excluded = True

            # MAKE SURE THAT obj_label IS IN VOCABULARIES
            if vocab_subset:
                for x in sample["obj_label"].split(" "):
                    if x not in vocab_subset:
                        excluded = True
                        msg += "\tEXCLUDED object label {} not in vocab subset\n".format(
                            sample["obj_label"]
                        )
                        samples_exluded += 1
                        break

            if excluded:
                pass
            elif obj_label_ids is None:   #不在bert里，但在common里（不在common里的exclude了）也exclude
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1

            elif not recostructed_word or recostructed_word != sample["obj_label"]:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1

            # elif vocab_subset is not None and sample['obj_label'] not in vocab_subset:
            #   msg += "\tEXCLUDED object label {} not in vocab subset\n".format(sample['obj_label'])
            #   samples_exluded+=1
            elif "judgments" in sample:
                # only for Google-RE
                num_no = 0
                num_yes = 0
                for x in sample["judgments"]:
                    if x["judgment"] == "yes":
                        num_yes += 1
                    else:
                        num_no += 1
                if num_no > num_yes:
                    # SKIP NEGATIVE EVIDENCE
                    pass
                else:
                    new_samples.append(sample)
            else:
                new_samples.append(sample)
        else:
            msg += "\tEXCLUDED since 'obj_label' not sample or 'sub_label' not in sample: {}\n".format(
                sample
            )
            samples_exluded += 1
    msg += "samples exluded  : {}\n".format(samples_exluded)
    return new_samples, msg

def main(args, shuffle_data=True, model=None):
    if len(args.models_names) > 1:
        raise ValueError('Please specify a single language model (e.g., --lm "bert").')

    msg = ""

    [model_type_name] = args.models_names

    print(model)
    if model is None :
        model = build_model_by_name(model_type_name, args)

    if model_type_name == "fairseq":
        model_name = "fairseq_{}".format(args.fairseq_model_name)
    elif model_type_name == "bert":
        model_name = "BERT_{}".format(args.bert_model_name)
    elif model_type_name == "elmo":
        model_name = "ELMo_{}".format(args.elmo_model_name)
    else:
        model_name = model_type_name.title()

    # initialize logging
    if args.full_logdir:
        log_directory = args.full_logdir
    else:
        log_directory = create_logdir_with_timestamp(args.logdir, model_name)
    logger = init_logging(log_directory)
    msg += "model name: {}\n".format(model_name)

    # deal with vocab subset
    vocab_subset = None
    index_list = None
    msg += "args: {}\n".format(args)
    if args.common_vocab_filename is not None:
        vocab_subset = load_vocab(args.common_vocab_filename)
        msg += "common vocabulary size: {}\n".format(len(vocab_subset))

        # optimization for some LM (such as ELMo)
        # model.optimize_top_layer(vocab_subset)  ###########################################

        filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(
            vocab_subset, logger
        )

    logger.info("\n" + msg + "\n")

    # dump arguments on file for log
    with open("{}/args.json".format(log_directory), "w") as outfile:
        json.dump(vars(args), outfile)

    # stats
    samples_with_negative_judgement = 0
    samples_with_positive_judgement = 0

    # Mean reciprocal rank
    MRR = 0.0
    MRR_negative = 0.0
    MRR_positive = 0.0

    # Precision at (default 10)
    Precision = 0.0
    Precision1 = 0.0
    Precision_negative = 0.0
    Precision_positivie = 0.0

    data = load_file(args.dataset_filename)

    print(len(data))

    all_samples, ret_msg = filter_samples(
        model, data, vocab_subset, args.max_sentence_length, args.template
    )
    logger.info("\n" + ret_msg + "\n")

    print(len(all_samples))

    # if template is active (1) use a single example for (sub,obj) and (2) ...
    if args.template and args.template != "":
        facts = []
        for sample in all_samples:
            sub = sample["sub_label"]
            obj = sample["obj_label"]
            if (sub, obj) not in facts:
                facts.append((sub, obj))
        local_msg = "distinct template facts: {}".format(len(facts))
        logger.info("\n" + local_msg + "\n")
        print(local_msg)
        all_samples = []
        for fact in facts:
            (sub, obj) = fact
            sample = {}
            sample["sub_label"] = sub
            sample["obj_label"] = obj
            # substitute all sentences with a standard template
            sample["masked_sentences"] = parse_template(
                args.template.strip(), sample["sub_label"].strip(), base.MASK
            )
            if args.use_negated_probes:
                # substitute all negated sentences with a standard template
                sample["negated"] = parse_template(
                    args.template_negated.strip(),
                    sample["sub_label"].strip(),
                    base.MASK,
                )
            all_samples.append(sample)

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1

    def myfunc(sample):
        object_token = re.compile(re.escape(sample['obj_label']), re.IGNORECASE)
        query = sample['masked_sentences'][0].replace("[MASK]", "")
        scores,retrieved=bert_dataset.get_nearest_examples("text",query,k=num_samples)
        new_retrieved=[object_token.sub("",l) for l in [" ".join(k) for k in [j.split() for j in [retrieved['text'][i] for i in range(num_samples)]] if len(k) < 450]]
        if len(new_retrieved)>0:
            new_retrieved=new_retrieved[0]
        else:
            new_retrieved=object_token.sub(""," ".join(retrieved['text'][0].split()[:450]))
        sample['masked_sentences'].append(new_retrieved)
        return sample
    all_bm25_samples=list(map(myfunc,all_samples))
    with open('produced_context_bm25/%s.json' % args.relation, "a", encoding='utf-8') as fp:
        for sample in all_bm25_samples:
            json.dump(sample, fp)
            fp.write("\n")
    # all_bm25_samples = []
    # for line in open('bm25_produced_context/%s.json' % args.relation, 'r'):
    #     all_bm25_samples.append(json.loads(line))
    # samples_batches, sentences_batches, samples_batches_bm25, sentences_batches_bm25, ret_msg = batchify(all_samples, all_bm25_samples, args.batch_size)  #[[32],[32]] first is whole sample, second is masksed sentence
    #
    #
    # # ThreadPool
    # num_threads = args.threads
    # if num_threads <= 0:
    #     # use all available threads
    #     num_threads = multiprocessing.cpu_count()
    # pool = ThreadPool(num_threads)
    # ########################
    # #####Count1,2,3,4,5#####
    # ########################
    # final_condition = []
    # list_of_results = []
    #
    # for i in tqdm(range(len(samples_batches))):
    #
    #     samples_b = samples_batches[i]
    #     sentences_b = sentences_batches[i]
    #     samples_b_bm25 = samples_batches_bm25[i]
    #     sentences_b_bm25 = sentences_batches_bm25[i]
    #
    #     (
    #         original_log_probs_list,
    #         token_ids_list,
    #         masked_indices_list,
    #         attention_map_bert,
    #     ) = model.get_batch_generation(sentences_b,output_atten=True)
    #     (
    #         original_log_probs_list1, #32,14,28996
    #         token_ids_list1,  #32,14
    #         masked_indices_list1, #32,1
    #         attention_map_bm25,  #12,32,12,14,14
    #     ) = model.get_batch_generation(sentences_b_bm25,output_atten=True)
    #     # print(attention_map1[0].shape)
    #     _attentions_bert = np.swapaxes([att.detach().cpu().numpy() for att in attention_map_bert],0,1)
    #     _attentions_bert_bm25 = np.swapaxes([att.detach().cpu().numpy() for att in attention_map_bm25],0,1)
    #
    #     if vocab_subset is not None:
    #         # filter log_probs
    #         filtered_log_probs_list = model.filter_logprobs(
    #             original_log_probs_list, filter_logprob_indices   #filtered_log_probs_list的20000个东西in original bert vocab is mapped to vocab_subset's location!!!!!!!
    #         )
    #         filtered_log_probs_list1 = model.filter_logprobs(
    #             original_log_probs_list1, filter_logprob_indices
    #             # everything in original bert vocab is mapped to vocab_subset's location!!!!!!!
    #         )
    #     else:
    #         filtered_log_probs_list = original_log_probs_list
    #         filtered_log_probs_list1 = original_log_probs_list1
    #
    #     label_index_list = []
    #
    #     for sample in samples_b:
    #         obj_label_id = model.get_id(sample["obj_label"])
    #         # MAKE SURE THAT obj_label IS IN VOCABULARIES
    #         if obj_label_id is None:
    #             raise ValueError(
    #                 "object label {} not in model vocabulary".format(
    #                     sample["obj_label"]
    #                 )
    #             )
    #         elif model.vocab[obj_label_id[0]] != sample["obj_label"]:
    #             raise ValueError(
    #                 "object label {} not in model vocabulary".format(
    #                     sample["obj_label"]
    #                 )
    #             )
    #         elif vocab_subset is not None and sample["obj_label"] not in vocab_subset:
    #             raise ValueError(
    #                 "object label {} not in vocab subset".format(sample["obj_label"])
    #             )
    #
    #         label_index_list.append(obj_label_id)
    #
    #
    #     arguments = [
    #         {
    #             "original_log_probs": original_log_probs,
    #             "filtered_log_probs": filtered_log_probs,
    #             "token_ids": token_ids,
    #             "vocab": model.vocab,
    #             "label_index": label_index[0],  # the object's vocab index
    #             "masked_indices": masked_indices,   # the [mask]'s index in a sentence
    #             "interactive": args.interactive,
    #             "index_list": index_list,          # vocab index common in bert and common subset
    #             "sample": sample,  #raw sample, 没有被tokenize或mask过的string
    #             "attention":atten, #以下三个是后面加入的，不太占地方，所以所有task都可以用他们
    #             "sentence":sent,
    #             "model":model
    #         }
    #         for original_log_probs, filtered_log_probs, token_ids, masked_indices, label_index, sample, atten,sent in zip(
    #             original_log_probs_list,
    #             filtered_log_probs_list,
    #             token_ids_list,
    #             masked_indices_list,
    #             label_index_list,
    #             samples_b,
    #             _attentions_bert,
    #             sentences_b
    #         )
    #     ]
    #     arguments1 = [
    #         {
    #             "original_log_probs": original_log_probs1,
    #             "filtered_log_probs": filtered_log_probs1,
    #             "token_ids": token_ids1,
    #             "vocab": model.vocab,
    #             "label_index": label_index[0],  # the object's vocab index
    #             "masked_indices": masked_indices1,  # the [mask]'s index in a sentence
    #             "interactive": args.interactive,
    #             "index_list": index_list,  # vocab index common in bert and common subset
    #             "sample": sample_bm25,
    #             "attention":atten_bm25,       #以下三个是后面加入的，不太占地方，所以所有task都可以用他们
    #             "sentence": sent_bm25,
    #             "model": model
    #         }
    #         for original_log_probs1, filtered_log_probs1, token_ids1, masked_indices1, label_index, sample_bm25, atten_bm25,sent_bm25 in zip(
    #             original_log_probs_list1,
    #             filtered_log_probs_list1,
    #             token_ids_list1,
    #             masked_indices_list1,
    #             label_index_list,
    #             samples_b_bm25,
    #             _attentions_bert_bm25,
    #             sentences_b_bm25
    #         )
    #     ]
    #     res = pool.starmap(run_thread, zip(arguments,arguments1))
    #
    #     for idx, result in enumerate(res):
    #
    #         result_masked_topk, sample_MRR, sample_P, sample_perplexity, msg, situation = result
    #         logger.info("\n" + msg + "\n")
    #
    #         sample = samples_b[idx]
    #
    #         element = {}
    #         element["sample"] = sample
    #         element["uuid"] = sample["uuid"]
    #         element["token_ids"] = token_ids_list[idx]
    #         element["masked_indices"] = masked_indices_list[idx]
    #         element["label_index"] = label_index_list[idx]
    #         element["masked_topk"] = result_masked_topk
    #         element["sample_MRR"] = sample_MRR
    #         element["sample_Precision"] = sample_P
    #         element["sample_perplexity"] = sample_perplexity
    #         element["sample_Precision1"] = result_masked_topk["P_AT_1"]
    #
    #         MRR += sample_MRR
    #         Precision += sample_P
    #         Precision1 += element["sample_Precision1"]
    #
    #         # the judgment of the annotators recording whether they are
    #         # evidence in the sentence that indicates a relation between two entities.
    #         num_yes = 0
    #         num_no = 0
    #
    #         if "judgments" in sample:
    #             # only for Google-RE
    #             for x in sample["judgments"]:
    #                 if x["judgment"] == "yes":
    #                     num_yes += 1
    #                 else:
    #                     num_no += 1
    #             if num_no >= num_yes:
    #                 samples_with_negative_judgement += 1
    #                 element["judgement"] = "negative"
    #                 MRR_negative += sample_MRR
    #                 Precision_negative += sample_P
    #             else:
    #                 samples_with_positive_judgement += 1
    #                 element["judgement"] = "positive"
    #                 MRR_positive += sample_MRR
    #                 Precision_positivie += sample_P
    #
    #         list_of_results.append(element)
    #         final_condition.append(situation)
    #
    # pool.close()
    # pool.join()
    #
    # ############################## modified so doesn't stuck here
    # if (len(list_of_results)==0):
    #     return -1
    # else:
    #     MRR /= len(list_of_results)
    # ########################
    # #####Count1,2,3,4,5#####
    # ########################
    # print("................................")
    # print(final_condition.count(0),final_condition.count(1),final_condition.count(2),final_condition.count(3),final_condition.count(4))
    # print("................................")
    # # Precision
    # Precision /= len(list_of_results)
    # Precision1 /= len(list_of_results)

    # msg = "all_samples: {}\n".format(len(all_samples))
    # msg += "list_of_results: {}\n".format(len(list_of_results))
    # msg += "global MRR: {}\n".format(MRR)
    # msg += "global Precision at 10: {}\n".format(Precision)
    # msg += "global Precision at 1: {}\n".format(Precision1)
    #
    # if samples_with_negative_judgement > 0 and samples_with_positive_judgement > 0:
    #     # Google-RE specific
    #     MRR_negative /= samples_with_negative_judgement
    #     MRR_positive /= samples_with_positive_judgement
    #     Precision_negative /= samples_with_negative_judgement
    #     Precision_positivie /= samples_with_positive_judgement
    #     msg += "samples_with_negative_judgement: {}\n".format(
    #         samples_with_negative_judgement
    #     )
    #     msg += "samples_with_positive_judgement: {}\n".format(
    #         samples_with_positive_judgement
    #     )
    #     msg += "MRR_negative: {}\n".format(MRR_negative)
    #     msg += "MRR_positive: {}\n".format(MRR_positive)
    #     msg += "Precision_negative: {}\n".format(Precision_negative)
    #     msg += "Precision_positivie: {}\n".format(Precision_positivie)
    #
    # logger.info("\n" + msg + "\n")
    # print("\n" + msg + "\n")
    #
    # # dump pickle with the result of the experiment
    # all_results = dict(
    #     list_of_results=list_of_results, global_MRR=MRR, global_P_at_10=Precision
    # )
    # with open("{}/result.pkl".format(log_directory), "wb") as f:
    #     pickle.dump(all_results, f)

    return Precision1

if __name__ == "__main__":
    parser = options.get_eval_KB_completion_parser()
    args = options.parse_args(parser)
    main(args)
