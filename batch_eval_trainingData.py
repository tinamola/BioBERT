from lama.modules import build_model_by_name
import lama.utils as utils
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
from tqdm import tqdm
from random import shuffle
import os
import json
import spacy
import lama.modules.base_connector as base
from pprint import pprint
import logging.config
import logging
import pickle
from multiprocessing.pool import ThreadPool
import multiprocessing
import lama.evaluation_metrics as metrics
import time, sys
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import re
from datasets import load_dataset
import elasticsearch
from transformers import BertTokenizer, BertForPreTraining
new=load_dataset('wikipedia','20220301.en',cache_dir='D:\OneDrive\Desktop\WikipediaCorpus')
new=new['train']
es_client=elasticsearch.Elasticsearch('http://localhost:9200',http_auth=("elastic","rTmD2wr3a_-qxvQq+QRL"))
new.load_elasticsearch_index('text',es_client=es_client,es_index_name="wikipedia")
num_samples=10

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]

def batchify(data, batch_size):
    msg = ""
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    c = 0

    # sort to group together sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
    ):
        masked_sentences = sample["masked_sentences"]
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches, msg

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

    # deal with vocab subset
    vocab_subset = None
    index_list = None
    if args.common_vocab_filename is not None:
        vocab_subset = load_vocab(args.common_vocab_filename)

        filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(
            vocab_subset
        )  #this two variables are the same, filter_logprob_indices contain common vocab indices of bert vocab and vocab subset

    data = load_file(args.dataset_filename)

    print(len(data))

    all_samples, ret_msg = filter_samples(
        model, data, vocab_subset, args.max_sentence_length, args.template
    )

    print(len(all_samples))

    # if template is active (1) use a single example for (sub,obj) and (2) ...
    if args.template and args.template != "":
        facts = []
        for sample in all_samples:
            sub = sample["sub_label"]
            obj = sample["obj_label"]
            if (sub, obj) not in facts:
                facts.append((sub, obj))

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

            all_samples.append(sample)

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1
    """
    remove the [MASK] in the query when getting the paragraph
    """
    def myfunc(sample):
        object_token = re.compile(re.escape(sample['obj_label']), re.IGNORECASE)
        query=sample['masked_sentences'][0].replace("[MASK]","")
        scores,retrieved=new.get_nearest_examples("text",query,k=num_samples)
        #not filtering object word
        # new_retrieved = [" ".join(k) for k in [j.split() for j in [retrieved['text'][i] for i in range(num_samples)]] if len(k) < 450]
        # filtering object word
        new_retrieved=[object_token.sub("",l) for l in [" ".join(k) for k in [j.split() for j in [retrieved['text'][i] for i in range(num_samples)]] if len(k) < 450]]
        if len(new_retrieved) > 0:
            return new_retrieved[0]
        else:
            new_retrieved=object_token.sub(""," ".join(retrieved['text'][0].split()[:450]))
            # new_retrieved = " ".join(retrieved['text'][0].split()[:450])
        return new_retrieved

    all_samples=list(map(myfunc,all_samples))
    filename=args.relation
    with open('trainData/%s.txt' % filename, "w", encoding='utf-8') as output:
        for row in all_samples:
                output.write('%s\n' % row)
    return 0

