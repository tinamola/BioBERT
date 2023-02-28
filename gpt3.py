# import openai
from lama.modules import build_model_by_name
import lama.utils as utils
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
from tqdm import tqdm
from random import shuffle
import os
import json
import lama.modules.base_connector as base
from pprint import pprint
import logging
from multiprocessing.pool import ThreadPool
import multiprocessing
import lama.evaluation_metrics as metrics
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import re
# os.environ["OPENAI_API_KEY"] = "your key here"
# openai.api_key = os.getenv("OPENAI_API_KEY")


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

def run_thread(arguments):

    msg = ""

    # 1. compute the ranking metrics on the filtered log_probs tensor
    sample_MRR, sample_P, experiment_result, return_msg = metrics.get_ranking(
        arguments["filtered_log_probs"],
        arguments["masked_indices"],
        arguments["vocab"],
        label_index=arguments["label_index"],
        index_list=arguments["index_list"],
        print_generation=arguments["interactive"],
        topk=10000,
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

    return experiment_result, sample_MRR, sample_P, sample_perplexity, msg


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

    # deal with vocab subset
    vocab_subset = None
    index_list = None

    if args.common_vocab_filename is not None:
        vocab_subset = load_vocab(args.common_vocab_filename)

        filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(
            vocab_subset
        )  #this two variables are the same, filter_logprob_indices contain common vocab indices of bert vocab and vocab subset
        # though doesn't have all the indices in common vocab list. inverse vocab is made of bert vocab.



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

    # spearman rank correlation
    # overlap at 1
    if args.use_negated_probes:
        Spearman = 0.0
        Overlap = 0.0
        num_valid_negation = 0.0

    data = load_file(args.dataset_filename)

    print(len(data))

    all_samples, ret_msg = filter_samples(
        model, data, vocab_subset, args.max_sentence_length, args.template
    )


    #logger.info("\n" + ret_msg + "\n")

    print(len(all_samples))

    # if template is active (1) use a single example for (sub,obj) and (2) ...
    if args.template and args.template != "":
        facts = []
        for sample in all_samples:
            sub = sample["sub_label"]
            obj = sample["obj_label"]
            if (sub, obj) not in facts:
                facts.append((sub, obj))
        #local_msg = "distinct template facts: {}".format(len(facts))
        #logger.info("\n" + local_msg + "\n")
        #print(local_msg)
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
    # def generate_context_func(sample,relation):
    #     thisdict = {
    #         "place_of_death": "place of death of",
    #         "place_of_birth": "place of birth of",
    #         "date_of_birth": "date of birth of",
    #         "P19":"place of birth of",
    #         "P20":"place of death of",
    #         "P279": "parent class of",
    #         "P37": "official language of",
    #         "P413": "position played on team / speciality of",
    #         #"P166": "award received",
    #         "P449": "aired place of",
    #         #"P69": "educated at",
    #         "P47": "place that shared border with",
    #         "P138": "eponymous of",  #<----changed from "x named after y"
    #         "P364": "original language of film or TV show of",
    #         #"P54": "member of sports team of",
    #         "P463":"organization or club joined by", #<--- is a member of
    #         "P101":"field of work of",
    #         #"P1923": "participated team of",
    #         "P106": "occupation of",
    #         "P527": "component of",
    #         #"P102": "political party of",
    #         "P530": "country that maintains diplomatic relations with",
    #         "P27": "country of citizenship of",
    #         "P407": "language of work or name of",
    #         "P30": "continent of",
    #         "P176": "manufacturer of",
    #         "P178": "developer of",
    #         "P1376": "country, state, department, canton or other administrative division of",
    #         "P131": "location of",
    #         "P1412": "languages spoken, written or signed of",
    #         "P108": "employer of",
    #         "P136": "music genre played by",
    #         "P17": "location of",
    #         "P39": "position held by",
    #         "P264": "record label of",
    #         "P276": "location of",
    #         "P937": "work place of",
    #         "P140": "religion of",
    #         "P1303": "musical instrument played by",
    #         "P127": "owner of",
    #         "P103": "native language of",
    #         "P190": "twin city of",
    #         "P1001": "jurisdiction of",
    #         "P31": "generalization of", # <---- instance of
    #         "P495": "country of origin of",
    #         "P159": "headquarter location of",
    #         "P36": "capital of",
    #         "P740": "location of formation of",
    #         "P361": "object that contains",  # <--- x is part of y
    #         "HasSubevent":"simultaneous event of",
    #         "IsA": "category of",
    #         "CausesDesire": "cause and desire of",
    #         "AtLocation": "location of",
    #         "CapableOf": "capability of",
    #         "PartOf": "object that contains",
    #         "HasA": "constituent of",
    #         "UsedFor": "usage of",
    #         "Causes": "consequence of",
    #         "HasProperty": "property of",
    #         "MadeOf": "component of",
    #         "HasPrerequisite": "requirement of",
    #         "MotivatedByGoal": "motivation of",
    #         "NotDesires": "undesirable thing of",
    #         "Desires": "desires of",
    #         "ReceivesAction": "doable actions upon"
    #     }
    #     if relation != "test":
    #         if "test" in relation:
    #             relation=sample['pred']
    #         #不reveal的版本
    #         temp_data="Generate some general knowledge and some knowledge about the "+\
    #                   thisdict[relation]+" the concepts in the input. Produce a long paragraph."+\
    #                   "\nInput: " + sample["sub_label"]+"\nKnowledge:"
    #         #reveal的版本
    #         # temp_data = sample['masked_sentences'][0].replace("[MASK]", sample['obj_label']) + " Produce a long paragraph and explain this."
    #     else:
    #         replace_word = "what"
    #         temp = sample['masked_sentences'][0].replace("[MASK]", replace_word)
    #         temp_data = temp[:len(temp) - 1] + "?" + " Produce a long paragraph and explain this."
    #
    #     response = openai.Completion.create(model="text-davinci-002",
    #                                         prompt=temp_data,
    #                                         temperature=0,
    #                                         max_tokens=250)
    #     sample['masked_sentences'].append(response.choices[0].text.strip('\n'))
    #
    #     return sample

    # code when we generate query with gpt3
    # with open('gpt3_produced_context/%s.json' % args.relation, "a", encoding='utf-8') as fp:
    #     for sample in all_samples:
    #         new_sample=generate_context_func(sample,args.relation)
    #         json.dump(new_sample, fp)
    #         fp.write("\n")
    def mask_query_func(example):
        object_token = re.compile(re.escape(example['obj_label']), re.IGNORECASE)
        example['masked_sentences'][1] = object_token.sub("", " ".join(example['masked_sentences'][1].split()[:450]))
        # example['masked_sentences'][1] = " ".join(example['masked_sentences'][1].split()[:450])
        return example
    #haven't mask the answer yet
    all_samples1 = []
    for line in open('produced_context_gpt3/%s.json' % args.relation, 'r'):
        all_samples1.append(json.loads(line))
    all_samples1 = list(map(mask_query_func, all_samples1))

    if len(all_samples1)>len(all_samples):
        print("what?????????")
        tmp_all_samples1=all_samples1
        all_samples1 = []
        for i in range(len(all_samples)):
            while all_samples[i]['masked_sentences'][0]!=tmp_all_samples1[i]['masked_sentences'][0]:
                del tmp_all_samples1[i]
            all_samples1.append(tmp_all_samples1[i])

    samples_batches, sentences_batches, ret_msg = batchify(all_samples1, args.batch_size)  #[[32],[32]] first is whole sample, second is masksed sentence

    # ThreadPool
    num_threads = args.threads
    if num_threads <= 0:
        # use all available threads
        num_threads = multiprocessing.cpu_count()
    pool = ThreadPool(num_threads)
    list_of_results = []

    for i in tqdm(range(len(samples_batches))):

        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]

        (
            original_log_probs_list,
            token_ids_list,
            masked_indices_list,
        ) = model.get_batch_generation(sentences_b, output_atten=False)
        # print(original_log_probs_list)
        if vocab_subset is not None:
            # filter log_probs
            filtered_log_probs_list = model.filter_logprobs(
                original_log_probs_list, filter_logprob_indices   #everything in original bert vocab is mapped to vocab_subset's location!!!!!!!
            )
        else:
            filtered_log_probs_list = original_log_probs_list

        label_index_list = []
        for sample in samples_b:
            obj_label_id = model.get_id(sample["obj_label"])

            # MAKE SURE THAT obj_label IS IN VOCABULARIES
            if obj_label_id is None:
                raise ValueError(
                    "object label {} not in model vocabulary".format(
                        sample["obj_label"]
                    )
                )
            elif model.vocab[obj_label_id[0]] != sample["obj_label"]:
                raise ValueError(
                    "object label {} not in model vocabulary".format(
                        sample["obj_label"]
                    )
                )
            elif vocab_subset is not None and sample["obj_label"] not in vocab_subset:
                raise ValueError(
                    "object label {} not in vocab subset".format(sample["obj_label"])
                )

            label_index_list.append(obj_label_id)

        arguments = [
            {
                "original_log_probs": original_log_probs,
                "filtered_log_probs": filtered_log_probs,
                "token_ids": token_ids,
                "vocab": model.vocab,
                "label_index": label_index[0],  # the object's vocab index
                "masked_indices": masked_indices,   # the [mask]'s index in a sentence
                "interactive": args.interactive,
                "index_list": index_list,          # vocab index common in bert and common subset
                "sample": sample,
            }
            for original_log_probs, filtered_log_probs, token_ids, masked_indices, label_index, sample in zip(
                original_log_probs_list,
                filtered_log_probs_list,
                token_ids_list,
                masked_indices_list,
                label_index_list,
                samples_b,
            )
        ]

        res = pool.map(run_thread, arguments)


        for idx, result in enumerate(res):

            result_masked_topk, sample_MRR, sample_P, sample_perplexity, msg = result

            #logger.info("\n" + msg + "\n")

            sample = samples_b[idx]

            element = {}
            element["sample"] = sample
            element["uuid"] = sample["uuid"]
            element["token_ids"] = token_ids_list[idx]
            element["masked_indices"] = masked_indices_list[idx]
            element["label_index"] = label_index_list[idx]
            element["masked_topk"] = result_masked_topk
            element["sample_MRR"] = sample_MRR
            element["sample_Precision"] = sample_P
            element["sample_perplexity"] = sample_perplexity
            element["sample_Precision1"] = result_masked_topk["P_AT_1"]

            MRR += sample_MRR
            Precision += sample_P
            Precision1 += element["sample_Precision1"]

            # the judgment of the annotators recording whether they are
            # evidence in the sentence that indicates a relation between two entities.
            num_yes = 0
            num_no = 0

            if "judgments" in sample:
                # only for Google-RE
                for x in sample["judgments"]:
                    if x["judgment"] == "yes":
                        num_yes += 1
                    else:
                        num_no += 1
                if num_no >= num_yes:
                    samples_with_negative_judgement += 1
                    element["judgement"] = "negative"
                    MRR_negative += sample_MRR
                    Precision_negative += sample_P
                else:
                    samples_with_positive_judgement += 1
                    element["judgement"] = "positive"
                    MRR_positive += sample_MRR
                    Precision_positivie += sample_P

            list_of_results.append(element)

    pool.close()
    pool.join()

    # stats
    # Mean reciprocal rank
    ############################## modified so doesn't stuck here
    if (len(list_of_results)==0):
        return -1
    else:
        MRR /= len(list_of_results)

    # Precision
    Precision /= len(list_of_results)
    Precision1 /= len(list_of_results)

    msg = "all_samples: {}\n".format(len(all_samples))
    msg += "list_of_results: {}\n".format(len(list_of_results))
    msg += "global MRR: {}\n".format(MRR)
    msg += "global Precision at 10: {}\n".format(Precision)
    msg += "global Precision at 1: {}\n".format(Precision1)

    if args.use_negated_probes:
        Overlap /= num_valid_negation
        Spearman /= num_valid_negation
        msg += "\n"
        msg += "results negation:\n"
        msg += "all_negated_samples: {}\n".format(int(num_valid_negation))
        msg += "global spearman rank affirmative/negated: {}\n".format(Spearman)
        msg += "global overlap at 1 affirmative/negated: {}\n".format(Overlap)

    if samples_with_negative_judgement > 0 and samples_with_positive_judgement > 0:
        # Google-RE specific
        MRR_negative /= samples_with_negative_judgement
        MRR_positive /= samples_with_positive_judgement
        Precision_negative /= samples_with_negative_judgement
        Precision_positivie /= samples_with_positive_judgement
        msg += "samples_with_negative_judgement: {}\n".format(
            samples_with_negative_judgement
        )
        msg += "samples_with_positive_judgement: {}\n".format(
            samples_with_positive_judgement
        )
        msg += "MRR_negative: {}\n".format(MRR_negative)
        msg += "MRR_positive: {}\n".format(MRR_positive)
        msg += "Precision_negative: {}\n".format(Precision_negative)
        msg += "Precision_positivie: {}\n".format(Precision_positivie)

    #logger.info("\n" + msg + "\n")
    #print("\n" + msg + "\n")

    return Precision1


