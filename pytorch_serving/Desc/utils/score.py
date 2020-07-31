import numpy as np
from tqdm import tqdm
from pytorch_serving.Desc.utils import loader
import json
# from difflib import SequenceMatcher
# from scipy.optimize import linear_sum_assignment
import numpy as np
from pytorch_serving.Desc.utils import constant
from tqdm import tqdm


def extract_entities(probs, threshold, threshold2):
    subject_starts = np.where(probs[:, 0] > threshold)[0]
    subject_ends = np.where(probs[:, 1] > threshold)[0]
    # print(subject_starts, subject_ends)
    subjects = {}
    for i in subject_starts:
        j = subject_ends[subject_ends >= i]
        if len(j) > 0:
            for je in j:
                if probs[i, 0] * probs[je, 1] > threshold2:
                    subjects[(i, je)] = [probs[i, 0] * probs[je, 1], 1]
    # print(probs)
    # print(subjects)
    # print('--------')
    # print(subjects)
    for key in subjects:
        if subjects[key][1] == 1:
            # print('***************')
            # print(key)
            chongdie_set = set()
            xiangjiao_set = set()
            for another in subjects:
                if key == another:
                    continue
                if subjects[another][1] == 1:
                    if another[0] >= key[0] and another[1] <= key[1]:
                        chongdie_set.add(another)
                    elif another[0] <= key[0] and another[1] >= key[1]:
                        continue
                    elif set(range(key[0], key[1] + 1)) & set(range(another[0], another[1] + 1)):
                        xiangjiao_set.add(another)
            chongdie_prob = sum([subjects[another][0] for another in chongdie_set])
            if chongdie_prob > subjects[key][0]:
                subjects[key][1] = 0
            else:
                for another in chongdie_set:
                    subjects[another][1] = 0
                flag = True
                for another in xiangjiao_set:
                    if subjects[another][0] > subjects[key][0]:
                        #     subjects[another][1] = 0
                        # else:
                        subjects[key][1] = 0
                        flag = False
                        break
                if flag:
                    for another in xiangjiao_set:
                        subjects[another][1] = 0

    return [(key, subjects[key][0]) for key in subjects if subjects[key][1] == 1]


def extract_items(sentence_list, subject, tokenizer, model,user_cuda = None):
    results = []
    s = tokenizer.encode(subject)[0][1:-1]
    if len(s) < 1:
        return []
    token_idss = []
    masks = []
    s_starts = []
    s_ends = []
    distance_to_ss = []
    mappings = []

    for text in sentence_list:
        tokens = tokenizer.tokenize(text, max_length=128)
        mapping = tokenizer.rematch(text, tokens)
        token_ids, segment_ids = tokenizer.encode(text, max_length=128)
        mask = list(np.ones(len(token_ids)))
        mask[0] = 0;
        mask[-1] = 0
        s_idx = loader.search(s, token_ids)
        if s_idx == -1:
            continue
        s_start, s_end = s_idx, s_idx + len(s) - 1
        distance_to_s = loader.get_positions(s_start, s_end, len(token_ids))
        token_idss.append(token_ids)
        masks.append(mask)
        s_starts.append(s_start)
        s_ends.append(s_end)
        distance_to_ss.append(distance_to_s)
        mappings.append(mapping)

    _t = np.array(loader.seq_padding(token_idss, 128))
    mask = np.array(loader.seq_padding(masks, 128))

    distance_to_s = np.array(loader.seq_padding(distance_to_ss, 128))
    s_start = np.array(s_starts)
    s_end = np.array(s_ends)

    inputs = _t, distance_to_s, s_start, s_end

    o_probs = model.predict_obj_per_instance(inputs, mask,user_cuda)

    # print(o_probs)
    # print(extract_entities(o_probs, 0.1, 0))

    for idx in range(o_probs.shape[0]):
        objects = extract_entities(o_probs[idx], 0.2, 0)

        for object, score in objects:
            start_idx = mappings[idx][object[0]][0]
            end_idx = mappings[idx][object[1]][-1] + 1
            results.append(
                {
                    "context": sentence_list[idx],
                    "text": subject,
                    "offset": [start_idx, end_idx],
                    "object": sentence_list[idx][start_idx:end_idx],
                    "score": str(round(score, 2))
                }
            )

    # [{"text":--, "context":--, "offset":--, "score":---}, ...]

    # print(spoes)
    return results


def evaluate(data, tokenizer, model, opt):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10
    P, R, F1 = [], [], []

    results = []
    for i, d in enumerate(iter(data)):
        predication = extract_items(d['text'], d['subject'], tokenizer, model, opt)
        official = set(d['object_list'])
        results.append(
            {'text': d['text'], 'subject': d['subject'], 'predict': list(predication), 'truth': list(official)})
        official_A += len(predication & official)
        official_B += len(predication)
        official_C += len(official)
    return official_A / official_B, official_A / official_C, 2 * official_A / (official_B + official_C), results


def infer(data, tokenizer, model, opt):
    results = []
    for i, d in tqdm(enumerate(iter(data))):
        text = d['text']
        predication = extract_items(d['text'], d['subject'], tokenizer, model, opt)
        results.append({'text': text, 'predict': list(predication), 'truth': d['object_list']})
    return results













