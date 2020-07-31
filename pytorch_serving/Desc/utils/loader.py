import json
import numpy as np
import random
from random import choice
from tqdm import tqdm
import collections
from pytorch_serving.Desc.utils import constant
from pytorch_pretrained_bert import BertTokenizer

global num


def seq_padding(X, max_len):
    # L = [len(x) for x in X]
    # ML =  config.MAX_LEN #max(L)
    # ML = max(L)
    return [x + [0] * (max_len - len(x)) for x in X]


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
           list(range(1, length - end_idx))


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def random_generate(d):
    d1 = {}
    d1['text'] = d['text']
    subject_start = random.randint(0, len(d1['text']) - 5)
    subject_end = subject_start + random.randint(0, 4) + 1
    subject = d1['text'][subject_start:subject_end]
    # if random.random() > 0.7:
    #     d['subject'] = d['object_list'][0]
    #     d['object_list'] = []
    d1['subject'] = subject
    d1['object_list'] = []
    return d1, d


class DataLoader(object):
    def __init__(self, data, batch_size, max_len, tokenizer, evaluation=False):

        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer

        data = self.preprocess(data)
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)
        self.data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def preprocess(self, data):
        processed = []
        discard_num = 0
        for idd, d in enumerate(data):
            # if idd > 1000:
            #     break
            # for d in random_generate(dd):
            # d = random_generate(d)
            spoes = {}
            text = d['text']
            token_ids, segment_ids = self.tokenizer.encode(
                text, max_length=self.max_len
            )
            mask = list(np.ones(len(token_ids)))
            mask[0] = 0;
            mask[-1] = 0
            s = self.tokenizer.encode(d['subject'])[0][1:-1]
            if len(s) < 1:
                continue
            s_idx = search(s, token_ids)
            if s_idx == -1:
                continue

            s_start, s_end = s_idx, s_idx + len(s) - 1
            # print(d['subject'], s_start, s_end)
            o_list = []
            for o in d['object_list']:
                o = self.tokenizer.encode(o)[0][1:-1]
                if len(o) < 1:
                    continue
                o_idx = search(o, token_ids)
                if o_idx != -1:
                    o = (o_idx, o_idx + len(o) - 1)
                    o_list.append(o)

            # if o_list:
            o_labels = np.zeros((self.max_len, 2))
            distance_to_s = get_positions(s_start, s_end, len(token_ids))

            for o_start, o_end in o_list:
                o_labels[o_start, 0] = 1
                o_labels[o_end, 1] = 1

            processed += [(token_ids, s_start, s_end, o_labels, distance_to_s, mask)]
        print(discard_num)
        return processed

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 6

        # orig_idx = lens
        token_ids = np.array(seq_padding(batch[0], self.max_len))
        s_start, s_end = np.array(batch[1]), np.array(batch[2])
        o_labels = np.array(batch[3])
        distance_to_s = np.array(seq_padding(batch[4], self.max_len))
        mask = np.array(seq_padding(batch[5], self.max_len))

        # print(token_ids, s_start, s_end, o_labels)

        return (token_ids, distance_to_s, s_start, s_end, o_labels, mask)

# if __name__ == '__main__':
# data = json.load(open('../data/dataset' + '/train_me.json')) + json.load(open('../data/dataset' + '/dev_me.json'))
# max_len = 0
# for d in data:
#     text = d['text']
#     if max_len < len(text):
#         max_len =  len(text)
# print(max_len)



