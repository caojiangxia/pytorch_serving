from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open
import numpy as np
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
import torch
import requests

logger = logging.getLogger(__name__)


class MyTest(torch.nn.Module):
    def __init__(self):
        super(MyTest, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('./saved_models/', num_labels=2).cuda()

    def forward(self, input_ids, segment_ids, input_mask):
        logits = self.model(input_ids, segment_ids, input_mask, labels=None)
        # preds.append(logits.detach().cpu().numpy())
        # preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        # output = tuple(preds.tolist())
        # print(output)
        return logits


VOCAB_NAME = 'vocab.txt'


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, vocab_file, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        logger.info("loading vocabulary file {}".format(vocab_file))
        kwargs['max_len'] = 512
        # Instantiate tokenizer.
        tokenizer = cls(vocab_file, *inputs, **kwargs)
        return tokenizer


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_text_to_features(text, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    max_seq_length = 256

    tokens = tokenizer.tokenize(text)

    features = []

    tokens = tokens[:(max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features.append(
        InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids))
    return features


def infer(text, tokenizer):
    features = convert_text_to_features(text, tokenizer)

    # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
    # all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
    # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).cuda()

    all_input_ids = [f.input_ids for f in features]
    all_input_mask = [f.input_mask for f in features]
    all_segment_ids = [f.segment_ids for f in features]

    data = {
        "input_ids": all_input_ids,
        "segment_ids": all_segment_ids,
        "input_mask": all_input_mask,
    }
    req = {
        "model_name": "Bowen",
        "version": "3",
        "data": data
    }
    endpoint = "http://192.168.124.51:8066/api/inference"

    result = requests.post(endpoint, json=req).json()

    print(result)



if __name__ == '__main__':
    # test_model = MyTest()
    tokenizer = BertTokenizer.from_pretrained('./')
    text = '和国家机关各部委，各人民团体：    加强和完善机构编制管理，是加强党的执政能力建设和国家政权建设的重要基础性工作。党的十六大以来，各地区各部门按照全面落实科学发展观、构建社会主义和谐社会的要求，采取多种措施加强和完善机构编制管理，做了大量工作。但是，目前一些地方和部门仍然存在擅自增设机构、在机关使用事业编制、超编制配备人员、超职数配备领导干部等问题，上级业务部门干预下级机构编制的情况也时有发生。这些问题在一定程度上影响了党政机关自身建设，影响了党和政府的形象，不利于加快深化行政管理体制改革。为严格执行中央有关机构编制工作的方针政策，进一步加强和完善机构编制管理，严格控制机构编制，经党中央、国务院领导同志同意，现就有关问题通知如下。    一、进一步严格行政机构编制管理    行政编制的审批权限在中央，中央机构编制部门审核中央和国家机关各部门、有关群众团体机关的行政编制和地方各级行政编制总额，各地区各部门不得在中央批准的行政编制总额外越权审批行政编制或自行确定用于党政机关的编制。各地区各部门可根据工作需要调整行政编制，但必须在批准的总额内进行；地方不同层级之间行政编制的调整由省级机构编制部门报中央机构编制部门审批。各地区各部门要严格按照规定使用行政编制，不得擅自改变行政编制的使用范围。    中央和国家机关各部门、有关群众团体机关和各省、自治区、直辖市党政机关副厅（局）级以上机构设置的审批权限在中央。各地区各部门要严格按照规定设置机构，不得擅自设立行政机构。    严格控制行政机构设置和行政编制。工作任务增加的部门，所需机构编制主要通过内部整合和调剂解决。各地区各部门要加快职能转变，推进体制改革，改进管理方式，整合资源，挖掘潜力，从源头上控制机构编制增长。确需增加机构编制的，严格按程序办理。各地在推行乡镇机构改革试点工作中，要按照中央要求，确保乡镇机构编制（包括行政编制、事业编制）和财政供养人员在“十一五”期间只减不增。    二、加强和规范事业单位机构编制管理    强化省级机构编制部门对本地区各级事业单位机构编制的管理职责。各省、自治区、直辖市机构编制部门统一制定本省、自治区、直辖市事业单位机构编制管理办法，上级机构编制部门要对下级事业编制总量和结构进行管理，根据各地实际适当上收事业单位机构编制审批权限。乡镇事业编制总量的调整，由县级机构编制部门报上一级机构编制部门审核，省级机构编制部门审批。根据实际情况，可探索试行将县（市、区）党委、政府及其工作部门所属事业单位机构编制由上一级机构编制部门审批，具体实施办法由省级机构编制部门结合本地实际研究制定。中央机构编制部门要加强对地方事业单位机构编制管理工作的宏观指导，对各省、自治区、直辖市事业编制的总量和结构进行调控。    除法律法规和党中央、国务院有关文件规定外，各地区各部门不得将行政职能转由事业单位承担，不再批准设立承担行政职能的事业单位和从事生产经营活动的事业单位。公益服务事业发展需要增加事业单位机构编制的，应在现有总量中进行调整；确需增加的，要严格按程序审批。    严格控制事业编制的使用范围。事业编制只能用于事业单位，不得把事业编制用于党政机关，不得把事业编制与行政编制混合使用。    三、坚持机构编制的集中统一管理    严格执行机构编制审批程序和制度。凡涉及职能调整，机构、编制和领导职数增减的，统一由机构编制部门审核，按程序报同级机构编制委员会或党委、政府审批。其他任何部门和单位无权决定机构编制事项。    除专项机构编制法律法规外，各地区各部门拟订法规或法律草案不得就机构编制事项作出具体规定；在制定规范性文件和起草领导同志讲话时，涉及机构编制事项的，必须征求机构编制部门意见。确需增加和调整机构编制的，必须按规定权限和程序由机构编制部门专项办理。    上级业务部门不得干预下级部门的机构编制事项，不得要求下级部门设立与其业务对口的机构或提高机构规格，不得要求为其业务对口的机构配备或增加编制。各业务部门制定的行业标准，不得作为审批机构编制的依据。    四、加快机构编制管理法制化规范化建设    进一步加强机构编制管理的法制建设，完善国务院机构设置和编制管理条例，认真贯彻落实地方各级人民政府机构设置和编制管理条例；制定部门职责分工协调办法，加大职责分工协调力度；完善各级党政部门主要职责、内设机构和人员编制规定，在明确各部门职能的同时必须逐项明确其应承担的责任，并增强其规范性、可操作性和严肃性。    不断探索和完善行之有效的机构编制管理办法。试行机构编制实名制管理制度，凡按规定批准成立的党政机关和事业单位都要实行定编定员，确保具体机构设置与按规定审批的机构相一致、实有人员与批准的编制和领导职数相对应。加强机构编制标准的研究制定工作，根据工作需要对现行标准进行修改完善，制定更加符合实际的有关机构编制标准。逐步建立机构编制公开制度，对不涉及国家秘密的机构编制及其执行情况，要通过有效形式向本单位工作人员或社会公开，接受监督。建立健全机构编制管理工作考核和评估制度。完善机构编制统计和报告制度，确保统计数据准确、及时、全面。    五、建立健全机构编制部门与相关部门间的协调配合和制约机制    机构编制部门与相关部门要密切配合，各负其责，形成合力。强化机构编制管理与组织（人事）管理、财政管理等的综合约束机制。只有在机构编制部门审核同意设置的机构和核批的编制范围内，组织（人事）部门才能配备人员和核定工资，财政部门才能列入政府预算并核拨经费，银行才能开设账户并发放工资。    对超编进入的人员，组织（人事）部门不得办理录用、聘用（任）、调任手续、核定工资，财政部门不得纳入统发工资范围，公安等部门不得办理户口迁移等手续。对擅自增设的机构，财政部门不得纳入政府预算范围、核拨经费，银行不得开设账户。    各地区各部门要严格执行机构编制规定，不准超编进人，不准擅自设立机构和提高机构规格，不准超职数、超机构规格配备领导干部，不准以虚报人员等方式占用行政和事业编制、冒领财政资金。    六、加大机构编制监督检查力度    机构编制部门要加强对机构编制有关规定执行情况的监督检查。认真贯彻实施机构编制监督检查有关规定，进一步建立健全对违反机构编制纪律问题的举报受理制度，加强“12310”机构编制监督举报电话和群众来信来访的受理工作。纪检监察机关应当依纪依法履行职责，检查执行机构编制管理规定中存在的问题，查处机构编制违纪违法行为。各级机构编制部门、纪检监察机关等要密切配合，建立健全机构编制监督检查协调机制，共同维护机构编制纪律的严肃性。    加大对违反机构编制纪律问题的查处力度，严格责任追究。对违反机构编制纪律的单位，要责令其限期改正；逾期不改的，予以纠正并进行通报批评。对于需要追究相关人员责任的，由纪检监察机关或主管部门按规定给予党纪政纪处分；涉嫌犯罪的，移送司法机关依法处理。对上级业务部门以下发文件、召开会议、批资金、上项目、搞评比、打招呼等方式干预下级部门机构设置、职能配置和人员编制配备的，要严肃查处。    七、加强对机构编制工作的领导    各级党委和政府要适应深化行政管理体制改革的新形势，加强对机构编制工作的领导，严格控制机构编制，及时掌握工作情况，研究解决重要问题，切实抓好中央有关机构编制工作方针政策的贯彻落实。要加强机构编制部门领导班子建设和队伍建设，健全领导体制和工作体系，大力支持机构编制部门的工作，不断提高机构编制管理的能力和水平。    各级机构编制部门要以科学发展观为统领，坚持解放思想、实事求是、与时俱进，努力研究新情况新问题，不断创新和完善工作机制，进一步强化为经济社会发展服务、为基层群众服务的意识，坚持原则，严格管理，秉公办事，充分发挥把关、协调、监督等职能作用，促进机构编制管理工作健康发展。中共中央办公'
    infer(text, tokenizer)



