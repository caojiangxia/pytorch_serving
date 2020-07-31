"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from pytorch_serving.Desc.utils import torch_utils, constant
import pytorch_serving.Desc.layers as layers
import pytorch_serving.Desc.submodel as submodel
from pytorch_pretrained_bert import BertModel, BertAdam
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam


class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, batch_num):
        self.opt = opt
        self.model = Extraction(opt)
        self.model.cuda()

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        num_train_optimization_steps = batch_num * (opt['num_epoch'] + 1)
        self.optimizer = BertAdam(optimizer_grouped_parameters, lr=opt['lr'], warmup=0.1,
                                  t_total=num_train_optimization_steps)
        self.bce = nn.BCELoss(reduction='none')

        self.ema = layers.EMA(self.model, opt['ema'])
        self.ema.register()

    def update(self, batch, i):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [Variable(torch.LongTensor(b).cuda()) for b in batch[:4]]
            o_labels = Variable(torch.FloatTensor(batch[4]).cuda())
            mask = Variable(torch.FloatTensor(batch[5]).cuda())

        # step forward
        self.model.train()
        self.optimizer.zero_grad()

        loss_mask = mask.unsqueeze(-1)
        o_probs = self.model(inputs, mask)

        o_probs = o_probs.pow(2)

        o_loss = self.bce(o_probs, o_labels)  # .view(batch_size, seq_len, 2)
        o_loss = 0.5 * torch.sum(o_loss.mul(loss_mask)) / torch.sum(loss_mask)

        loss = o_loss

        # backward
        loss.backward()
        self.optimizer.step()

        self.ema.update()

        loss_val = loss.data.item()
        return loss_val

    def predict_obj_per_instance(self, inputs, mask, user_cuda= None):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            if user_cuda == None:
                inputs = [Variable(torch.LongTensor(b).cuda()) for b in inputs]
                mask = Variable(torch.FloatTensor(mask).cuda())
            else:
                inputs = [Variable(torch.LongTensor(b).cuda(user_cuda)) for b in inputs]
                mask = Variable(torch.FloatTensor(mask).cuda(user_cuda))

        self.model.eval()

        words, distance_to_s, s_start, s_end = inputs
        hidden, sentence_rep = self.model.based_encoder(words)

        o_probs = self.model.o_sublayer(hidden, sentence_rep, distance_to_s, s_start, s_end, mask)

        o_probs = o_probs.pow(2)

        o_probs = o_probs.mul(mask.unsqueeze(-1)).data.cpu().numpy()

        return o_probs

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']


class Extraction(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt):
        super(Extraction, self).__init__()
        self.bert = BertModel.from_pretrained(opt['bert_model'])

        self.o_sublayer = submodel.ObjBaseModel(opt)

        self.opt = opt

    def based_encoder(self, words):
        batch_masks = words.gt(0)

        hidden, sentence_rep = self.bert(words, token_type_ids=None, attention_mask=batch_masks,
                                         output_all_encoded_layers=False)

        return hidden, sentence_rep

    def forward(self, inputs, mask):
        token_ids, distance_to_s, s_start, s_end = inputs  # unpack
        hidden, sentence_rep = self.based_encoder(token_ids)
        o_probs = self.o_sublayer(hidden, sentence_rep, distance_to_s, s_start, s_end, mask)

        return o_probs


