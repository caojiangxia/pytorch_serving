import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from pytorch_serving.Desc.utils import torch_utils


# class Attention(nn.Module):
#     """
#     A position-augmented attention layer where the attention weight is
#     a = T' . tanh(Ux + Vq + Wf)
#     where x is the input, q is the query, and f is additional position features.
#     """

#     def __init__(self, input_size, attn_size):
#         super(Attention, self).__init__()
#         self.input_size = input_size
#         self.attn_size = attn_size
#         self.ulinear = nn.Linear(input_size, attn_size)
#         self.tlinear = nn.Linear(attn_size, 1)
#         self.init_weights()

#     def init_weights(self):
#         nn.init.xavier_uniform_(self.ulinear.weight)
#         nn.init.xavier_uniform_(self.tlinear.weight)
#         nn.init.uniform_(self.ulinear.bias)
#         self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning

#     def forward(self, x, x_mask):
#         """
#         x : batch_size * seq_len * input_size
#         q : batch_size * query_size
#         f : batch_size * seq_len * feature_size
#         """
#         batch_size, seq_len, _ = x.size()

#         x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
#             batch_size, seq_len, self.attn_size)
#         scores = self.tlinear(F.tanh(x_proj).view(-1, self.attn_size)).view(
#             batch_size, seq_len)

#         # mask padding
#         x_mask = x_mask < 1
#         scores.data.masked_fill_(x_mask.data, -float('inf'))
#         weights = F.softmax(scores, dim=-1)
#         # weighted average input vectors
#         outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
#         return outputs


class DGCNN(nn.Module):
    def __init__(self, channels, dilation):
        super(DGCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                              dilation=dilation)
        self.conv_gate = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                   dilation=dilation)
        # self.dropout = nn.Dropout(0.1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.uniform_(self.conv.bias)
        nn.init.xavier_uniform_(self.conv_gate.weight)
        nn.init.uniform_(self.conv_gate.bias)

    def forward(self, inputs, mask):
        inputs = torch.mul(inputs, mask)

        inputs = inputs.permute(0, 2, 1)

        gate = torch.sigmoid(self.conv_gate(inputs))
        gated_outputs = torch.mul(gate, self.conv(inputs))
        # outputs = gated_outputs + self.trans(inputs.squeeze(1))
        gated_inputs = torch.mul(1 - gate, inputs)
        outputs = gated_outputs + gated_inputs

        return outputs.permute(0, 2, 1)


class MaskedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MaskedCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    #     self.init_weights()

    # def init_weights(self):
    #     nn.init.xavier_uniform_(self.conv.weight)
    #     nn.init.uniform_(self.conv.bias)

    def forward(self, inputs, mask):
        inputs = torch.mul(inputs, mask)
        inputs = inputs.permute(0, 2, 1)
        outputs = F.relu(self.conv(inputs))

        return outputs.permute(0, 2, 1)


class ConvFor1D(nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, out_dim, step=1, dropout=0.1):
        super(ConvFor1D, self).__init__()
        """
        一维卷积
        :param num_layer:卷积的层数
        :param in_channels: 输入的通道数量。输入数据的维度应该为 B * in_channels * H * W
        :param out_channels: 输出的通道数量，输出的维度是：输入数据的维度应该为 B * out_channels * H * W，可以在最终的输出拼接后面两个维度
        :param out_dim: 最终的输出维度
        :param step: 卷积的步长， 默认为1
        :param dropout:
        """
        assert (
                           in_channels - out_channels) % num_layer == 0, "(in_channels - out_channels) % num_layer should be zero, now % d " % (
                (in_channels - out_channels) % num_layer)
        self.in_channel = in_channels
        self.out_channel = out_channels
        per_layer_plus = int((in_channels - out_channels) / num_layer)
        self.convs = nn.Sequential(*[
            nn.Conv2d(in_channels - per_layer_plus * i, out_channels + per_layer_plus * (num_layer - i - 1), [1, 1],
                      stride=step, padding=0, dilation=1,
                      bias=True, padding_mode='zeros') for i in range(num_layer)])
        self.activation = F.tanh
        self.linear = nn.Linear(out_channels, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        shape = [input.size(0), input.size(1)]
        conved = self.dropout(
            self.activation(self.convs(input.permute(0, 2, 1).view(shape[0], self.in_channel, -1, shape[1]))))
        conved = conved.permute(0, 2, 1, 3).view(shape[0], shape[1], -1)
        return self.linear(conved)


def seq_and_vec(seq_len, vec):
    return vec.unsqueeze(1).repeat(1, seq_len, 1)


def mask_logits(target, mask):
    return target * (1 - mask) + mask * (-1e30)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class LayerNorm(nn.Module):
    def __init__(self, shape, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        shape: inputs.shape
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.shape = (shape[-1],)
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(self.shape))
        if self.scale:
            self.gamma = Parameter(torch.ones(self.shape))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=self.shape[0], bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=self.shape[0], bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢?
            if self.center:
                torch.nn.init.constant(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
            if self.center:
                # print(self.beta_dense.weight.shape, cond.shape)
                self.beta_dense(cond)
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class Attention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """
    """
    input_size 输入的hidden的size，batch*seq*seq*dim
    query_size 输入的目标位置的size,batch*seq*seq*dim
    feature_size 输入的实体位置的size，bath*seq*seq*dim
    """

    def __init__(self, input_size, attn_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.attn_size = attn_size

        self.ulinear = nn.Linear(self.input_size, attn_size)
        self.vlinear = nn.Linear(self.input_size, attn_size, bias=False)
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        # self.ulinear.weight.data.normal_(std=0.001)
        # self.vlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()  # use zero to give uniform attention at the beginning

    def forward(self, hidden, mask):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """

        batch_size, seq_len, _ = hidden.size()
        mask = (mask < 1).unsqueeze(1).repeat(1, seq_len, 1)
        x_proj = self.ulinear(hidden.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size).unsqueeze(1).repeat(1, seq_len, 1, 1)
        q_proj = self.vlinear(hidden.contiguous().view(-1, self.input_size)).contiguous().view(
            batch_size, seq_len, self.attn_size).unsqueeze(2).repeat(1, 1, seq_len, 1)
        projs = [x_proj, q_proj]
        scores = self.tlinear(F.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len, seq_len)

        # mask padding
        scores.data.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores, dim=2)
        # weighted average input vectors
        outputs = weights.bmm(hidden)
        return outputs


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """
    """
    input_size 输入的hidden的size，batch*seq*seq*dim
    query_size 输入的目标位置的size,batch*seq*seq*dim
    feature_size 输入的实体位置的size，bath*seq*seq*dim
    """

    def __init__(self, input_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.attn_size = attn_size

        self.wlinear = nn.Linear(self.feature_size, attn_size, bias=False)
        self.ulinear = nn.Linear(self.input_size, attn_size)
        self.vlinear = nn.Linear(self.input_size, attn_size, bias=False)
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        # self.ulinear.weight.data.normal_(std=0.001)
        # self.vlinear.weight.data.normal_(std=0.001)
        # self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()  # use zero to give uniform attention at the beginning

    def forward(self, hidden, mask, query):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """

        batch_size, seq_len, _ = hidden.size()
        mask = (mask < 1).unsqueeze(1).repeat(1, seq_len, 1)
        x_proj = self.ulinear(hidden.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size).unsqueeze(1).repeat(1, seq_len, 1, 1)
        q_proj = self.vlinear(hidden.contiguous().view(-1, self.input_size)).contiguous().view(
            batch_size, seq_len, self.attn_size).unsqueeze(2).repeat(1, 1, seq_len, 1)
        f_proj = self.wlinear(query.contiguous().view(-1, self.feature_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).unsqueeze(2).repeat(1, seq_len, seq_len, 1)
        projs = [x_proj, q_proj, f_proj]
        scores = self.tlinear(F.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len, seq_len)

        # mask padding
        scores.data.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores, dim=2)
        # weighted average input vectors
        outputs = weights.bmm(hidden)
        return outputs


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

