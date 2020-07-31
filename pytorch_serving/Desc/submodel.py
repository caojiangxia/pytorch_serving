import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch_serving.Desc.layers import *



class ObjBaseModel(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """

    def __init__(self, opt):
        super(ObjBaseModel, self).__init__()
        self.input_size = 2 * opt['hidden_dim'] + opt['dis_dim']

        self.layernorm = LayerNorm(shape=(opt['max_len'], opt['hidden_dim']), cond_dim=2 * opt['hidden_dim'],
                                   conditional=True)
        self.dis_embedding = nn.Embedding(600, opt['dis_dim'])
        self.dis_embedding.weight.data.uniform_(-1.0, 1.0)
        self.dgcnn = DGCNN(self.input_size, 1)
        self.attention = PositionAwareAttention(opt['hidden_dim'], 2 * opt['hidden_dim'], int(opt['hidden_dim'] / 2))

        self.start_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.input_size, 512),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.end_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.input_size, 512),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden, sentence_rep, distance_to_s, s_start, s_end, masks):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """

        batch_size, seq_len, input_size = hidden.shape

        subj_start_hidden = torch.gather(hidden, dim=1,
                                         index=s_start.unsqueeze(1).unsqueeze(2).repeat(1, 1, input_size)).squeeze(1)
        subj_end_hidden = torch.gather(hidden, dim=1,
                                       index=s_end.unsqueeze(1).unsqueeze(2).repeat(1, 1, input_size)).squeeze(1)
        subj_cond = torch.cat([subj_start_hidden, subj_end_hidden], dim=1)

        dis_emb = self.dis_embedding(distance_to_s + 300)

        global_rep = self.attention(hidden, masks, subj_cond)

        outputs = self.layernorm(hidden, subj_cond)
        outputs = torch.cat([outputs, dis_emb, global_rep], dim=2)

        masks = masks.unsqueeze(-1).float()
        outputs = self.dgcnn(outputs, masks)

        start_probs = self.start_linear(outputs)
        end_probs = self.end_linear(outputs)
        probs = torch.cat([start_probs, end_probs], dim=-1)

        return probs





