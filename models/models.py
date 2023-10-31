import torch

import numpy as np
import torch

"""
Models for SASRec. Reference: https://github.com/pmixer/SASRec.pytorch   ;    https://github.com/kang205/SASRec
"""


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


""""
Base models
"""

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.args = args

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_size, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_seq, args.hidden_size)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_size,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs, seqs_embs):
        seqs = seqs_embs * self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor((log_seqs == 0).to('cpu')).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality

        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats, seqs_embs

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, adv=None):
        log_feats, seqs_embs = self.log2feats(log_seqs, adv)

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        seqs_embs = self.item_emb(log_seqs)
        log_feats, seqs_embs = self.log2feats(log_seqs, seqs_embs)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(item_indices)  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def predict_counter(self, user_ids, log_seqs, item_indices, delta):
        seqs_embs = self.item_emb(log_seqs)
        delta = torch.unsqueeze(delta, dim=-1).repeat(1, 1, seqs_embs.shape[-1])
        seqs_embs = seqs_embs * delta
        log_feats, seqs_embs = self.log2feats(log_seqs, seqs_embs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(item_indices)  # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

"""
Models for GRU4Rec
"""


class GRU4Rec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GRU4Rec, self).__init__()
        self.user_num = user_num  # may not be useful
        self.item_num = item_num
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.item_embedding = torch.nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)  # item embedding
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.rnn = torch.nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                batch_first=True)  # TODO: check batch first

    def get_seq_embedding(self, log_vecs):
        """
        get intermediate sequence embedding according to each position
        :return: embeddings to each position in the sequence
        """
        output, hidden = self.rnn(log_vecs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, adv=None):
        seqs_embs = self.item_embedding(log_seqs)
        pos_embs = self.item_embedding(pos_seqs)
        neg_embs = self.item_embedding(neg_seqs)
        seqs_embs_drop = self.emb_dropout(seqs_embs)
        log_feats, _ = self.rnn(seqs_embs_drop)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_vecs = self.item_embedding(log_seqs)
        log_feats, _ = self.rnn(log_vecs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits  # preds # (U, I)
    
    def predict_counter(self, user_ids, log_seqs, item_indices, delta):  # for counterfactual explanation
        log_vecs = self.item_embedding(log_seqs)
        delta = torch.unsqueeze(delta, dim=-1).repeat(1, 1, log_vecs.shape[-1])
        log_vecs = log_vecs * delta
        log_feats, _ = self.rnn(log_vecs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits