from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch_utils
import numpy as np
from torch.nn import  BCEWithLogitsLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import constant
from sols.sols import SOLS


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


class MainSequenceClassification(nn.Module):
    def __init__(self, num_labels, vocab, args):
        super(MainSequenceClassification, self).__init__()
        self.args = args

        def init_weights(module):
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

        self.latent_type = self.args.latent_type
        self.first_layer = args.first_layer
        self.second_layer = args.second_layer
        self.latent_dropout = args.latent_dropout
        self.speaker_reg = args.speaker_reg
        self.lasso_reg = args.lasso_reg
        self.l0_reg = args.l0_reg
        self.hidden = args.rnn_hidden_size
        self.dataset = args.dataset

        self.dropout = nn.Dropout(0.5)

        self.num_latent_graphs = args.num_latent_graphs

        if self.dataset == 'dialogre':
            self.NUM = 36

        # sols
        self.sols = SOLS(self.hidden // 2, args.diff_mlp_hidden, 1, args.diff_position,
                         self.latent_dropout, args.num_layer, self.first_layer,
                         self.second_layer, self.l0_reg, self.lasso_reg, self.speaker_reg,
                         alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        self.linear_latent = nn.Linear(2 * self.hidden, self.hidden // 2)
        self.num_layer = args.num_layer
        self.diff_position = args.diff_position
        self.diff_mlp_hidden = args.diff_mlp_hidden
        self.linear_latent = nn.Linear(2 * self.hidden, self.hidden // 2)
        self.latent_graph = SOLS(self.hidden // 2, self.diff_mlp_hidden, 1, self.diff_position,
                                 self.latent_dropout, self.num_layer, self.first_layer,
                                 self.second_layer, self.l0_reg, self.lasso_reg, self.speaker_reg,
                                 alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        self.linear_ablation = nn.Linear(768, self.hidden // 2)

        # classifier hidden dim
        self.classifier = nn.Linear(self.hidden + 768, num_labels * self.NUM)

        # LSTM needed layers
        self.vocab = vocab
        self.embedding = self.embedding = nn.Embedding(self.vocab.n_words, self.args.embed_dim,
                                                       padding_idx=constant.PAD_ID)
        self.init_pretrained_embeddings_from_numpy(np.load(open(args.embed_f, 'rb'), allow_pickle=True))
        self.lstm = nn.LSTM(args.embed_dim + 30, args.rnn_hidden_size, args.lstm_layers, batch_first=True,
                            dropout=args.lstm_dropout, bidirectional=True)
        self.pooling = MultiHeadedPooling(2 * args.rnn_hidden_size)

        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), 30)
        self.ner_emb = nn.Embedding(6, 20, padding_idx=5)

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if self.args.tune_topk <= 0:
            print("Do not fine tune word embedding layer")
            self.embedding.weight.requires_grad = False
        elif self.args.tune_topk < self.vocab.n_words:
            print(f"Finetune top {self.args.tune_topk} word embeddings")
            self.embedding.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.args.tune_topk))
        else:
            print("Finetune all word embeddings")

    def forward(self, input_ids, token_type_ids, attention_mask, subj_pos=None, obj_pos=None, pos_ids=None, x_type=None,
                y_type=None, labels=None, n_class=1):
        seq_length = input_ids.size(2)
        attention_mask_ = attention_mask.view(-1, seq_length)
        batch_size = input_ids.size(0)
        mask_local = ~(attention_mask_.data.eq(0).view(batch_size, seq_length))

        l = (attention_mask_.data.cpu().numpy() != 0).astype(np.int64).sum(1)

        real_length = max(l)

        input_ids = input_ids.view(-1, seq_length)
        emb = self.embedding(input_ids)
        pos_ids = pos_ids.view(-1, seq_length)
        pos_emb = self.pos_emb(pos_ids)
        emb = torch.cat([emb, pos_emb], dim=2)

        h0, c0 = rnn_zero_state(batch_size, 384, 1)
        rnn_input = pack_padded_sequence(emb, l, batch_first=True, enforce_sorted=False)
        rnn_output, (ht, ct) = self.lstm(rnn_input, (h0, c0))
        word_embedding, _ = pad_packed_sequence(rnn_output, batch_first=True)

        attention_mask_ = attention_mask_[:, :real_length]
        word_embedding = word_embedding * attention_mask_[:, :, None]
        mask_local = mask_local[:, :real_length].unsqueeze(-1)
        pooled_output = pool(word_embedding, ~mask_local)

        subj_pos, obj_pos = subj_pos.view(batch_size, -1), obj_pos.view(batch_size, -1)
        subj_pos = subj_pos[:, :real_length]
        obj_pos = obj_pos[:, :real_length]
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(-1), obj_pos.eq(0).eq(0).unsqueeze(-1)  # invert mask
        subj_mask, obj_mask = subj_mask.cuda(), obj_mask.cuda()

        subj_out = pool(word_embedding, subj_mask, type='max')
        obj_out = pool(word_embedding, obj_mask, type='max')

        x_ner_emb = self.ner_emb(x_type)
        y_ner_emb = self.ner_emb(y_type)
        x_ner_emb = x_ner_emb.view(batch_size, -1)
        y_ner_emb = y_ner_emb.view(batch_size, -1)
        subj_out = torch.cat([subj_out, x_ner_emb], dim=-1)
        obj_out = torch.cat([obj_out, y_ner_emb], dim=-1)
        pooled_output_ = pooled_output
        pooled_output = torch.cat([pooled_output, subj_out, obj_out], dim=1)
        if self.latent_type =='sols':
            pooled_output = pooled_output_

        # latent type
        ctx = self.dropout(self.linear_latent(word_embedding))
        # generate speaker mask
        x = torch.tensor(0)
        if self.args.encoder_type == 'LSTM':
            x = False

        bz, max_len, _ = ctx.shape
        if self.num_latent_graphs == 1:
            speaker_mask = torch.zeros(bz, max_len, max_len)
            for i, (x_p_t, y_p_t) in enumerate(zip(subj_mask, obj_mask)):
                x_p = []
                y_p = []
                for idx in range(len(x_p_t.squeeze(1))):
                    if x_p_t.squeeze(1)[idx] == x or x_p_t.squeeze(1)[idx] == False:
                        x_p.append(idx)
                for idx in range(len(y_p_t.squeeze(1))):
                    if y_p_t.squeeze(1)[idx] == x or y_p_t.squeeze(1)[idx] == False:
                        y_p.append(idx)
                if len(x_p) == 0:
                    x_p = [0]
                if len(y_p) == 0:
                    y_p = [0]
                for x in x_p:
                    speaker_mask[i][x] = 1
                for y in y_p:
                    speaker_mask[i][y] = 1
            speaker_mask = speaker_mask.cuda()
            h_a, _, aux_loss = self.latent_graph(ctx, mask_local.squeeze(-1), speaker_mask, None)

            h = h_a[:, :real_length, :]

            subj_out = pool(h, subj_mask, type='max')
            obj_out = pool(h, obj_mask, type='max')
            subj_out = torch.cat([subj_out], dim=-1)
            obj_out = torch.cat([obj_out], dim=-1)

            output = self.dropout(torch.cat([pooled_output, subj_out, obj_out], dim=1))
            logits = self.classifier(output)
            logits = logits.view(-1, self.NUM)
            labels = labels.view(-1, self.NUM)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            if loss.item() > 10:
                print("something wrong with loss!")
            loss = loss + aux_loss
            return loss, logits

        elif self.num_latent_graphs == 2:
            speaker_mask_a = torch.zeros(bz, max_len, max_len)
            speaker_mask_b = torch.zeros(bz, max_len, max_len)

            for i, (x_p_t, y_p_t) in enumerate(zip(subj_mask, obj_mask)):
                x_p = (x_p_t.squeeze(1) == x).nonzero().flatten()
                y_p = (y_p_t.squeeze(1) == x).nonzero().flatten()
                x_p = x_p.tolist()
                y_p = y_p.tolist()
                if len(x_p) == 0:
                    x_p = [0]
                if len(y_p) == 0:
                    y_p = [0]

                # explicitly learning paradigm matters
                for x in x_p:
                    speaker_mask_a[i][x] = 1

                for y in y_p:
                    speaker_mask_b[i][y] = 1

            speaker_mask_a = speaker_mask_a.cuda()
            speaker_mask_b = speaker_mask_b.cuda()

            h_a, h_b, aux_loss = self.latent_graph(ctx, mask_local.squeeze(-1), speaker_mask_a, speaker_mask_b)

            h_a = h_a[:, :real_length, :]
            h_b = h_b[:, :real_length, :]

            subj_out = pool(h_a, subj_mask, type='max')
            obj_out = pool(h_b, obj_mask, type='max')

            output = self.dropout(torch.cat([pooled_output, subj_out, obj_out], dim=1))
            logits = self.classifier(output)
            logits = logits.view(-1, self.NUM)
            labels = labels.view(-1, self.NUM)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            if loss.item() > 10:
                print("something wrong with loss!")
            loss = loss + aux_loss
            return loss, logits


class MultiHeadedPooling(nn.Module):
    def __init__(self, model_dim):
        self.model_dim = model_dim
        super(MultiHeadedPooling, self).__init__()
        self.linear_keys = nn.Linear(model_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        scores = self.linear_keys(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e18)

        attn = self.softmax(scores).unsqueeze(-1)
        output = torch.sum(attn * x, -2)
        return output


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


import torch
from torch._six import with_metaclass


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


# mypy doesn't understand torch._six.with_metaclass
class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase)):  # type: ignore
    pass


from torch._C import _ImperativeEngine as ImperativeEngine

Variable._execution_engine = ImperativeEngine()  # type: ignore