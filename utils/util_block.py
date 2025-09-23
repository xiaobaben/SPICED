import torch.nn as nn
import math
import torch
import copy
from utils.util import ModelConfig
import numpy as np
import random


def fix_randomness(SEED):
    """
    :param SEED:  Random SEED
    :return:
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_head):
        super(MultiHeadAttention, self).__init__()
        self.w_query = nn.Linear(input_size, input_size)
        self.w_key = nn.Linear(input_size, input_size)
        self.w_value = nn.Linear(input_size, input_size)
        self.num_head = num_head
        self.dense = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.input_size = input_size
        self.att_dropout = nn.Dropout(0.25)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        query = self.w_query(input_tensor)
        key = self.w_key(input_tensor)
        value = self.w_value(input_tensor)
        query = query.view(batch_size, -1, self.num_head, self.input_size//self.num_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_head, self.input_size // self.num_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_head, self.input_size // self.num_head).permute(0, 2, 1, 3)
        attention_score = torch.matmul(query, key.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(input_tensor.shape[2])
        attention_prob = nn.Softmax(dim=1)(attention_score)
        attention_prob = self.att_dropout(attention_prob)
        context = torch.matmul(attention_prob, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.num_head*(self.input_size//self.num_head))
        hidden_state = self.dense(context)
        return hidden_state


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.relu(self.linear1(x)))
        x = self.drop(self.relu(self.linear2(x)))
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a1 = nn.Parameter(torch.ones(features))
        self.b1 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a1 * (x - mean) / (std + self.eps) + self.b1


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.pos_emb = PositionalEncoding(size)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        return self.sublayer[1](x, self.feed_forward)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, layer_num, drop_out, n_head):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multi_attention = MultiHeadAttention(d_model, d_model, n_head)
        self.dropout_rate = drop_out
        self.feedforward = FeedForward(d_model, d_model*4, self.dropout_rate)
        self.encoder = EncoderLayer(d_model, self_attn=self.multi_attention, feed_forward=self.feedforward,
                                    dropout=self.dropout_rate)
        self.layer_num = layer_num

    def forward(self, x):
        for _ in range(self.layer_num):
            x = self.encoder(x)
        return x


def evaluator(model, dl, args, noise=False):
    if type(model) == tuple:
        model[0].eval()
        model[1].eval()
        model[2].eval()
    else:
        model.eval()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model[0].to(device)
    model[1].to(device)
    model[2].to(device)

    model_param = ModelConfig(args.dataset)
    y_pred = []
    y_test = []
    predictions = None
    bh = False
    with torch.no_grad():
        for batch_idx, data in enumerate(dl):
            x, _, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            if args.dataset == 'ISRUC':
                epoch_size = model_param.EpochLength
                eog = x[:, :, :2, :]
                eeg = x[:, :, 2:, :]
                eog = eog.contiguous().view(-1, model_param.EogNum, epoch_size)
                eeg = eeg.contiguous().view(-1, model_param.EegNum, epoch_size)
                eeg_eog_feature = model[0](eeg, eog)
                eeg_eog_feature = model[1](eeg_eog_feature)
                prediction = model[2](eeg_eog_feature)
            elif args.dataset == 'FACED':
                emotion_feature = model[0](x)
                emotion_feature = model[1](emotion_feature)
                prediction = model[2](emotion_feature)
            elif args.dataset == 'BCI2000':
                mi_feature = model[0](x)
                mi_feature = model[1](mi_feature)
                prediction = model[2](mi_feature)
            if not bh:
                predictions = prediction
                bh = True
            else:
                predictions = torch.concat((predictions, prediction), dim=0)
            _, predicted = torch.max(prediction.data, dim=1)
            predicted, labels = torch.flatten(predicted), torch.flatten(labels)

            predicted = predicted.tolist()
            y_pred.extend(predicted)
            labels = labels.tolist()
            y_test.extend(labels)
        report = (y_test, y_pred, predictions)
        return report


def weighted_model_fusion(models, importance_weights, args):
    assert len(models) == len(importance_weights), "Number of models does not match number of weights"
    assert len(models) > 0, "Model list cannot be empty"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    models = [m.to(device) for m in models]

    # Weight normalization
    weights = torch.tensor(importance_weights, dtype=torch.float32, device=device)
    weights /= weights.sum() + 1e-8

    fused_model = copy.deepcopy(models[0])
    # fused_model = type(models[0])(args).to(device)
    # fused_model.load_state_dict(models[0].state_dict())
    fused_model.eval()

    # Initialization
    fused_params = {name: torch.zeros_like(param) for name, param in fused_model.named_parameters()}
    fused_buffers = {name: torch.zeros_like(buffer) for name, buffer in fused_model.named_buffers()}

    # Fusion
    with torch.no_grad():
        for model, weight in zip(models, weights):
            model.eval()

            for name, param in model.named_parameters():
                fused_params[name] += weight * param.data

            for name, buffer in model.named_buffers():
                target = fused_buffers[name]
                if buffer.dtype.is_floating_point:
                    target += weight * buffer.data.to(target.dtype)

    with torch.no_grad():
        fused_model.load_state_dict({
            'params': fused_params,
            'buffers': fused_buffers
        }, strict=False)

    validate_parameter_range(fused_model, models)
    return fused_model


def validate_parameter_range(fused_model, original_models):
    for name, param in fused_model.named_parameters():
        min_val = min(m.state_dict()[name].min() for m in original_models)
        max_val = max(m.state_dict()[name].max() for m in original_models)
        assert param.min() >= min_val and param.max() <= max_val, f"Parameter {name} exceeds the original range"


# def get_item(rand, signal):
#     if rand == 1:
#         return flipping(signal)
#     elif rand == 2:
#         return scaling(signal)
#     else:
#         return negate(signal)
#
#
# def augmentation(eeg, eog, args):
#     rand = np.random.randint(args["rand"])
#     fix_randomness(rand)
#     rand1 = np.random.randint(1, 4)
#     rand2 = np.random.randint(1, 4)
#     while rand1 == rand2:
#         rand2 = np.random.randint(1, 4)
#
#     eeg_aug1 = get_item(rand1, eeg)
#     eog_aug1 = get_item(rand1, eog)
#
#     eeg_aug2 = get_item(rand2, eeg)
#     eog_aug2 = get_item(rand2, eog)
#
#     return eeg_aug1, eog_aug1, eeg_aug2, eog_aug2
#
#
# def negate(signal):
#     """
#     negate the signal
#     """
#     # sigma = np.random.uniform(0.8, 1.2)
#     negated_signal = signal * (-1)
#     return negated_signal
#
#
# def scaling(x):
#     sigma = np.random.uniform(1.1)
#     return x*sigma
#
#
# def flipping(x):
#     return torch.flip(x, dims=[1])
#
#
# class NTXentLoss(torch.nn.Module):
#
#     def __init__(self, device, batch_size, temperature, use_cosine_similarity):
#         super(NTXentLoss, self).__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.device = device
#         self.softmax = torch.nn.Softmax(dim=-1)
#         # self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
#         self.similarity_function = self._get_similarity_function(use_cosine_similarity)
#         self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
#
#     def _get_similarity_function(self, use_cosine_similarity):
#         if use_cosine_similarity:
#             self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
#             return self._cosine_simililarity
#         else:
#             return self._dot_simililarity
#
#     def get_correlated_mask(self):
#         diag = np.eye(2 * self.batch_size)
#         l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
#         l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
#         mask = torch.from_numpy((diag + l1 + l2))
#         mask = (1 - mask).type(torch.bool)
#         return mask.to(self.device)
#
#     @staticmethod
#     def _dot_simililarity(x, y):
#         v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
#         # x shape: (N, 1, C)
#         # y shape: (1, C, 2N)
#         # v shape: (N, 2N)
#         return v
#
#     def _cosine_simililarity(self, x, y):
#         # x shape: (N, 1, C)
#         # y shape: (1, 2N, C)
#         # v shape: (N, 2N)
#         v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
#         return v
#
#     def forward(self, zis, zjs):
#         losses = 0
#         for i in range(zis.shape[0]):
#             zis_ = zis[i, :, :]
#             zjs_ = zjs[i, :, :]
#
#             representations = torch.cat([zjs_, zis_], dim=0)
#
#             similarity_matrix = self.similarity_function(representations, representations)
#             # filter out the scores from the positive samples
#             self.batch_size = zis_.shape[0]
#             l_pos = torch.diag(similarity_matrix, self.batch_size)
#             r_pos = torch.diag(similarity_matrix, -self.batch_size)
#
#             positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
#             # print(positives.shape)
#             # negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
#             negatives = similarity_matrix[self.get_correlated_mask().type(torch.bool)].view(2 * self.batch_size, -1)
#             # print(negatives.shape)
#             logits = torch.cat((positives, negatives), dim=1)
#             logits /= self.temperature
#
#             labels = torch.zeros(2 * self.batch_size).to(self.device).long()
#             loss = self.criterion(logits, labels)
#             losses += loss
#         return losses / (2 * self.batch_size)
#
#
# class ProjHead(nn.Module):
#     def __init__(self, args):
#         super(ProjHead, self).__init__()
#         self.input_length = 512
#         self.classifier = nn.Sequential(
#             nn.Linear(self.input_length, self.input_length // 2),
#             nn.BatchNorm1d(self.input_length // 2),
#             nn.ReLU(True),
#             nn.Linear(self.input_length // 2, self.input_length // 4),
#         )
#
#     def forward(self, x):
#         batch = x.shape[0]
#         x = x.view(-1, self.input_length)
#         x = self.classifier(x)
#         x = x.view(batch, 20, -1)
#         return x