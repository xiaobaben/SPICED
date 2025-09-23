import copy
import torch
import torch.nn as nn
from utils.util import ModelConfig
from utils.util_block import MultiHeadAttentionBlock
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class SynapticFramework(object):
    def __init__(self, backbone, args):
        super(SynapticFramework, self).__init__()
        self.args = args
        self.model_param = ModelConfig(self.args.dataset)
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        """Backbone Model"""
        self.feature_extractor = backbone[0].to(self.device)
        self.feature_encoder = backbone[1].to(self.device)
        self.classifier = backbone[2].to(self.device)

        self.feature_extractor_t = copy.deepcopy(backbone[0]).to(self.device)
        self.feature_encoder_t = copy.deepcopy(backbone[1]).to(self.device)
        self.classifier_t = copy.deepcopy(backbone[2]).to(self.device)

        """CPC Framework"""
        self.num_channels = self.model_param.EncoderParam.d_model
        self.d_model = self.model_param.EncoderParam.d_model
        self.timestep = 3
        self.Wk = nn.ModuleList([nn.Sequential(nn.Linear(self.d_model, self.d_model * 4),
                                               nn.Dropout(0.1),
                                               nn.GELU(),
                                               nn.Linear(self.d_model * 4, self.d_model)).to(self.device)
                                 for _ in range(self.timestep)])

        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.encoder = MultiHeadAttentionBlock(self.d_model,
                                               self.model_param.EncoderParam.layer_num,
                                               self.model_param.EncoderParam.drop,
                                               self.model_param.EncoderParam.n_head).to(self.device)


        self.optimizer = torch.optim.Adam([
            {"params": list(self.feature_extractor.parameters())},
            {"params": list(self.feature_encoder.parameters())},
            {"params": list(self.classifier.parameters())}],
            lr=self.args.cl_lr, betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay)

        self.optimizer_t = torch.optim.Adam([
            {"params": list(self.feature_extractor_t.parameters())},
            {"params": list(self.feature_encoder_t.parameters())}],
            lr=self.args.cl_lr, betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay)

        self.optimizer_cpc = torch.optim.Adam([
            {"params": list(self.encoder.parameters())},
            {"params": list(self.Wk.parameters())}],
            lr=self.args.lr, betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay)

        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.confidence_level = self.args.training_confidence

    def new_cpc(self, x_new, feature_extractor, feature_encoder):
        batch = x_new.shape[0]
        if self.args.dataset == 'ISRUC':
            eog = x_new[:, :, :2, :]
            eeg = x_new[:, :, 2:, :]
            eog = eog.contiguous().view(-1, self.model_param.EogNum, self.model_param.EpochLength)
            eeg = eeg.contiguous().view(-1, self.model_param.EegNum, self.model_param.EpochLength)
            ff = feature_extractor(eeg, eog)
            ff = feature_encoder(ff)  # batch, 20, 128
            t_samples = torch.randint(low=10, high=(self.model_param.SeqLength - self.timestep), size=(1,)).long().to(self.device)
        elif self.args.dataset in ['FACED', 'BCI2000']:
            ff = feature_extractor(x_new)
            ff = feature_encoder(ff)
            t_samples = torch.randint(low=5, high=(10 - self.timestep), size=(1,)).long().to(self.device)

        loss = 0
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = ff[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = ff[:, :t_samples + 1, :]

        output = self.encoder(forward_seq)  # batch, 15, 128

        c_t = output[:, t_samples, :].view(batch, -1)  # batch, 128

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)  # 5, batch, 128
        for i in range(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # batch, 128
        for i in range(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # batch, 128   128, batch
            loss += torch.sum(torch.diag(self.lsoftmax(total)))
        loss /= -1. * batch * self.timestep
        return loss

    def replay_cls(self, x_new, x_replay, replay_label):
        batch = x_new.shape[0]
        if self.args.dataset == 'ISRUC':
            eog_new = x_new[:, :, :2, :]
            eeg_new = x_new[:, :, 2:, :]

            eog_replay = x_replay[:, :, :2, :]
            eeg_replay = x_replay[:, :, 2:, :]

            eeg = torch.concat((eeg_new, eeg_replay), dim=0)
            eog = torch.concat((eog_new, eog_replay), dim=0)

            eog = eog.view(-1, self.model_param.EogNum, self.model_param.EpochLength)
            eeg = eeg.view(-1, self.model_param.EegNum, self.model_param.EpochLength)

            eeg_eog_feature = self.feature_extractor(eeg, eog)
            eeg_eog_feature = self.feature_encoder(eeg_eog_feature)

            eeg_eog_feature_replay = eeg_eog_feature[batch:, :, :]
            eeg_eog_feature_new = eeg_eog_feature[:batch, :, :]
            pred_target = self.classifier(eeg_eog_feature_new)
            pred_target = pred_target.permute(0, 2, 1)
            pred_target = pred_target.view(-1, 5)

            with torch.no_grad():
                eog_new = eog_new.contiguous().view(-1, self.model_param.EogNum, self.model_param.EpochLength)
                eeg_new = eeg_new.contiguous().view(-1, self.model_param.EegNum, self.model_param.EpochLength)
                ff = self.feature_extractor_t(eeg_new, eog_new)
                ff = self.feature_encoder_t(ff)
                mean_t_pred = self.classifier_t(ff)
                mean_t_pred = mean_t_pred.permute(0, 2, 1)
                mean_t_pred = mean_t_pred.view(-1, 5)
                mean_t_pred = self.softmax(mean_t_pred)  # 640, 5
                pred_prob = mean_t_pred.max(1, keepdim=True)[0].squeeze()
                target_pseudo_labels = mean_t_pred.max(1, keepdim=True)[1].squeeze()

            confident_pred = pred_target[pred_prob > self.confidence_level]
            confident_labels = target_pseudo_labels[pred_prob > self.confidence_level]
            loss_new = self.cross_entropy(confident_pred, confident_labels.long())
            if self.args.replay_strategy == 'cls':
                pred_replay = self.classifier(eeg_eog_feature_replay)
                loss_replay = self.cross_entropy(pred_replay, replay_label.long())

        if self.args.dataset in ['FACED', 'BCI2000']:
            x = torch.concat((x_new, x_replay), dim=0)

            emotion_feature = self.feature_extractor(x)  # batch, 20, 512
            emotion_feature = self.feature_encoder(emotion_feature)

            emotion_feature_replay = emotion_feature[batch:, :, :]
            emotion_feature_new = emotion_feature[:batch, :, :]

            pred_target = self.classifier(emotion_feature_new)
            pred_target = pred_target.view(-1, self.model_param.NumClasses)

            with torch.no_grad():
                ff = self.feature_extractor_t(x_new)
                ff = self.feature_encoder_t(ff)
                mean_t_pred = self.classifier_t(ff)
                mean_t_pred = mean_t_pred.view(-1, self.model_param.NumClasses)
                mean_t_pred = self.softmax(mean_t_pred)  # 640, 5
                pred_prob = mean_t_pred.max(1, keepdim=True)[0].squeeze()
                target_pseudo_labels = mean_t_pred.max(1, keepdim=True)[1].squeeze()

            confident_pred = pred_target[pred_prob > self.confidence_level]
            confident_labels = target_pseudo_labels[pred_prob > self.confidence_level]
            loss_new = self.cross_entropy(confident_pred, confident_labels.long())

            if self.args.replay_strategy == 'cls':
                pred_replay = self.classifier(emotion_feature_replay)
                replay_label = replay_label.view(-1)
                loss_replay = self.cross_entropy(pred_replay, replay_label.long())

        loss = self.args.weight_new_cls*loss_new + (1-self.args.weight_new_cls)*loss_replay
        return loss

    def update(self, x_new, x_replay, y_replay, is_cls, epoch):
        if is_cls:
            if epoch <= 10:
                self.optimizer_t.zero_grad()
                self.optimizer_cpc.zero_grad()
                loss = self.new_cpc(x_new, self.feature_extractor_t, self.feature_encoder_t)
                loss.backward()
                self.optimizer_t.step()
                self.optimizer_cpc.step()
            else:
                self.optimizer.zero_grad()
                loss = self.replay_cls(x_new, x_replay, y_replay)
                loss.backward()
                self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            self.optimizer_cpc.zero_grad()
            loss = self.new_cpc(x_new, self.feature_extractor, self.feature_encoder)
            loss.backward()
            self.optimizer.step()
            self.optimizer_cpc.step()
        return loss.item(), (self.feature_extractor, self.feature_encoder, self.classifier)

