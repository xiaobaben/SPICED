import numpy as np


class EEGFeatureStructure:
    def __init__(self, args, n_channels=6):
        self.n_channels = n_channels
        self.args = args
        if self.args.prototype == 'prototype1':
            self.feature_groups = {
                'time_stats': self._get_mask(0, 4),
                'hjorth': self._get_mask(4, 6),
                'psd': self._get_mask(6, 11),
                'wavelet': self._get_mask(11, 15),
            }
            self.domain_weights = {
                'time_stats': 0.9,
                'hjorth': 1.0,
                'psd': 1.5,
                'wavelet': 1.2}

        elif self.args.prototype == 'prototype2':
            self.feature_groups = {
                'time_stats': self._get_mask(0, 6),
                'psd': self._get_mask(6, 11),
                'wavelet': self._get_mask(11, 15),
                'sleep': self._get_mask(15, 19)
            }
            self.domain_weights = {
                'time_stats': 0.9,
                'psd': 1.3,
                'wavelet': 1.35,
                'sleep': 0.9
            }
        elif self.args.prototype == 'prototype3':
            self.feature_groups = {
                'time_stats': self._get_mask(0, 6),
                'psd': self._get_mask(6, 11),
                'wavelet': self._get_mask(11, 15),
            }
            self.domain_weights = {
                'time_stats': 0.9,
                'psd': 1.5,
                'wavelet': 1.2
            }

    def _get_mask(self, start_feat, end_feat):
        return np.concatenate([np.arange(c * 15 + start_feat, c * 15 + end_feat)
                               for c in range(self.n_channels)])

    def flatten_to_vector(self, feature_matrix):
        return feature_matrix.reshape(-1)

    def group_features(self, feature_vector):
        return {domain: feature_vector[mask]
                for domain, mask in self.feature_groups.items()}


class PrototypeSimilarityCalculator:
    def __init__(self, feature_structure):
        self.fs = feature_structure

    def domain_weighted_cosine(self, x, y):
        """分域加权余弦相似度"""
        x_groups = self.fs.group_features(x)
        y_groups = self.fs.group_features(y)

        total_sim = 0.0
        for domain in self.fs.feature_groups:
            vec_x = x_groups[domain]
            vec_y = y_groups[domain]
            dot = np.dot(vec_x, vec_y)
            norm = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
            sim = dot / (norm + 1e-8)
            total_sim += sim * self.fs.domain_weights[domain]

        return total_sim / sum(self.fs.domain_weights.values())

    # def channel_correlation(self, x, y):
    #     x_matrix = x.reshape(self.fs.n_channels, -1)
    #     y_matrix = y.reshape(self.fs.n_channels, -1)
    #
    #     corrs = []
    #     for ch in range(self.fs.n_channels):
    #         ch_corr = np.corrcoef(x_matrix[ch], y_matrix[ch])[0, 1]
    #         corrs.append(ch_corr)
    #     return np.nanmean(corrs)  # 忽略NaN值
    #
    # def dynamic_euclidean(self, x, y):
    #     variances = np.var([x, y], axis=0)
    #     select_mask = variances > np.percentile(variances, 30)
    #     x_sel = x[select_mask]
    #     y_sel = y[select_mask]
    #
    #     weights = 1 / (variances[select_mask] + 1e-8)
    #     weighted_diff = (x_sel - y_sel) * weights
    #     return 1 / (1 + np.sqrt(np.sum(weighted_diff ** 2)))


def get_SimilarityCalculator(args, n_channels):
    feature_structure = EEGFeatureStructure(args=args, n_channels=n_channels)
    return PrototypeSimilarityCalculator(feature_structure)




