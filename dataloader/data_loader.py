import math

from torch.utils.data import Dataset
import numpy as np
import torch


class BuildDataset(Dataset):
    def __init__(self, data_path, dataset):
        self.data_path = data_path
        self.dataset = dataset
        self.len = len(self.data_path[0])

    def __getitem__(self, index):
        x_data = np.load(self.data_path[0][index])
        y_data = np.load(self.data_path[1][index])
        x_data = torch.from_numpy(np.array(x_data).astype(np.float32))
        y_data = torch.from_numpy(np.array(y_data).astype(np.float32))
        return x_data, x_data, y_data

    def __len__(self):
        return self.len


class BuildReplayedDataset(Dataset):
    def __init__(self, new_path, synaptic_net, node, args, source_net):
        """
        :param new_path: Path to the incremental individual sample
        :param synaptic_net: Synaptic network
        :param node: Incremental individual node
        """
        self.new_path = new_path
        self.synaptic_net = synaptic_net
        self.source_net = source_net
        self.node = node
        self.len = len(self.new_path[0])
        self.args = args
        self.train_path_data, self.train_path_label = self.get_replayed_data()
        print("Len New Data: ", len(self.new_path[0]), "Len Replay Data: ", len(self.train_path_data))

    def get_replayed_data(self):
        """
        Method1: Replay pseudo-label sample pairs from the connected nodes of the new node proportionally
        based on node importance, with a maximum length of self.len.
        Note: If the new node connects to more than K existing nodes, only the samples from the top K
        most important nodes are replayed.
        :return:
        """
        data_path = []
        label_path = []
        replay_radio = self.node.find_importance_knn(self.args)
        for replay_sub, replay_sub_imp in replay_radio:
            replay_sub_data_path = self.synaptic_net.nodes[replay_sub].sample_path['data']
            replay_sub_label_path = self.synaptic_net.nodes[replay_sub].sample_path['label']
            replay_sub_data_len = len(replay_sub_data_path)  # Get the number of saved samples from connected nodes

            replay_sub_replay_len = int(self.len * replay_sub_imp)  # Get the maximum number of samples that can be replayed for this node
            if not replay_sub_replay_len and replay_sub_data_len:
                replay_sub_replay_len += 1
            if replay_sub_replay_len >= replay_sub_data_len:  # If the maximum number of samples that can be replayed for this node is greater than the number of samples saved for this node
                data_path.extend(replay_sub_data_path)
                label_path.extend(replay_sub_label_path)
            else:
                sample_idx = list(np.random.choice(range(replay_sub_data_len), replay_sub_replay_len, replace=False))
                data_path.extend([replay_sub_data_path[i] for i in sample_idx])
                label_path.extend([replay_sub_label_path[i] for i in sample_idx])

        if len(data_path) < self.len:
            if not len(data_path):
                # If none of the connected nodes have high-confidence pseudo-labels, replay the sample label pairs from the nearest source domain node.
                max_sim = -math.inf
                replay_node = None
                for exist_node in self.source_net.nodes.values():
                    if exist_node.name == self.node.name:
                        continue
                    sim = self.source_net.sim_calculator.domain_weighted_cosine(exist_node.prototype, self.node.prototype)
                    if sim > max_sim:
                        replay_node = exist_node
                        max_sim = sim
                data_path = replay_node.sample_path['data']
                label_path = replay_node.sample_path['label']
                print(f"####Connected Nodes not not exist stored confident pseudo-labeled sample ####")
                print(f"####Thus replay the most similar individual {replay_node.name} from the source domain ####")
            repeat_times = (self.len // len(data_path)) + 1
            data_path = (data_path * repeat_times)
            label_path = (label_path * repeat_times)

        return data_path, label_path

    def __getitem__(self, index):
        x_data_new = np.load(self.new_path[0][index])
        x_data_new = torch.from_numpy(np.array(x_data_new).astype(np.float32))

        x_data_replay = np.load(self.train_path_data[index])
        y_data_replay = np.load(self.train_path_label[index])
        x_data_replay = torch.from_numpy(np.array(x_data_replay).astype(np.float32))
        y_data_replay = torch.from_numpy(np.array(y_data_replay).astype(np.float32))

        return x_data_new, x_data_replay, y_data_replay

    def __len__(self):
        return self.len


class Builder(object):
    def __init__(self, data_path, args):
        super(Builder, self).__init__()
        self.data_set = args.dataset
        self.data_path = data_path
        self.Dataset = BuildDataset(self.data_path, self.data_set)


class SynapticDatasetBuilder(object):
    def __init__(self, data_path, synaptic_net, node_name, args, source_net):
        super(SynapticDatasetBuilder, self).__init__()
        self.Dataset = BuildReplayedDataset(data_path, synaptic_net, synaptic_net.nodes[node_name], args, source_net)

