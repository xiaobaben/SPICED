import numpy as np
from utils.calculate_similarity import get_SimilarityCalculator
from builder.synaptic_visualization import visualize_synaptic_net
import argparse


class SynapticNode:
    def __init__(self, name, feature):
        """
        :param name: Node name（Individual-specific）
        :param feature: Individual Initial Feature

        self.connections: Record information of nodes connected to the current node,
                          including similarity and connection strength.
        self.sample_path: Record the high-confidence sample label pairs of the current node.
        self.model_path: Record the model trained after the current node.
        """
        self.name = name
        self.prototype = feature
        self.connections = {}  # {target_node: (similarity, strength)}
        self.sample_path = {'data': [], 'label': []}
        self.model_path = []
        self.step = 1

    def connection_count(self):
        return len(self.connections)

    def get_average_strength(self):
        """Obtain the average connection strength of the current node,
        reflecting the overall activation level of its associated neural pathways."""
        return np.mean([strength for _, strength in self.connections.values()])

    def get_target_importance(self, alpha=0.6, beta=0.4):
        """Obtain the importance of nodes connected to the current node with respect to the current node:
        Importance = alpha*Similarity + beta*target_nodes_average_strength"""
        importance = dict()
        if len(self.connections) < 1:
            return importance

        connect_info = []
        for connected_node, (similarity, _) in self.connections.items():
            connected_node_average_strength = connected_node.get_average_strength()
            connect_info.append([connected_node, similarity, connected_node_average_strength])

        strength_sum = sum([n[2] for n in connect_info])

        for i in range(len(connect_info)):
            connect_info[i].append(connect_info[i][2] / strength_sum)
        for node in connect_info:
            importance[node[0].name] = {'importance': alpha*node[1]+beta*node[3],
                                        'similarity': node[1],
                                        'average_strength': node[2]}
        return importance

    def find_importance_knn(self, args):
        """Fina most importance node to current node"""
        if not self.connections:
            return None
        k = args.k_factor
        importance = self.get_target_importance(alpha=args.alpha_sim, beta=args.alpha_str).items()
        importance_info = []
        for (connect_name, connect_info) in importance:
            importance_info.append([connect_name, connect_info['importance']])
        sum_ = sum([i[1] for i in importance_info])
        importance_info = sorted([[i[0], i[1] / sum_] for i in importance_info], key=lambda x: x[1], reverse=True)
        if len(importance_info) > k:
            importance_info = importance_info[:k]
            sum_2 = sum([i[1] for i in importance_info])
            importance_info = [[i[0], i[1] / sum_2] for i in importance_info]
        return importance_info

    def add_connection(self, target_node, similarity):
        """Add synaptic connection"""
        if target_node == self:
            return

        if self.connections.get(target_node):
            return

        new_strength = self.connections.get(target_node, (0, 0))[1] + 1
        self.connections[target_node] = (similarity, new_strength)
        target_node.connections[self] = (similarity, new_strength)

    def renormalization_synapses(self, decay_factor=20):
        """Synaptic Renormalization"""
        for node, (sim, strength) in self.connections.items():
            # new_strength = (strength * decay_factor)
            new_strength = strength * np.exp(-self.step / decay_factor)  # Ebbinghaus forgetting curve
            self.connections[node] = (sim, new_strength)
        self.step += 1

    def consolidate_memory(self, consolidation_node_list, consolidation_factor=1.3, up_bound=3):
        """Synaptic Consolidation"""
        for node in self.connections:
            sim, strength = self.connections[node]
            if strength != 1:
                update_strength = strength * consolidation_factor
                if update_strength >= up_bound:
                    update_strength = 3
                self.connections[node] = (sim, update_strength)
                """ Determine whether the target node node is among all enhanced nodes. 
                If not, update the connection strength of the target node; 
                if it is, the update will be handled in the next iteration."""
                """Ensure that synaptic consolidation is triggered only once for each node."""
                if node not in consolidation_node_list:
                    node.connections[self] = (sim, update_strength)
        self.step = 1

    def find_strength_knn(self, k=3):
        return sorted(self.connections.items(),
                      key=lambda x: (-x[1][1], -x[1][0]))[:k]

    def __repr__(self):
        return f"<SynapticNode {self.name} | Connections: {self.connection_count}>"


class SynapticNetwork:
    def __init__(self, args):
        self.nodes = {}
        self.args = args
        if self.args.dataset == 'ISRUC':
            self.sim_calculator = get_SimilarityCalculator(args=self.args, n_channels=6)
        if self.args.dataset == 'FACED':
            self.sim_calculator = get_SimilarityCalculator(args=self.args, n_channels=32)
        if self.args.dataset == 'BCI2000':
            self.sim_calculator = get_SimilarityCalculator(args=self.args, n_channels=64)

    def net_initialization(self, initial_info, initial_model_path, seed, similarity_threshold=0.6):
        """
        Synaptic Network Initilization
        :param initial_info: [[sub_name, sub_prototype, sub_data_path, sub_label_path]]
        :param initial_model_path: initial model path  .pkl
        :param similarity_threshold: similarity threshold
        :return:
        """
        for subject in initial_info:
            sub_name, sub_prototype = subject[0], subject[1]
            sub_data_path, sub_label_path = subject[2], subject[3]
            new_node = SynapticNode(sub_name, sub_prototype)
            self.nodes[sub_name] = new_node
            self.nodes[sub_name].sample_path = {'data': sub_data_path, 'label': sub_label_path}
            self.nodes[sub_name].model_path.append(initial_model_path + f'feature_extractor_parameter_{seed}.pkl')
            self.nodes[sub_name].model_path.append(initial_model_path + f'feature_encoder_parameter_{seed}.pkl')
            self.nodes[sub_name].model_path.append(initial_model_path + f'sleep_classifier_parameter_{seed}.pkl')

        for idx, node in enumerate(list(self.nodes.values())):
            for nt_node in list(self.nodes.values())[idx+1:]:
                sim = self.sim_calculator.domain_weighted_cosine(node.prototype, nt_node.prototype)
                if sim > similarity_threshold:
                    node.add_connection(nt_node, sim)

    def add_node(self, name, feature, args, similarity_threshold=0.6):
        """Add New Node to the Synaptic Network"""
        new_node = SynapticNode(name, feature)
        self.nodes[name] = new_node

        """Build synaptic connection with existing nodes"""
        for exist_node in self.nodes.values():
            if exist_node.name == name:
                continue

            sim = self.sim_calculator.domain_weighted_cosine(exist_node.prototype, feature)
            if sim > similarity_threshold:
                exist_node.add_connection(new_node, sim)

        """Enhance the synaptic strength of the pathways associated with the top-K most important connected nodes"""
        if self.nodes[name].connection_count():
            consolidation_nodes = [self.nodes[connect_sub[0]] for connect_sub in self.nodes[name].find_importance_knn(args)]
            for connected_node in consolidation_nodes:
                connected_node.consolidate_memory(consolidation_nodes, consolidation_factor=self.args.consolidation_factor)

    def global_renormalization_synapses(self):
        """Global Synaptic Renormalization"""
        for node in self.nodes.values():
            node.renormalization_synapses(decay_factor=self.args.decay_factor)

    def visualize_network(self):
        visualize_synaptic_net(self.nodes, self.args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synaptic-Inspired Continual Brain Decoding')
    parser.add_argument('--alpha_sim', type=float, default=0.6, help='importance coefficient of similarity')
    parser.add_argument('--alpha_str', type=float, default=0.4, help='importance coefficient of average strength')
    parser.add_argument('--k_factor', type=int, default=1, help='find importance knn')
    parser.add_argument('--dataset', type=str, default='FACED', help='dataset')
    parser.add_argument('--prototype', type=str, default='prototype3', help='training preserved threshold')
    parser.add_argument('--decay_factor', type=float, default=30, help='renormalization_synapses, Ebbinghaus S')
    parser.add_argument('--consolidation_factor', type=float, default=1.3, help='consolidate_memory')
    params = parser.parse_args()
    path = f"/data/xbb/Synaptic Homeostasis/dataset/{params.dataset}/{params.prototype}"

    source_domain = [7, 16, 18, 23, 24, 28, 30, 34, 35, 37, 38, 41, 45, 48, 50, 53, 69, 71, 74, 78, 79, 82, 93, 94, 12, 21, 29, 58, 76, 77]
    target_domain = [1, 2, 3, 4, 5, 9, 10, 11, 13, 15, 17, 19, 20, 22, 25, 26, 27, 31, 33, 36, 42, 46, 47, 49, 52, 54, 55, 57, 60, 61, 63, 64, 65, 66, 70, 80, 81, 83, 84, 85, 86, 87, 89, 91, 95, 96, 97, 98, 99]

    synaptic_net = SynapticNetwork(params)
    initial_info = []
    for source_id in source_domain:
        sub_name = f'Sub_{source_id}'
        sub_prototype = np.load(path + f"/{source_id}.npy")
        sub_data_path = None
        sub_label_path = None
        initial_info.append([sub_name, sub_prototype, sub_data_path, sub_label_path])
        synaptic_net.net_initialization(initial_info, f"dd", 4321, similarity_threshold=0.7)
    for sub in target_domain:
        prototype = np.load(path + f"/{sub}.npy")
        synaptic_net.add_node(f"Sub_{sub}", prototype, params, similarity_threshold=0.7)
        synaptic_net.global_renormalization_synapses()

    for nod in synaptic_net.nodes.values():
        print(nod.name, nod.connection_count(), nod.get_target_importance())
    synaptic_net.visualize_network()
