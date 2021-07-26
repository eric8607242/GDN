import os

import torch

from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sb

class LoggingTracker:
    def __init__(self, writer, config, title):
        self.writer = writer
        self.config = config
        self.title = title

        self.track_list = []

    def step(self, out, y):
        distances = (y - out)**2
        distances = torch.mean(distances, dim=0)

        self.track_list.append(distances)


    def record(self, test_alpha_weights_list, test_edge_index, node_similarity, epoch):
        track_list = torch.stack(self.track_list)
        track_list = torch.mean(track_list, dim=0)
        track_list = track_list.detach().clone().cpu().tolist()
        for i, track_feature in enumerate(track_list):
            self.writer.add_scalar(f"Train/_feature_{i}/", track_feature, epoch)

        self._record_graph(test_alpha_weights_list, test_edge_index, epoch)
        self._record_node_similarity(node_similarity, epoch)

    def _record_node_similarity(self, node_similarity, epoch):
        fig, ax = plt.subplots(figsize=(11, 9))
        sb.heatmap(node_similarity.numpy())

        self.writer.add_figure("Test/_node_similarity", fig, epoch)


    def _record_graph(self, alpha_weights, edge_indexs, epoch):
        alpha_weights = alpha_weights.numpy()
        edge_indexs = edge_indexs.numpy()

        G = Network()

        # Add nodes
        for i in range(self.config["dataset"]["feature_num"]):
            G.add_node(i)

        # Add edges
        for i in range(edge_indexs.shape[1]):
            e1, e2 = edge_indexs[:, i]
            alpha = alpha_weights[i]
            G.add_edge(int(e1), int(e2), title=str(alpha), weight=float(alpha))

        G.show_buttons(filter_=['physics'])
        G.show(os.path.join(self.config["logs_path"]["logger_path"], self.title, f"graph_{epoch}.html"))


