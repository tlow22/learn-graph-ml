# ==============================================================================
# This script will import a dataset from Pytorch Geometric and visualize it via 
# networkx and matplotlib.
# ==============================================================================

from networkx.algorithms import coloring
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx 
from matplotlib import cm
import torch
import networkx as nx
import matplotlib.pyplot as plt

# ==============================================================================
# HELPER FUNCTIONS 
# ==============================================================================
def visualize(h, color, epoch=None, loss=None):
    """
    Visualize the a networkx or Pytorch graph using duck typing  
    """
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h): 
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], c=color, s=140, cmap="Set2")
        if epoch is not None and loss is not None: 
            plt.xlabel(f"Epoch: {epoch}, Loss: {loss:.4f}", fontsize=16)
    else:
        nx.draw_networkx(h, 
                         pos=nx.spring_layout(h, seed=42),
                         with_labels=False,
                         node_color=color,
                         cmap="Set2")

    plt.show() 

# ==============================================================================
# MAIN SCRIPT 
# ==============================================================================
# import Zach Karate Club dataset
dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# convert PyG to networkx graph and visualize it 
G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)