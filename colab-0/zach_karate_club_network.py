# ==============================================================================
# This script will 
#   - import a dataset from Pytorch Geometric and visualize it via 
#     networkx and matplotlib.
#   - define a graph convolutional model with Pytorch Geometric.
#   - train the model on the data.
# ==============================================================================
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch 
import time

from networkx.algorithms import coloring
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx 
from matplotlib import cm
from torch.nn import Linear
from torch_geometric.nn import GCNConv


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
# MAIN SCRIPT Part I: Import and visualize graph dataset 
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

# ==============================================================================
# MAIN SCRIPT Part II: Perform training 
# ==============================================================================
# define a graph convolutional network class 
class GCN(torch.nn.Module): 
    # define 3 stacks of graph convolutional layers which maps 
    # input features --> 4 --> 4 --> 2
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4) # notice the matrix multiplication chaining in sizes
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    # enhance each graph convolutional layer with a 'tanh' non-linearity 
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        # Apply a final (linear) classifier
        out = self.classifier(h)

        return out, h

# initialize the model and visualize the initial guess. One really cool thing to 
# note is that graph neural networks have an inductive bias, meaning that even 
# though the network is not yet trained, our visualize function shows a somewhat 
# intuitive classification of the nodes in the graph. 
model = GCN()
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')
visualize(h, color=data.y)

# define loss, optimizer, and training functions
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(data): 
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)                                     # single forward pass 
    loss   = criterion(out, data.y)                                             # compute loss
    loss.backward()                                                             # backpropagate for gradients
    optimizer.step()                                                            # update weights
    return loss, h

# train the model
epochs = 401
for epoch in range(epochs):
    loss, h = train(data)                                                       
    
visualize(h, color=data.y, epoch=epoch, loss=loss)