# ==============================================================================
# This script will simply walk through using networkx to 
#     - create a graph
#     - add nodes and edges
#     - visualize/print out the graph in terminal 
#     - demonstrate a pagerank example 
# ==============================================================================
import networkx as nx

# Create graph objects  
G1 = nx.Graph()                                                                 # undirected graph
G2 = nx.DiGraph()                                                               # directed graph

# Inform user 
print("G1 is_directed: ", G1.is_directed())
print("G2 is_directed: ", G2.is_directed())

# Add graph attributes (i.e. graph name) 
G1.graph["name"] = "my graph"                                                   # Set graph name
print(G1.graph)

# ==============================================================================
# ADDING/GETTING NODES
# ==============================================================================
# add nodes and get attributes 
G1.add_node(0, feature=0, label=0)
print("Node 0 has attributes {}".format(G1.nodes[0]))

# print all nodes and attributes via a for loop
#    * setting "data=True" reuturns node attributes
print("Printing all node attributes:")
for node in G1.nodes(data=True):
    print(node)

print("G has {} nodes".format(G1.number_of_nodes()))

# ==============================================================================
# ADDING/GETTING EDGES 
# ==============================================================================
G1.add_edge(0, 1, weight=0.5)
print("Edge connecting nodes 0 <--> 1 has attributes {}".format(G1.edges[0, 1]))

G1.add_edges_from([
    (1, 2, {"weight": 0.3}),
    (2, 0, {"weight": 0.1})
])

for edge in G1.edges(): 
    print(edge)

print("G1 has {} nodes".format(G1.number_of_nodes()))

# adding edges when nodes don't exist will automatically add the relevant nodes 
# reprint the node list and we'll see that the node list has now increased by 2
for node in G1.nodes(data=True):
    print(node)

# ==============================================================================
# VISUALIZATION
# ==============================================================================
# draw the graph
nx.draw(G1, with_labels=True)

# ==============================================================================
# NODE DEGREE AND NEIGHBOR
# ==============================================================================
node_id = 1 
print("Node {} has degree {}".format(node_id, G1.degree(node_id)))

# get neighbor of node 1 
for neighbor in G1.neighbors(node_id):
    print("Node {} is a neighbor of node {}".format(neighbor, node_id))


# ==============================================================================
# PAGERANK EXAMPLE
# ==============================================================================
nNodes = 4 
G = nx.DiGraph(nx.path_graph(nNodes))

pr = nx.pagerank(G, alpha=0.8)
print(pr)