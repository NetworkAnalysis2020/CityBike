#!/usr/bin/python

#Import necessary modules
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import glob

#Imports data and concatenates csv:s into one dataframe
df = pd.concat([pd.read_csv(f) for f in glob.glob("./data/*.csv")], ignore_index = True)

#Transforms dataframe into a directed graph
def load_network():
    G = nx.from_pandas_edgelist(df, source='Departure station name', target='Return station name', edge_attr=['Covered distance (m)','Duration (sec.)'], create_using=nx.DiGraph())
    return G

def nodes_and_edges(graph):
    print("Number of Nodes: {}, Number of Edges: {}".format(graph.number_of_nodes(), graph.number_of_edges()))

def centrality(graph):
    #Degree centrality
    nodes_degrees = nx.degree_centrality(graph)
    max_centrality = max(nodes_degrees, key=nodes_degrees.get)
    print("Node with most Edges: {}, Number of Edges: {}".format(max_centrality, graph.degree(max_centrality)))

    #Closeness centrality
    clos = nx.closeness_centrality(graph)
    s = sorted(clos.items(), key=itemgetter(1), reverse=True)
    print("Node with greatest closeness centrality: {}, Closeness centrality value: {}".format(s[0][0], s[0][1]))

def clustering_coefficient(graph):
    #Removes self loops from graph
    graph.remove_edges_from(nx.selfloop_edges(graph))
    C = nx.average_clustering(graph)
    print("Average clustering coefficient: {}".format(C))

def main():
    G = load_network()
    centrality(G)
    nodes_and_edges(G)
    clustering_coefficient(G)

if __name__ == '__main__':
    main()


