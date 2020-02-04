#!/usr/bin/python

#Import necessary modules
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import glob


def import_data():
    #Imports data and concatenates csv:s into one dataframe
    df = pd.concat([pd.read_csv(f) for f in glob.glob("./data/*.csv")], ignore_index = True)
    #Transforms timestamps into datetime format
    df.Departure = pd.to_datetime(df.Departure)
    df.Return = pd.to_datetime(df.Return)

    return df

def create_network(df):
    #Transforms dataframe into a directed graph that accepts multiple parallel edges
    M = nx.from_pandas_edgelist(df, source='Departure station name', target='Return station name', edge_attr=['Covered distance (m)','Duration (sec.)'], create_using=nx.MultiDiGraph())

    #Creates a simple graph where edge weight expresses the number of parallel edges in multigraph
    G = nx.Graph()
    for u,v in M.edges():
        if G.has_edge(u,v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)

    return M, G

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
    C = nx.average_clustering(graph)
    print("Average clustering coefficient: {}".format(C))

def most_popular_routes(graph, n=3):
    '''
    Function for finding n routes that are most often used.
    @param n number of routes to find
    '''
    #Use the weights of the simple graph to create a sorted tuple with station names and number of trips
    popular_routes = sorted(graph.edges(data=True),key=lambda x: x[2]['weight'],reverse=True)

    #Show the results
    print(str(n) + " most popular routes:")
    for i in range(n):
      print("Stations: {} - {}, Number of trips: {}".format(popular_routes[i][0], popular_routes[i][1], popular_routes[i][2]['weight']))


def main():
    df = import_data()
    M, G = create_network(df)
    centrality(M)
    nodes_and_edges(M)
    clustering_coefficient(G)
    most_popular_routes(G, 5)

if __name__ == '__main__':
    main()
