#!/usr/bin/python

# Import necessary modules
import glob
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import itertools
from itertools import islice
from networkx.algorithms.community.centrality import girvan_newman
from networkx import edge_betweenness_centrality
from random import random
from matplotlib import cm
import community  # pip install python-louvain


def _validate_graph(graph):
    '''
    Function takes network graph as input and validates it.
    If invalid nodes are detected, they will be removed.
    @param graph: network graph.
    @return: validated graph.
    '''
    new_graph = graph
    nodes_to_remove = []
    for node in graph:
        if type(node) != str:
            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        new_graph.remove_node(node)
    return new_graph


def what_is_the_bounding_box(df):
    '''
    Print the bounding box for dataset.
    '''
    bbox = (df.x.min(), df.x.max(), df.y.min(), df.y.max())
    return bbox


def draw_stations_to_map(graph):
    '''
    Function for drawing stations on a map.
    @param graph
    '''
    # read spatial data from file
    df = pd.read_csv("./data/bikestations.csv")
    # print dataframe head
    print(df.head())
    # get min and max longitude latitude values
    bbox = what_is_the_bounding_box(df)
    print("Longitude min, Longitude max, Latitude min, Latitude max")
    print(bbox)
    # load map picture
    eh_map = plt.imread("./data/EspooHelsinki.PNG")
    # create figure with proper scale and background
    plt.figure()
    ax = plt.gca()
    ax.set_title('Bike stations spatial data')
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.imshow(eh_map, zorder=0, extent=bbox, aspect='auto')
    # draw stations to image
    ax.scatter(df.x, df.y, zorder=1, c='r', s=30)
    # show image
    plt.show()


def detect_communities(graph):
    '''
    Function to perform multiple community detection measurements.
    Uses Louvain algorithm. https://github.com/taynaud/python-louvain
    @param graph: the network to be analyzed.
    '''
    # fix errors in graph
    G = _validate_graph(graph)

    # first compute the best partition
    parts = community.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]
    layout = nx.spring_layout(G)

    # drawing the blank network graph
    plt.axis("off")
    nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap("jet"),
                     node_color=values, node_size=35, with_labels=False)
    plt.show()
