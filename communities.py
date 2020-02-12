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


def __what_is_the_bounding_box(df):
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
    bbox = __what_is_the_bounding_box(df)
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


def draw_stations_and_edges_to_map(graph):
    '''
    Function for drawing stations and edges between them to map.
    '''
    # read spatial data from file
    df = pd.read_csv("./data/bikestations.csv")
    # get min and max longitude latitude values
    bbox = __what_is_the_bounding_box(df)
    # load map picture
    eh_map = plt.imread("./data/EspooHelsinki.PNG")
    # create figure with proper scale and background
    plt.figure()
    ax = plt.gca()
    ax.set_title('Bike station network')
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.imshow(eh_map, zorder=0, extent=bbox, aspect='auto')
    # draw edges between stations
    for edge in graph.edges:
        start = edge[0]
        end = edge[1]
        row_start = df.loc[df['Nimi'] == start]
        row_end = df.loc[df['Nimi'] == end]
        if row_start.empty or row_end.empty:
            # did not found stations --> skip
            print("Could not find row for one of the two stations: " + str(start) + ", " + str(end))
            continue
        x1 = row_start.iloc[0]["x"]
        x2 = row_end.iloc[0]["x"]
        y1 = row_start.iloc[0]["y"]
        y2 = row_end.iloc[0]["y"]
        ax.plot([x1, x2], [y1, y2], 'k-', zorder=1, alpha=0.1)
        # draw stations to image
    ax.scatter(df.x, df.y, zorder=2, c='r', s=30)
    # show image
    plt.show()


def draw_popular_routes_to_map(routes):
    '''
    Function for drawing the most popular routes to map.
    @param graph
    '''
    # read spatial data from file
    df = pd.read_csv("./data/bikestations.csv")
    # get min and max longitude latitude values
    bbox = __what_is_the_bounding_box(df)
    # load map picture
    eh_map = plt.imread("./data/EspooHelsinki.PNG")
    # create figure with proper scale and background
    plt.figure()
    ax = plt.gca()
    ax.set_title('Bike station network')
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.imshow(eh_map, zorder=0, extent=bbox, aspect='auto')
    # draw popular routes and stations
    for route in routes:
        start = route[0]
        end = route[1]
        row_start = df.loc[df['Nimi'] == start]
        row_end = df.loc[df['Nimi'] == end]
        if row_start.empty or row_end.empty:
            # did not found stations --> skip
            print("Could not find row for one of the two stations: " + str(start) + ", " + str(end))
            continue
        x1 = row_start.iloc[0]["x"]
        x2 = row_end.iloc[0]["x"]
        y1 = row_start.iloc[0]["y"]
        y2 = row_end.iloc[0]["y"]
        ax.plot([x1, x2], [y1, y2], 'b-', zorder=1, linewidth=7)
        # draw stations to image
        ax.scatter([x1, x2], [y1, y2], zorder=2, c='r', s=30)
    # show image
    plt.show()        


def __draw_communities_to_map(graph, values):
    '''
    Function for drawing communities to map.
    '''
    # read spatial data from file
    df = pd.read_csv("./data/bikestations.csv")
    # print dataframe head
    print(df.head())
    # get min and max longitude latitude values
    bbox = __what_is_the_bounding_box(df)
    # load map picture
    eh_map = plt.imread("./data/EspooHelsinki.PNG")
    # create figure with proper scale and background
    plt.figure()
    ax = plt.gca()
    ax.set_title('Bike station communities')
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.imshow(eh_map, zorder=0, extent=bbox, aspect='auto')
    # draw stations to image
    x_axis = []
    y_axis = []
    colors = []
    colorlist = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    index = 0
    for station in graph:
        row = df.loc[df['Nimi'] == station]
        if row.empty:
            print("could not find station " +
                  station + " from bikestations.csv")
            x_axis.append(-1)
            y_axis.append(-1)
            colors.append('w')
        else:
            x_axis.append(row.iloc[0]["x"])
            y_axis.append(row.iloc[0]["y"])
            colors.append(colorlist[values[index]])
        index += 1
    ax.scatter(x_axis, y_axis, zorder=1, c=colors, s=40)
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

    # draw the detected community structure to map
    __draw_communities_to_map(G, values)
