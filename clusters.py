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


def sort_by_ratio(val):
    '''
    Function that count ratio of incomving edges / outgoing edges.
    '''
    if val[1] == 0 or val[2] == 0:
        return 1
    # assuming there is no statoin with zero trafic --> zero division
    return val[1]/val[2]


def calculate_degrees(df):
    '''
    Calculate input and output degree for each station.
    '''
    # dictionary where station id is key and value is list [count incoming edges, count leaving edges]
    stations = {}

    for index, row in df.iterrows():
        out_id = row['Departure station id']
        in_id = row['Return station id']
        # update outgoing edges count
        if out_id not in stations:
            stations[out_id] = [0, 1]
        else:
            stations[out_id][1] += 1
        # update incoming edges count
        if in_id not in stations:
            stations[in_id] = [1, 0]
        else:
            stations[in_id][0] += 1

    # create a list structure that can be sorted easily
    # list of list where each list is [station id, incoming bikes, outgoing bikes]
    station_data = []
    for key, value in stations.items():
        station_data.append([key, value[0], value[1]])

    # sort list by ratio
    station_data.sort(key=sort_by_ratio)
    
    # count 10 stations where ratio of incoming bikes / outgoing bikes is highes
    print("\nHighest ratios of incoming and outgoing edges")
    print("----------------------------------------------")
    for station in station_data[:10]:
        row = df[df['Departure station id'] == station[0]].iloc[0]
        print(str(row['Departure station name']) + ", edges_in: " + str(station[1]) + ", edges_out: " + str(station[2]))        

    # count 10 stations where ratio of incoming bikes / outgoing bikes is lowest
    print("\nLowest ratios of incoming and outgoing edges")
    print("----------------------------------------------")
    for station in station_data[-10:]:
        row = df[df['Departure station id'] == station[0]].iloc[0]
        print(str(row['Departure station name']) + ", edges_in: " + str(station[1]) + ", edges_out: " + str(station[2]))  