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
import math

'''
This file holds functionality to find the stations 
that are furthest apart from each other.
'''

# read spatial data from file
df = pd.read_csv("./data/bikestations.csv")

station_1 = ""
station_2 = ""
d_max = 0
cx1 = 0
cx2 = 0
cy1 = 0
cy2 = 0

for row1 in df.itertuples():
    x1 = row1.x
    y1 = row1.y
    nimi1 = row1.Nimi
    for row2 in df.itertuples():
        x2 = row2.x
        y2 = row2.y
        nimi2 = row2.Nimi

        d = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))

        if d > d_max:
            d_max = d
            station_1 = nimi1
            station_2 = nimi2
            cx1 = x1
            cx2 = x2
            cy1 = y1
            cy2 = y2

print("Results")
print("-------------------------")
print("Maximum distance between stations: " + str(d_max))
print("Station 1: " + station_1)
print("X: " + str(cx1))
print("Y: " + str(cy1))
print("Station 2: " + station_2)
print("X: " + str(cx2))
print("Y: " + str(cy2))