#!/usr/bin/python

# Import necessary modules
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import glob
from communities import detect_communities

def import_data():
    # Imports data and concatenates csv:s into one dataframe
    df = pd.concat([pd.read_csv(f)
                    for f in glob.glob("./data/*.csv")], ignore_index=True)
    # Transforms timestamps into datetime format
    df.Departure = pd.to_datetime(df.Departure)
    df.Return = pd.to_datetime(df.Return)
    # drop rows without all values
    df.dropna()

    return df

def split_data(df):
    #Dataframe is reversed to chronological order
    df = df.iloc[::-1]
    #Date-column is created and set as index
    df['Date'] = df['Departure'].dt.date
    df = df.set_index("Date")
    #Two dataframes are created, one including only weekdays and other only weekends (saturday-sunday)
    wd = pd.date_range("2019-04-01", "2019-08-31", freq="b")
    weekdays=df.loc[wd,:]
    weekend = df.drop(wd.date)

    return weekdays, weekend

def create_network(df):
    # Transforms dataframe into a directed graph that accepts multiple parallel edges
    M = nx.from_pandas_edgelist(df, source='Departure station name', target='Return station name', edge_attr=[
                                'Covered distance (m)', 'Duration (sec.)'], create_using=nx.MultiDiGraph())

    # Creates a simple graph where edge weight expresses the number of parallel edges in multigraph
    G = nx.Graph()
    for u, v in M.edges():
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)

    return M, G


def nodes_and_edges(graph):
    print("Number of Nodes: {}, Number of Edges: {}".format(
        graph.number_of_nodes(), graph.number_of_edges()))


def centrality(graph):
    # Degree centrality
    nodes_degrees = nx.degree_centrality(graph)
    max_centrality = max(nodes_degrees, key=nodes_degrees.get)
    print("Node with most Edges: {}, Number of Edges: {}".format(
        max_centrality, graph.degree(max_centrality)))

    # Closeness centrality
    clos = nx.closeness_centrality(graph)
    s = sorted(clos.items(), key=itemgetter(1), reverse=True)
    print("Node with greatest closeness centrality: {}, Closeness centrality value: {}".format(
        s[0][0], s[0][1]))


def clustering_coefficient(graph):
    C = nx.average_clustering(graph)
    print("Average clustering coefficient: {}".format(C))


def most_popular_routes(graph, n=3):
    '''
    Function for finding n routes that are most often used.
    @param n number of routes to find.
    '''
    # Use the weights of the simple graph to create a sorted tuple with station names and number of trips
    popular_routes = sorted(graph.edges(data=True),
                            key=lambda x: x[2]['weight'], reverse=True)

    # Show the results
    print(str(n) + " most popular routes:")
    for i in range(n):
        print("Stations: {} - {}, Number of trips: {}".format(
            popular_routes[i][0], popular_routes[i][1], popular_routes[i][2]['weight']))

def plot_days(weekday, weekend):
    #The number of trips per hour is plotted separately for weekends and for weekdays
    plt.figure()
    plt.subplot(211)
    weekday.groupby(weekday['Departure'].rename('Hours').dt.hour).size().plot()
    plt.title('Weekdays')

    plt.subplot(212)
    weekend.groupby(weekend['Departure'].rename('Hours').dt.hour).size().plot()
    plt.title('Weekends')

    plt.show()

def count_averages(df, wd, wnd):
    avg_time = (df["Duration (sec.)"].mean(axis=0))/60 
    avg_distance = df["Covered distance (m)"].mean(axis=0) 

    avg_time_wd = (wd["Duration (sec.)"].mean(axis=0))/60 
    avg_distance_wd = wd["Covered distance (m)"].mean(axis=0) 

    avg_time_wnd = (wnd["Duration (sec.)"].mean(axis=0))/60 
    avg_distance_wnd = wnd["Covered distance (m)"].mean(axis=0)

    print("Average trip duration: {:0.2f} min, Average trip length: {:0.2f} m".format(
        avg_time, avg_distance))
    print("Average trip duration during weekdays: {:0.2f} min, Average trip length during weekdays: {:0.2f} m".format(
        avg_time_wd, avg_distance_wd))
    print("Average trip duration during weekends: {:0.2f} min, Average trip length during weekdays: {:0.2f} m".format(
        avg_time_wnd, avg_distance_wnd))

    labels = ('Total', 'Weekdays', 'Weekend')
    y_pos = np.arange(len(labels))
    time_values = (avg_time, avg_time_wd, avg_time_wnd)
    distance_values = (avg_distance, avg_distance_wd, avg_distance_wnd)

    plt.subplot(121)
    plt.title('Average trip duration')
    plt.bar(y_pos, time_values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Minutes')

    plt.subplot(122)
    plt.title('Average trip length')
    plt.bar(y_pos, distance_values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Meters')

    plt.show()

def main():
    df = import_data()
    wd, wnd = split_data(df)
    count_averages(df, wd, wnd)
    plot_days(wd, wnd)    
    M, G = create_network(df)
    WM, WG = create_network(wd)
    WNM, WNG = create_network(wnd)
    centrality(M)
    nodes_and_edges(M)
    clustering_coefficient(G)
    most_popular_routes(G, 5)
    most_popular_routes(WG, 5)
    most_popular_routes(WNG, 5)
    detect_communities(G)
    detect_communities(WG)
    detect_communities(WNG)

if __name__ == '__main__':
    main()
