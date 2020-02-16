#!/usr/bin/python

# Import necessary modules
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import glob
from communities import detect_communities, draw_stations_to_map, draw_stations_and_edges_to_map, draw_popular_routes_to_map
from predictions import predict_destination
from clusters import calculate_degrees

stations = {}


def import_data():
    global stations
    # Imports data and concatenates csv:s into one dataframe
    df = pd.concat([pd.read_csv(f)
                    for f in glob.glob("./data/*.csv")], ignore_index=True)
    # Transforms timestamps into datetime format
    df.Departure = pd.to_datetime(df.Departure)
    df.Return = pd.to_datetime(df.Return)
    # Drop rows without all values
    df.dropna()
    #df['Departure station id'] = df['Departure station id'].astype(float)
    # Creates a dict with the station id:s as keys and station names as values
    #stations = dict(zip(df['Departure station id'], df['Departure station name']))

    return df


def split_data(df):
    # Dataframe is reversed to chronological order
    df = df.iloc[::-1]
    # Date-column is created and set as index
    df['Date'] = df['Departure'].dt.date
    df = df.set_index("Date")
    # Two dataframes are created, one including only weekdays and other only weekends (saturday-sunday)
    wd = pd.date_range("2019-04-01", "2019-08-31", freq="b")
    weekdays = df.loc[wd, :]
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

    print("Number of Nodes: {}, Number of Edges: {}".format(
        graph.number_of_nodes(), graph.number_of_edges()))


def centrality(graph):
    # Average degree
    degrees = graph.degree()
    sum_of_edges = sum([pair[1] for pair in degrees])
    avg_degree = sum_of_edges / graph.number_of_nodes()
    print("Average degree of the graph: {}".format(int(round(avg_degree))))

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
    
    # Plot node degree values
    degrees = [d for n,d in graph.degree()]
    y = sorted(degrees)
    x = range(len(degrees))
    plt.plot(x, y)
    plt.ylabel('Node degree')
    plt.show()
    
    # Plot degree centrality values
    d = sorted(nodes_degrees.values())
    y = d
    x = range(len(d))
    plt.plot(x, y)
    plt.ylabel('Degree centrality')

    plt.show()


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
    
    # return the n most popular routes
    return popular_routes[:n]


def plot_days(weekday, weekend):
    # Counts the number of unique days in each dataframe 
    wd = len(weekday['Departure'].dt.date.unique())
    wnd = len(weekend['Departure'].dt.date.unique())

    # The average daily number of trips per hour is plotted separately for weekends and for weekdays
    plt.figure()
    plt.subplot(211)
    (weekday.groupby(weekday['Departure'].dt.hour).size() / wd).plot()
    plt.xlabel('Hour of departure')
    plt.title('Weekdays')

    plt.subplot(212)
    (weekend.groupby(weekend['Departure'].dt.hour).size() / wnd).plot(color='r')
    plt.xlabel('Hour of departure')
    plt.title('Weekends')

    plt.tight_layout()
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

    # Creates barplots for average trip duration and trip length
    plt.subplot(421)
    plt.title('Average trip duration')
    plt.bar(y_pos, time_values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    for index, data in enumerate(time_values):
        plt.text(x=index, y=data-3,
                 s=f"{data:0.2f}", fontdict=dict(fontsize=8))
    plt.ylabel('Minutes')

    plt.subplot(422)
    plt.title('Average trip length')
    plt.bar(y_pos, distance_values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    for index, data in enumerate(distance_values):
        plt.text(x=index, y=data-400,
                 s=f"{data:0.2f}", fontdict=dict(fontsize=8))
    plt.ylabel('Meters')

    # Plots hourly average trip duration and trip length for the whole data, weekdays and weekends
    plt.subplot(423)
    plt.title('Average trip distance by hour')
    df.groupby(df['Departure'].dt.hour)[
        'Covered distance (m)'].mean().plot(color='r')
    plt.ylabel('Meters')
    plt.xlabel('Hour')

    plt.subplot(424)
    plt.title('Average trip duration by hour')
    df.groupby(df['Departure'].dt.hour)[
        'Duration (sec.)'].mean().div(60).plot(color='r')
    plt.ylabel('Minutes')
    plt.xlabel('Hour')

    plt.subplot(425)
    plt.title('Average trip distance by hour on weekdays')
    wd.groupby(wd['Departure'].dt.hour)[
        'Covered distance (m)'].mean().plot(color='y')
    plt.ylabel('Meters')
    plt.xlabel('Hour')

    plt.subplot(426)
    plt.title('Average trip duration by hour on weekdays')
    wd.groupby(wd['Departure'].dt.hour)[
        'Duration (sec.)'].mean().div(60).plot(color='y')
    plt.ylabel('Minutes')
    plt.xlabel('Hour')

    plt.subplot(427)
    plt.title('Average trip distance by hour on weekends')
    wnd.groupby(wnd['Departure'].dt.hour)[
        'Covered distance (m)'].mean().plot(color='g')
    plt.ylabel('Meters')
    plt.xlabel('Hour')

    plt.subplot(428)
    wnd.groupby(wnd['Departure'].dt.hour)[
        'Duration (sec.)'].mean().div(60).plot(color='g')
    plt.title('Average trip duration by hour on weekends')
    plt.ylabel('Minutes')
    plt.xlabel('Hour')

    plt.tight_layout()
    plt.show()


def main():
    df = import_data()
    #wd, wnd = split_data(df)
    #plot_days(wd, wnd)
    #predict_destination(df, 'All days')
    calculate_degrees(df)
    '''
    predict_destination(wd, 'Weekdays')
    predict_destination(wnd, 'Weekends')
    count_averages(df, wd, wnd)
    
    M, G = create_network(df)
    WM, WG = create_network(wd)
    WNM, WNG = create_network(wnd)
    centrality(M)
    nodes_and_edges(M)
    clustering_coefficient(G)
    popular_routes = most_popular_routes(G, 5)
    draw_popular_routes_to_map(popular_routes)
    most_popular_routes(WG, 5)
    most_popular_routes(WNG, 5)
    detect_communities(G)
    detect_communities(WG)
    detect_communities(WNG)
    draw_stations_to_map(G)
    #draw_stations_and_edges_to_map(G) # takes a while
    '''

if __name__ == '__main__':
    main()
