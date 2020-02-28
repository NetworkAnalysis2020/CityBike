
#!/usr/bin/python

# Import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def predict_destination(df, title):
    '''
    Function takes network data in dataframe form as input,
    uses RandomForestClassifier to predict the destination
    of the trip based on different variables, and calls
    the plotting function to plot the accuracy scores.
    @param dataframe: network data.
    '''
    Y = df['Return station id'].to_numpy().astype(int)
    departure_station = df['Departure station id'].to_numpy().astype(int).reshape(-1, 1)
    distance = df['Covered distance (m)'].to_numpy().reshape(-1, 1)
    duration = df['Duration (sec.)'].to_numpy().reshape(-1, 1)
    hour = df['Departure'].dt.hour.to_numpy().reshape(-1, 1)
    
    # Starting point, Hour, duration and length of the journey are stacked together
    all_variables = np.hstack((departure_station, distance, duration, hour))
    
    y_train, y_test = train_test_split(Y, test_size=0.6, random_state=0)
    
    # The number of trees in RandomForestClassifier is set to 10
    # classifier = RandomForestClassifier(n_estimators=10)
    scores = []
    
    for set in (departure_station, distance, duration, hour, all_variables):
        # Data is split into train and test sets, train set size being 60% of the data
        x_train, x_test = train_test_split(set, test_size=0.6, random_state=0)
        classifier.fit(x_train, y_train)
        # Model's prediction of the destinations
        y_pred = classifier.predict(x_test)
        # Accuracy score for the model's prediction
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
    
    plot_scores(scores, title)
    
    
def plot_scores(scores, title):
    '''
    Function for drawing a barplot of the prediction scores.
    @param list of scores
    '''
    
    models = ('Departure station', 'Distance', 'Duration', 'Hour', 'Combined')
    y_pos = np.arange(len(models))
    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, models)
    plt.title(title)
    plt.ylabel('Accuracy score')
    plt.show()