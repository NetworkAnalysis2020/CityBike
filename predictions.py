
#!/usr/bin/python

# Import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def predict_destination(df, title):
    test_df = df[df['Departure station name'] == 'Töölönlahdenkatu']
    Y = test_df['Return station id'].to_numpy().astype(int)
    distance = test_df['Covered distance (m)'].to_numpy().reshape(-1,1)
    duration = test_df['Duration (sec.)'].to_numpy().reshape(-1,1)
    hour = test_df['Departure'].dt.hour.to_numpy().reshape(-1,1)

    
    all_variables = np.hstack((distance, duration, hour)) # Hour, duration and length of the journey are stacked together

    y_train, y_test = train_test_split(Y, test_size=0.6, random_state=0)

    classifier = RandomForestClassifier(n_estimators = 10) # The number of trees in RandomForestClassifier is set to 10
    
    scores = []

    for set in (distance, duration, hour, all_variables):
        x_train, x_test = train_test_split(set, test_size=0.6, random_state=0) # Data is split into train and test sets, train set size being 60% of the data
        classifier.fit(x_train, y_train)   
        y_pred = classifier.predict(x_test) # Model's prediction of the destinations  
        acc = accuracy_score(y_test, y_pred) # Accuracy score for the model's prediction
        scores.append(acc)

    plot_scores(scores, title)


def plot_scores(scores, title):
    models = ('Distance', 'Duration', 'Hour', 'Combined')
    y_pos = np.arange(len(models))
    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, models)
    plt.title(title)
    plt.ylabel('Accuracy score')
    plt.show()
