import argparse
import pandas as pd
import numpy as np
import os
import re

def gaussian_calculations(df, feat_name, feat_val, Y, label):
    df = df[df[Y] == label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    if std > 0:
        p_x_given_y = (1/np.sqrt(2* np.pi) * std) * np.exp(-((feat_val - mean)**2)/(2*std**2))
        return p_x_given_y
    else:
        return 1

# y = weather EX sunny
def catagory_calculations(df, x_value, X, Y, y_val):
    y = df[Y]
    x = df[X]
    total = np.sum(y == y_val)
    likelihood_val = np.sum((y == y_val) & (x == x_value))
    return (likelihood_val + 1) / (total + 1)

def y_calculations(df, Y):
    y_counts = df[Y].value_counts()
    total_count = len(df[Y])
    y_possibilities = df[Y].unique()

    likelihood = (y_counts + 1) / (total_count + len(y_possibilities))
    
    return likelihood.to_dict()

def fetching_lastday(X):
    lastday_vals = X.iloc[-1].to_dict()
    if 'time' in lastday_vals:
        lastday_vals['time'] = int(lastday_vals['time']) + 600
        if lastday_vals['time'] > 1800:
            lastday_vals['time'] = 0
    return lastday_vals


def native_bayes(df, X_test, Y, categorical_features, numerical_features):
    lastday = fetching_lastday(X_test)
    weathers = df[Y].unique()
    y_prob = y_calculations(df, Y)

    y = 0
    highest_prob = 0 
    prediction = ''
    for w in weathers:
        y = y_prob[w]
        for category in categorical_features:
            y *= catagory_calculations(df, lastday[category], category, Y, w)
        
        for feature in numerical_features:
            y *= gaussian_calculations(df, feature, lastday[feature], Y,  w)
        
        if y > highest_prob and y > 0:
            prediction = w
            highest_prob = y
    return prediction
    

# Y is the expected column while X is the current column 
def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0

def main():
    parser = argparse.ArgumentParser(description='Training file and test directory')
    parser.add_argument('training', type=str, help='Path to the training file (Excel format)')
    parser.add_argument('test', type=str, help='Path to the test file (Excel format)')
    
    # Reading training data
    args = parser.parse_args()
    df = pd.read_excel(args.training)
    categorical_features = ['time', 'wind_dir', 'precip', 'cloudcover', 'visibility']
    numerical_features = ['dewpoint', 'pressure', 'dewpoint', 'uv_index', 'temperature']
    Y = df['weather_descriptions']

    # Reading test data)

    Y = 'weather_descriptions'
    # X = 'wind_dir'
    # x = 'SSE'
    # result = catagory_calculations(df, x , X, Y, 'Cloudy')
    # result = gaussian_calculations(df, 'uv_index', 1, Y, 'Sunny')

    test_files = os.listdir(args.test)
    sorted_files = sorted(test_files, key=extract_number)

    for file in sorted_files:
        print(file)
        path = os.path.join(args.test, file)
        X_test = pd.read_excel(path)
        prediction = native_bayes(df, X_test, Y, categorical_features, numerical_features)
        print(prediction)
    
    return 1



if __name__ == "__main__":
    main()
