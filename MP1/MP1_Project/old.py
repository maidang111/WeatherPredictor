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

def y_calculations(df, Y, X):
    y_counts = df[Y].value_counts()
    total_count = len(df[Y])
    y_possibilities = df[Y].unique()

    likelihood = (y_counts + 1) / (total_count + len(y_possibilities))
    
    return likelihood.to_dict()

def fetching_lastday(X, day):
    day = day -29
    last_day_time = X.iloc[-1]['time']
    lastday_vals = X.iloc[day].to_dict()
    if 'time' in lastday_vals:
        lastday_vals['time'] = int(last_day_time) + 600
        if lastday_vals['time'] > 1800:
            lastday_vals['time'] = 0
    return lastday_vals

def most_frequent(List):
    unique, counts = np.unique(List, return_counts=True)
    index = np.argmax(counts)
    return unique[index]

def native_bayes(df, X_test, Y, categorical_features, numerical_features):
    weathers = df[Y].unique()
    y_prob = y_calculations(df, Y, X_test)

    predictions = []
    for day in range(5):
        date = 28 - day
        lastday = fetching_lastday(X_test, date)
        highest_prob = 0 
        prediction = ''
        y = 0
        for w in weathers:
            y = y_prob[w]
            for category in categorical_features:
                y *= catagory_calculations(df, lastday[category], category, Y, w)
            
            for feature in numerical_features:
                y *= gaussian_calculations(df, feature, lastday[feature], Y,  w)
            
            if y > highest_prob:
                prediction = w
                highest_prob = y
        
        # if (highest_prob > 2e-5) and date == 28:
        #     print(prediction)
        #     return prediction
        
        predictions.append(prediction)
        print(predictions)

    return most_frequent(predictions)
    

# Y is the expected column while X is the current column 
def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0

def main():
    parser = argparse.ArgumentParser(description='Training file and test directory')
    parser.add_argument('training', type=str, help='Path to the training file (Excel format)')
    parser.add_argument('test', type=str, help='Path to the test file (Excel format)')

    args = parser.parse_args()
    
    # Reading training data
    df = pd.read_excel(args.training)
    categorical_features = ['time', 'wind_dir', 'precip', 'cloudcover', 'visibility']
    # dewpoint/3
    numerical_features = ['pressure', 'temperature', 'dewpoint', 'dewpoint', 'dewpoint', 'uv_index', 'uv_index', 'uv_index']
    X = df.drop('weather_descriptions', axis=1)
    Y = df['weather_descriptions']

    # Reading test data)

    Y = 'weather_descriptions'
    # X = 'wind_dir'
    # x = 'SSE'
    # result = catagory_calculations(df, x , X, Y, 'Cloudy')
    # result = gaussian_calculations(df, 'uv_index', 1, Y, 'Sunny')

    test_files = os.listdir(args.test)

    sorted_files = sorted(test_files, key=extract_number)
    predictions = []
    for file in sorted_files:
        print(file)
        path = str(args.test + '/' + file)
        X_test = pd.read_excel(path)
        predictions.append(native_bayes(df, X_test, Y, categorical_features, numerical_features))
    
    correct_prediction = ["Cloudy", "Sunny", "Sunny", "Partly cloudy", "Clear", "Sunny", "Clear", "Sunny", "Partly cloudy", "Sunny", "Patchy rain possible", "Sunny", "Overcast", "Cloudy", "Sunny", "Cloudy", "Clear", "Clear", "Clear", "Sunny", "Patchy rain possible", "Partly cloudy", "Clear", "Sunny", "Sunny", "Sunny", "Clear", "Partly cloudy", "Clear", "Clear", "Overcast", "Sunny", "Sunny", "Partly cloudy", "Partly cloudy", "Sunny", "Sunny", "Cloudy", "Clear", "Cloudy", "Sunny", "Sunny", "Sunny", "Clear", "Partly cloudy", "Sunny", "Moderate rain at times", "Sunny", "Overcast", "Clear", "Sunny", "Partly cloudy", "Sunny", "Clear", "Clear", "Clear", "Sunny", "Clear", "Cloudy", "Clear", "Partly cloudy", "Overcast", "Clear", "Partly cloudy", "Sunny", "Sunny", "Clear", "Clear", "Overcast", "Clear", "Clear", "Sunny", "Cloudy", "Clear", "Clear", "Sunny", "Cloudy", "Sunny", "Sunny", "Partly cloudy", "Partly cloudy", "Clear", "Clear", "Partly cloudy", "Partly cloudy", "Sunny", "Sunny", "Clear", "Patchy rain possible", "Sunny", "Sunny", "Partly cloudy", "Clear", "Sunny", "Partly cloudy", "Clear", "Cloudy", "Patchy rain possible", "Clear", "Sunny", "Overcast", "Clear", "Overcast", "Sunny", "Clear", "Overcast", "Heavy rain at times", "Clear", "Sunny", "Moderate or heavy rain shower", "Clear", "Sunny", "Clear", "Clear", "Clear", "Clear", "Overcast", "Sunny", "Overcast", "Sunny", "Partly cloudy", "Sunny", "Sunny", "Cloudy", "Patchy rain possible", "Clear", "Sunny", "Sunny", "Clear", "Patchy rain possible", "Clear", "Sunny", "Moderate rain at times", "Overcast", "Clear", "Patchy rain possible", "Clear", "Partly cloudy", "Overcast", "Sunny", "Sunny", "Clear", "Clear", "Sunny", "Patchy moderate snow", "Clear", "Clear", "Clear", "Sunny", "Clear", "Patchy rain possible", "Cloudy", "Sunny", "Partly cloudy", "Clear", "Patchy rain possible", "Overcast", "Clear", "Sunny", "Heavy rain at times", "Sunny", "Overcast", "Patchy rain possible", "Overcast", "Clear", "Clear", "Clear", "Partly cloudy", "Clear", "Partly cloudy", "Overcast", "Partly cloudy", "Sunny", "Patchy rain possible", "Cloudy", "Clear", "Clear", "Clear", "Sunny", "Sunny", "Partly cloudy", "Clear", "Clear", "Clear", "Patchy rain possible", "Clear", "Clear", "Sunny", "Clear", "Cloudy", "Clear", "Clear", "Patchy rain possible", "Clear", "Overcast", "Partly cloudy", "Sunny", "Partly cloudy", "Sunny", "Sunny", "Sunny", "Clear", "Partly cloudy", "Clear", "Sunny", "Cloudy", "Clear", "Sunny", "Sunny", "Clear", "Cloudy", "Partly cloudy", "Moderate rain at times", "Clear", "Clear", "Partly cloudy", "Patchy rain possible", "Sunny", "Clear", "Sunny", "Sunny", "Sunny", "Cloudy", "Clear", "Partly cloudy", "Partly cloudy", "Sunny", "Patchy rain possible", "Partly cloudy", "Partly cloudy", "Clear", "Clear", "Clear", "Sunny", "Overcast", "Clear", "Partly cloudy", "Clear", "Sunny", "Partly cloudy", "Clear", "Partly cloudy", "Partly cloudy", "Clear", "Clear", "Cloudy", "Clear", "Clear", "Moderate or heavy rain shower", "Sunny", "Cloudy", "Clear", "Partly cloudy", "Cloudy", "Patchy rain possible", "Heavy rain at times", "Partly cloudy", "Clear", "Sunny", "Clear", "Clear", "Sunny", "Sunny", "Clear", "Partly cloudy", "Cloudy", "Cloudy", "Moderate rain at times", "Moderate or heavy rain shower", "Clear", "Partly cloudy", "Clear", "Sunny", "Clear", "Partly cloudy", "Sunny", "Clear", "Cloudy", "Sunny", "Sunny", "Sunny", "Clear", "Partly cloudy", "Clear", "Cloudy", "Sunny", "Partly cloudy", "Sunny", "Patchy rain possible", "Partly cloudy", "Clear", "Partly cloudy", "Cloudy", "Clear", "Clear", "Sunny", "Sunny", "Sunny", "Sunny", "Clear", "Clear", "Cloudy", "Overcast", "Sunny", "Partly cloudy", "Sunny", "Sunny", "Sunny", "Sunny", "Sunny", "Overcast", "Sunny", "Cloudy", "Sunny", "Sunny", "Patchy rain possible", "Partly cloudy", "Clear", "Clear", "Clear", "Partly cloudy", "Overcast", "Clear", "Sunny", "Moderate or heavy rain shower", "Cloudy", "Heavy rain at times", "Patchy rain possible", "Clear", "Sunny", "Partly cloudy", "Cloudy", "Cloudy", "Clear", "Clear", "Sunny", "Sunny", "Clear", "Partly cloudy", "Cloudy", "Clear", "Cloudy", "Sunny", "Patchy rain possible", "Cloudy", "Sunny", "Partly cloudy", "Clear", "Sunny", "Sunny", "Patchy rain possible", "Partly cloudy", "Sunny", "Sunny", "Moderate or heavy rain shower", "Clear", "Sunny", "Overcast", "Sunny", "Overcast", "Partly cloudy", "Patchy rain possible", "Partly cloudy", "Overcast", "Overcast", "Clear", "Clear", "Clear", "Sunny", "Clear", "Sunny", "Clear", "Clear", "Overcast", "Sunny", "Clear", "Sunny", "Clear", "Clear", "Sunny", "Sunny", "Sunny", "Cloudy", "Partly cloudy", "Sunny", "Clear", "Partly cloudy", "Cloudy", "Cloudy", "Clear", "Clear", "Sunny", "Moderate or heavy rain shower", "Patchy rain possible", "Clear", "Heavy rain at times", "Sunny", "Sunny", "Clear", "Sunny", "Overcast", "Sunny", "Sunny", "Sunny", "Clear", "Patchy rain possible", "Cloudy", "Sunny", "Clear", "Sunny", "Clear", "Sunny", "Partly cloudy", "Sunny", "Clear", "Overcast", "Sunny", "Clear", "Overcast", "Cloudy", "Cloudy", "Partly cloudy", "Moderate rain at times", "Clear", "Moderate rain at times", "Sunny", "Clear", "Sunny", "Clear", "Clear", "Partly cloudy", "Sunny", "Sunny", "Sunny", "Clear", "Clear", "Sunny", "Overcast", "Clear", "Overcast", "Overcast", "Sunny", "Cloudy", "Overcast", "Clear", "Clear", "Sunny", "Sunny", "Clear", "Partly cloudy", "Cloudy", "Heavy rain at times", "Sunny", "Sunny", "Clear", "Sunny", "Cloudy", "Sunny", "Overcast", "Cloudy", "Patchy rain possible", "Overcast", "Cloudy", "Sunny", "Overcast", "Cloudy", "Clear", "Sunny", "Overcast", "Clear", "Sunny", "Sunny", "Clear", "Clear", "Cloudy", "Clear", "Overcast", "Sunny", "Sunny", "Cloudy", "Partly cloudy", "Overcast", "Overcast", "Sunny", "Sunny", "Sunny", "Clear", "Sunny", "Partly cloudy", "Moderate or heavy rain shower", "Heavy rain at times", "Partly cloudy", "Cloudy", "Sunny", "Sunny", "Clear", "Partly cloudy", "Sunny", "Partly cloudy", "Sunny", "Moderate rain at times", "Overcast", "Partly cloudy", "Sunny", "Clear", "Partly cloudy", "Clear", "Partly cloudy", "Clear", "Moderate rain at times", "Heavy rain at times", "Overcast", "Cloudy", "Clear", "Clear", "Partly cloudy", "Sunny", "Cloudy", "Partly cloudy", "Sunny", "Partly cloudy", "Cloudy", "Cloudy", "Clear", "Partly cloudy", "Clear", "Sunny", "Clear", "Clear", "Clear", "Partly cloudy", "Clear", "Clear", "Sunny", "Partly cloudy", "Sunny", "Sunny", "Clear", "Clear", "Sunny", "Sunny", "Sunny", "Partly cloudy", "Sunny", "Cloudy", "Partly cloudy", "Clear", "Clear", "Cloudy", "Clear", "Sunny", "Clear", "Moderate snow", "Clear", "Clear", "Sunny", "Clear", "Sunny", "Clear", "Partly cloudy", "Clear", "Patchy rain possible", "Cloudy", "Overcast", "Clear", "Clear", "Clear", "Heavy rain at times", "Sunny", "Sunny", "Cloudy", "Patchy rain possible", "Cloudy", "Sunny", "Sunny", "Clear", "Clear", "Sunny", "Partly cloudy", "Clear", "Partly cloudy", "Cloudy", "Clear", "Clear", "Clear", "Overcast", "Sunny", "Sunny", "Cloudy", "Partly cloudy", "Overcast", "Sunny", "Clear", "Sunny", "Overcast", "Sunny", "Cloudy", "Moderate or heavy rain shower", "Clear", "Clear", "Clear", "Moderate or heavy rain shower", "Partly cloudy", "Clear", "Clear", "Sunny", "Moderate or heavy rain shower", "Cloudy", "Patchy rain possible", "Moderate rain at times", "Overcast", "Overcast", "Partly cloudy", "Sunny", "Sunny", "Sunny", "Overcast", "Partly cloudy", "Clear", "Patchy rain possible", "Patchy rain possible", "Overcast", "Sunny", "Clear", "Sunny", "Clear", "Cloudy", "Partly cloudy", "Sunny", "Patchy rain possible", "Partly cloudy", "Partly cloudy", "Clear", "Clear", "Cloudy", "Moderate or heavy rain shower", "Clear", "Sunny", "Patchy rain possible", "Clear", "Clear", "Partly cloudy", "Partly cloudy", "Overcast", "Sunny", "Sunny", "Clear", "Cloudy", "Heavy rain at times", "Sunny", "Patchy rain possible", "Clear", "Partly cloudy", "Cloudy", "Moderate or heavy rain shower", "Partly cloudy", "Overcast", "Sunny", "Clear", "Sunny", "Overcast", "Heavy rain at times", "Partly cloudy", "Moderate or heavy rain shower", "Patchy rain possible", "Clear", "Cloudy", "Moderate or heavy rain shower", "Partly cloudy", "Sunny", "Partly cloudy", "Partly cloudy", "Sunny", "Sunny", "Sunny", "Sunny", "Sunny", "Clear", "Patchy rain possible", "Partly cloudy", "Clear", "Clear", "Partly cloudy", "Partly cloudy", "Clear", "Partly cloudy", "Moderate or heavy rain shower", "Clear", "Clear", "Overcast", "Sunny", "Partly cloudy", "Clear", "Sunny", "Partly cloudy", "Partly cloudy", "Cloudy", "Sunny", "Sunny", "Partly cloudy", "Clear", "Partly cloudy", "Clear", "Cloudy", "Clear", "Sunny", "Sunny", "Sunny", "Sunny", "Partly cloudy", "Overcast", "Sunny", "Clear", "Partly cloudy", "Clear", "Clear", "Sunny", "Sunny", "Clear", "Cloudy", "Patchy rain possible", "Clear", "Overcast", "Sunny", "Clear", "Partly cloudy", "Sunny", "Clear", "Clear", "Partly cloudy", "Partly cloudy", "Clear", "Moderate or heavy rain shower", "Clear", "Sunny", "Cloudy", "Sunny", "Clear", "Sunny", "Partly cloudy", "Clear", "Heavy rain at times", "Sunny", "Overcast", "Moderate rain at times", "Sunny", "Heavy rain at times", "Clear", "Cloudy", "Cloudy", "Clear", "Partly cloudy", "Overcast", "Sunny", "Sunny", "Clear", "Cloudy", "Sunny", "Clear", "Partly cloudy", "Sunny", "Sunny", "Sunny", "Sunny", "Partly cloudy", "Partly cloudy", "Clear", "Overcast", "Sunny", "Clear", "Sunny", "Partly cloudy", "Overcast", "Cloudy", "Partly cloudy", "Clear", "Clear", "Overcast", "Patchy rain possible", "Clear", "Sunny", "Clear", "Partly cloudy", "Partly cloudy", "Sunny", "Cloudy", "Moderate or heavy rain shower", "Cloudy", "Sunny", "Sunny", "Partly cloudy", "Overcast", "Clear", "Partly cloudy", "Partly cloudy", "Clear", "Partly cloudy", "Overcast", "Clear", "Cloudy", "Overcast", "Clear", "Overcast", "Partly cloudy", "Cloudy", "Clear", "Overcast", "Sunny", "Sunny", "Clear", "Cloudy", "Cloudy", "Partly cloudy", "Overcast", "Sunny", "Sunny", "Moderate or heavy rain shower", "Sunny", "Sunny", "Clear", "Clear", "Sunny", "Sunny", "Partly cloudy", "Clear", "Cloudy", "Sunny", "Sunny", "Sunny", "Clear", "Sunny", "Sunny", "Partly cloudy", "Overcast", "Moderate rain at times", "Partly cloudy", "Clear", "Cloudy", "Partly cloudy", "Moderate or heavy rain shower", "Patchy rain possible", "Cloudy", "Sunny", "Clear", "Clear", "Sunny", "Partly cloudy", "Partly cloudy", "Clear", "Overcast", "Sunny", "Partly cloudy", "Clear", "Partly cloudy", "Clear", "Patchy rain possible", "Sunny", "Partly cloudy", "Partly cloudy", "Cloudy", "Moderate or heavy rain shower", "Partly cloudy", "Sunny", "Heavy rain at times", "Moderate rain at times", "Overcast", "Partly cloudy", "Cloudy", "Overcast", "Moderate or heavy rain shower", "Sunny", "Clear", "Overcast", "Clear", "Clear", "Partly cloudy", "Partly cloudy", "Clear", "Overcast", "Partly cloudy", "Sunny", "Cloudy", "Clear", "Cloudy", "Sunny", "Partly cloudy", "Moderate rain at times", "Sunny", "Clear", "Moderate rain at times", "Moderate or heavy rain shower", "Sunny", "Overcast", "Cloudy", "Clear", "Clear", "Sunny", "Moderate rain at times", "Clear", "Partly cloudy", "Overcast", "Clear", "Partly cloudy", "Clear", "Patchy rain possible", "Partly cloudy", "Cloudy", "Sunny", "Sunny", "Clear", "Clear", "Clear", "Sunny", "Partly cloudy", "Clear", "Partly cloudy", "Cloudy", "Sunny", "Clear", "Clear", "Partly cloudy", "Partly cloudy", "Overcast", "Cloudy", "Clear", "Overcast", "Sunny", "Sunny", "Clear", "Cloudy", "Overcast", "Sunny", "Partly cloudy", "Light freezing rain", "Clear", "Patchy rain possible", "Sunny", "Partly cloudy", "Sunny", "Partly cloudy", "Clear", "Clear", "Partly cloudy", "Moderate rain at times", "Partly cloudy", "Clear", "Clear", "Partly cloudy", "Heavy rain at times", "Overcast", "Sunny", "Sunny", "Moderate rain at times", "Partly cloudy", "Partly cloudy", "Cloudy", "Sunny", "Cloudy", "Cloudy", "Cloudy", "Overcast", "Patchy rain possible", "Patchy rain possible", "Clear", "Sunny", "Sunny", "Overcast", "Partly cloudy", "Clear", "Clear", "Clear", "Patchy rain possible", "Partly cloudy", "Partly cloudy", "Partly cloudy", "Sunny", "Partly cloudy", "Partly cloudy", "Sunny", "Cloudy", "Sunny", "Overcast", "Patchy rain possible", "Sunny", "Patchy rain possible", "Clear", "Partly cloudy", "Sunny", "Moderate or heavy rain shower", "Sunny", "Overcast", "Sunny", "Moderate or heavy rain shower", "Heavy rain at times", "Sunny", "Sunny", "Partly cloudy", "Clear", "Partly cloudy", "Partly cloudy"]

    score = 0
    for i in range(len(predictions)):
        if predictions[i] == correct_prediction[i]:
            score += 1
    print(score/len(predictions))
    return 1



if __name__ == "__main__":
    main()
