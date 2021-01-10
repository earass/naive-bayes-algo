import pandas as pd
from sklearn.datasets import load_iris
from math import sqrt, exp, pi
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

iris_data = load_iris()


def get_probability(stdev, mean, x):
    """ calculates probability """
    return (1.0 / (stdev * sqrt(2 * pi))) * exp(-0.5 * ((x - mean)/stdev) ** 2)


def get_iris_data():
    # loading iris data from the sklearn datasets
    df_iris = pd.DataFrame(iris_data['data'])
    df_iris['target'] = iris_data['target']
    print(f"Size of data: {len(df_iris)}")
    return df_iris


def train_test_split(df_iris, test_size):
    # picking random records with test size as test data
    df_test = df_iris.sample(n=test_size)
    print(f"length of test: {len(df_test)}")

    # remaining as training data
    df_train = df_iris.loc[~df_iris.index.isin(df_test.index)]
    print(f"length of train: {len(df_train)}")
    return df_train, df_test


def get_stats_model(df):
    """ Creates distributions for each class and features to be used as the stats model """
    stats = df.groupby('target').agg(['mean', 'std'])
    print('stats model')
    print(stats)
    return stats


def predict(model, X):
    """ Predict the classes for the unseen data using the stats model"""
    predictions = []

    # iterate over the input records to get prediction for each record
    for row in X:
        class_probs = pd.DataFrame()

        # iterating over each feature to get probability for each feature
        for ind, val in enumerate(row):
            feat_mdl = model.copy()[ind]
            # calculating feature prob
            feat_mdl['feature_prob'] = feat_mdl.apply(lambda x: get_probability(stdev=x['std'], mean=x['mean'], x=val),
                                                   axis=1)
            # aggregating with the other features' probabilities
            if not class_probs.empty:
                class_probs = feat_mdl.join(class_probs)
                class_probs['class_prob'] = class_probs['feature_prob'] * class_probs['class_prob']
                class_probs = class_probs[['class_prob']]
            else:
                class_probs = feat_mdl[['feature_prob']].rename(columns={'feature_prob': 'class_prob'})

        # getting class for which the probability is higher for the record
        best_prob_class = class_probs['class_prob'].idxmax()
        predictions.append(best_prob_class)
    return predictions


def execute():

    # Loading data
    df_iris = get_iris_data()

    # separating data
    df_train, df_test = train_test_split(df_iris, test_size=10)
    X_test, y_test = df_test.drop('target', axis=1), df_test['target']

    # creating distributions
    stats = get_stats_model(df_train)

    # predicting on test data points
    y_pred = predict(stats, X_test.values)
    print("actual classes: ", y_test.tolist())
    print("predicted classes: ", y_pred)

    # calculating accuracy
    print("accuracy: ", 100*accuracy_score(y_true=y_test, y_pred=y_pred))


if __name__ == '__main__':
    execute()
