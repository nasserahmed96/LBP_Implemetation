import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm


def read_data():
    # Check for dataset
    print(os.getcwd())
    print(os.listdir('../Dataset'))
    return pd.read_csv('../Dataset/Dataset_spine.csv')


def get_data_with_columns():
    back_data = read_data()
    del back_data['Unnamed: 13']
    back_data.columns = ['pelvic_incidence', 'pelvic tilt', 'lumbar_lordosis_angle', 'sacral_slope',
                         'pelvic_radius', 'degree_spondylolisthesis', 'pelvic_slope', 'Direct_tilt', 'thoracic_slope',
                         'cervical_tilt', 'sacrum_angle', 'scoliosis_slope', 'Status']
    back_data.info()
    return back_data


def draw_heatmap(data):
    corr_back = data.corr()
    mask = np.zeros_like(corr_back, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_back, mask=mask, center=0, square=True, linewidth=.5)
    plt.show()


def draw_correlations(data):
    print('Correlations')
    print(data.corr())
    print('Mean corresponding to status col')
    print(data.groupby('Status').mean())
    print('Median corresponding to status col')
    print(data.groupby('Status').median())


def draw_box_plot(data):
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))
    axes = axes.flatten()
    for i in range(0, len(data.columns)-1):
        sns.boxplot(x='Status', y=data.iloc[:,i], data=data, orient='v', ax=axes[i])
    plt.tight_layout()
    plt.show()


def data_preprocessing(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=0)
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
    scalar.fit(X_train)
    train_scaled = scalar.transform(X_train)
    test_scaled = scalar.transform(X_test)
    return train_scaled, test_scaled, y_train.astype(int), y_test.astype(int)


def logistic_regression(x, y):
    logreg = LogisticRegression().fit(x, y)
    return logreg


def training(X, y):
    X_train_scaled, X_test_scaled, y_train, y_test = data_preprocessing(X, y)
    logreg_result = logistic_regression(X_train_scaled, y_train)
    print("Training set score: {:0.3f}".format(logreg_result.score(X_train_scaled, y_train)))
    print("Test set score: {:0.3f}".format(logreg_result.score(X_test_scaled, y_test)))
    logit_model = sm.Logit(y_train, X_train_scaled)
    result = logit_model.fit()
    print(result.summary2())
    return logreg_result, X_test_scaled, y_test


def draw_confusion_matrix(y_test_string, y_pred_string):
    ax = plt.subplot()
    labels = ['Abnormal', 'Normal']
    cm = confusion_matrix(y_test_string, y_pred_string)
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Abnormal', 'Normal'])
    ax.yaxis.set_ticklabels(['Abnormal', 'Normal'])
    plt.show()


def predict(logreg_result, X_test_scaled):
    y_pred = logreg_result.predict(X_test_scaled)
    y_pred_string = y_pred.astype(str)
    y_pred_string[np.where(y_pred_string == '0')] = 'Normal'
    y_pred_string[np.where(y_pred_string == '1')] = 'Abnormal'
    return y_pred_string


def test(y_test):
    y_test_string = y_test.astype(str)
    y_test_string[np.where(y_test_string == '0')] = 'Normal'
    y_test_string[np.where(y_test_string == '1')] = 'Abnormal'
    return y_test_string


def run():
    lower_back_pain_data = get_data_with_columns()
    draw_correlations(lower_back_pain_data)
    draw_heatmap(lower_back_pain_data)
    draw_box_plot(lower_back_pain_data)
    lower_back_pain_data.loc[lower_back_pain_data.Status == 'Abnormal', 'Status'] = 1
    lower_back_pain_data.loc[lower_back_pain_data.Status == 'Normal', 'Status'] = 0
    X = lower_back_pain_data.loc[:, lower_back_pain_data.columns != 'Status']
    y = lower_back_pain_data.loc[:, lower_back_pain_data.columns == 'Status']

    # Removing the highly correlated variables which also had high standard error
    cols_to_include = [cols for cols in X.columns if cols not in ['pelvic_incidence', 'pelvic tilt', 'sacral_slope']]
    X = lower_back_pain_data[cols_to_include]
    logreg_result, X_test_scaled, y_test = training(X, y)

    X_trim_1 = X.loc[:, [
                            'lumbar_lordosis_angle',
                            'pelvic_radius',
                            'degree_spondylolisthesis'
                        ]
               ]

    logreg_result, X_test_scaled, y_test = training(X_trim_1, y)
    y_pred_string = predict(logreg_result, X_test_scaled)
    y_test_string = test(y_test)
    draw_confusion_matrix(y_test_string, y_pred_string)

if __name__ == '__main__':
    run()




