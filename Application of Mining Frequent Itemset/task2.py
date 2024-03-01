import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

creditcard_data=pd.read_csv('creditcard.csv', index_col=0)
print(creditcard_data.info())
print('\n')
pd.options.display.max_columns=len(creditcard_data)
print(creditcard_data.head(3))

round(creditcard_data['Class'].value_counts()*100/len(creditcard_data)).convert_dtypes()

def prep_data(df):
    X = df.iloc[:, 1:28]
    X = np.array(X).astype(float)
    y = df.iloc[:, 29]
    y = np.array(y).astype(float)
    return X, y
def plot_data(X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], label='Class #0', alpha=0.5, linewidth=0.15)
    plt.scatter(X[y==1, 0], X[y==1, 1], label='Class #1', alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()
X, y = prep_data(creditcard_data)
plot_data(X, y)

method=SMOTE()
X_resampled, y_resampled =method.fit_resample(X,y)
plot_data(X_resampled, y_resampled)

def compare_plot(X, y, X_resampled, y_resampled, method):
    f, (ax1, ax2) = plt.subplots(1, 2)
    c0 = ax1.scatter(X[y==0, 0], X[y==0, 1], label='Class #0',alpha=0.5)
    c1 = ax1.scatter(X[y==1, 0], X[y==1, 1], label='Class #1',alpha=0.5, c='r')
    ax1.set_title('Original set')
    ax2.scatter(X_resampled[y_resampled==0, 0], X_resampled[y_resampled==0, 1], label='Class #0', alpha=.5)
    ax2.scatter(X_resampled[y_resampled==1, 0], X_resampled[y_resampled==1, 1], label='Class #1', alpha=.5,c='r')
    ax2.set_title(method)
    plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center', ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    return plt.show()
print(f'Original set:\n'
      f'{pd.value_counts(pd.Series(y))}\n\n'
      f'SMOTE:\n'
      f'{pd.value_counts(pd.Series(y_resampled))}\n')
compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')

#logistic regression
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


