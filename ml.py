#This code trains machine learning model on data from 'view_data.py' and uses model to predict S&P change for the day from 'twitter.py' data. 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)

df = pd.read_csv ('mydata.csv')

for i in range(len(df)):
    df.loc[i,'Ratio'] = df.loc[i,'Positive']/df.loc[i,'Negative']

print(df[:10])

df['ChangeI'] = np.nan
for i in range(len(df)):
    if df.loc[i,'Change'] == 'large_decrease':
        df.loc[i,'ChangeI'] = -2
    elif df.loc[i,'Change'] == 'small_decrease':
        df.loc[i,'ChangeI'] = -1
    elif df.loc[i, 'Change'] == 'small_increase':
        df.loc[i, 'ChangeI'] = 1
    elif df.loc[i, 'Change'] == 'large_increase':
        df.loc[i, 'ChangeI'] = 2
    else:
        df.loc[i, 'ChangeI'] = 0

X, y = df.loc[:,['Positive','Negative','Ratio']], df['ChangeI'].tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype(np.float64))

svm_clf = SVC()
scores = cross_val_score(svm_clf, X_scaled, y, scoring="accuracy", cv=8)
#svm_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(scores)

y_pred = cross_val_predict(svm_clf, X_scaled, y, cv=8)
conf_mx = confusion_matrix(y, y_pred)
print(conf_mx)

plt.matshow(conf_mx, cmap=plt.cm.summer)
plt.show()


todays_stats = pd.read_csv ('todays_stat.csv')

myy = cross_val_predict(svm_clf, X_scaled, y, cv=8)

print(stats.mode(myy))
