import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

data = pd.read_csv("input/data.csv", header=0)
data.drop("id", axis=1, inplace=True)
features_mean = list(data.columns[1:11])
features_se = list(data.columns[11:20])
features_worst = list(data.columns[21:31])

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
x = [0, 1] #cele 2 categorii
sns.countplot(x=data['diagnosis'], label="Count")
plt.show()
corr = data[features_mean].corr()

plt.figure(figsize=(14, 14))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 15},
            xticklabels=features_mean, yticklabels=features_mean,
            cmap='coolwarm')
plt.show()

prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
train, test = train_test_split(data, test_size=0.25)
train_X = train[prediction_var]
train_y = train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

max_features=[0.1,0.5,0.8]
max_samples=[0.25,0.5,0.85]
for i in max_features:
    for j in max_samples:
        model=BaggingClassifier(max_features=i,max_samples=int(j*train.shape[0]),bootstrap=True,)
        model.fit(train_X,train_y)
        prediction_y=model.predict(test_X)
        print("Accuracy for node-size=",i*100,"% ; in-bag=",j*100,"% ",accuracy_score(test_y,prediction_y))