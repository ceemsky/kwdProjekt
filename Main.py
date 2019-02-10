import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"
names = ["Recency (months)", "Frequency (times)", "Monetary (c.c. blood)", "Time (months)",
         "whether he/she donated blood in March 2007"]
blood_df = pd.read_csv(url, names=names)

print(blood_df.describe())

data = blood_df.loc[1:,
       ["Recency (months)", "Frequency (times)", "Monetary (c.c. blood)", "Time (months)"]].values.astype(np.int)
target = blood_df.loc[1:, ['whether he/she donated blood in March 2007']].values.astype(np.int)

data = StandardScaler().fit_transform(data)

data_train, data_test, target_train, target_test = \
    train_test_split(data, target, test_size=0.2, random_state=15)

print("Data train shape", data_train.shape)
print("Data test shape", data_test.shape)
print("Target train shape", target_train.shape)
print("Target test shape", target_test.shape)

neural_network = MLPClassifier(hidden_layer_sizes=(300, 150, 50), random_state=20)

neural_network.fit(data_train, target_train.ravel())
prediction = neural_network.predict(data_test)
confusion_matrix = confusion_matrix(target_test.ravel(), prediction)

print("Confusion_matrix:")
print(confusion_matrix)

heatmap = sns.heatmap(confusion_matrix)
plt.show()
acc = accuracy_score(target_test.ravel(), prediction)
print("Neural network model accuracy is {0:0.2f}".format(acc))