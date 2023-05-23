import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, davies_bouldin_score

# Membaca dataset
dataset = pd.DataFrame({
    'att1': [0.051008, -0.748074, -0.772934, 0.218374, 0.372683],
    'att2': [0.160862, 0.089040, 0.263172, 0.127061, 0.496562],
    'label': [0, 0, 0, 0, 0]
})

# Memisahkan atribut dan atribut target
x = dataset[['att1', 'att2']]
y = dataset['label']

# Membagi data menjadi data latih dan data uji
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

# Membangun model Decision Tree
dt = DecisionTreeClassifier()
dt.fit(xTrain, yTrain)

# Mencetak akurasi model
print('Decision Tree Accuracy:', accuracy_score(yTest, dt.predict(xTest)))

# Visualisasi Decision Tree
plt.figure(figsize=(5, 5))
tree.plot_tree(dt, filled=True)

# Melakukan clustering dengan K-Means
kMeans = KMeans(n_clusters=2)
labels = kMeans.fit_predict(x)
db_score = davies_bouldin_score(x, labels)
print('Davies Bouldin Score:', db_score)

# Visualisasi hasil clustering
plt.scatter(x.loc[labels == 0, 'att1'],
            x.loc[labels == 0, 'att2'],
            s=100, c='purple',
            label='Cluster 1')
plt.scatter(x.loc[labels == 1, 'att1'],
            x.loc[labels == 1, 'att2'],
            s=100, c='orange',
            label='Cluster 2')
plt.scatter(kMeans.cluster_centers_[:, 0],
            kMeans.cluster_centers_[:, 1],
            s=100, c='red',
            label='Centroids')
plt.legend()
plt.show()
