import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("mall_customers.csv")

# Renaming the name of columns
df = df.rename(columns={'Annual Income (k$)': 'Annual_income','Spending Score (1-100)': 'Spending_score'})

# Replacing objects for numerical values ( Mapping )
df['Gender'].replace(['Female', 'Male'], [0, 1], inplace=True)

# Count and plot gender
sns.countplot(y='Gender', data=df, palette="husl", hue="Gender")

# Pairplot with variables we want to study and visualize 
sns.pairplot(df, vars=["Age", "Annual_income", "Spending_score"],kind="reg", hue="Gender", palette="husl", markers=['o', 'D'])

sns.lmplot(x="Age", y="Annual_income", data=df, hue="Gender")

sns.lmplot(x="Annual_income", y="Spending_score", data=df, hue="Gender")

sns.lmplot(x="Age", y="Spending_score", data=df, hue="Gender")

# Creating values for the Elbow
x = df.loc[:, ["Age", "Annual_income", "Spending_score"]]
inertia = []
k = range(1, 20)
for i in k:
    means_k = KMeans(n_clusters=i, random_state=0)
    means_k.fit(x)
    inertia.append(means_k.inertia_)

# Plotting the Elbow for KMeans Clustering Algorithm
plt.plot(k, inertia, 'bo-')
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()

# Training kmeans with 5 clusters
means_k = KMeans(n_clusters=5, random_state=0)
means_k.fit(x)
labels = means_k.labels_
centroids = means_k.cluster_centers_

# Create a 3d plot to view the data sepparation made by Kmeans
trace = go.Scatter3d(
    x=x['Spending_score'],
    y=x['Annual_income'],
    z=x['Age'],
    mode='markers',
    marker=dict(
        color=labels,
        size=10,
        line=dict(
            color=labels,
        ),
        opacity=0.9
    )
)
layout = go.Layout(
    title='Clusters',
    scene=dict(
        xaxis=dict(title='Spending_score'),
        yaxis=dict(title='Annual_income'),
        zaxis=dict(title='Age')
    )
)
fig = go.Figure(data=trace, layout=layout)
py.offline.iplot(fig)

"""
Yellow Cluster - The yellow cluster groups young people with moderate to low annual income who actually spend a lot.
Purple Cluster - The purple cluster groups reasonably young people with pretty decent salaries who spend a lot.
Pink Cluster - The pink cluster basically groups people of all ages whose salary isn't pretty high and their spending score is moderate.
Orange Cluster - The orange cluster groups people who actually have pretty good salaries and barely spend money, their age usually lays between thirty and sixty years.
Blue Cluster - The blue cluster groups whose salary is pretty low and don't spend much money in stores, they are people of all ages.
"""
