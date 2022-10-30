import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


data = loadarff('pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')

#For the following exercises, normalize the data using sklearn’s MinMaxScaler.
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

#Using sklearn, apply k-means clustering fully unsupervisedly (without targets) on the normalized data with 𝑘 = 3 and three different seeds (using random ε {0,1,2}). Assess the silhouette and purity of the produced solutions.

kmeans = KMeans(n_clusters=3, random_state=0).fit(df_scaled)
y_pred = kmeans.predict(df_scaled)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df_scaled, y_pred))
print("Purity: %0.3f" % metrics.adjusted_rand_score(df_scaled['class'], y_pred))

kmeans = KMeans(n_clusters=3, random_state=1).fit(df_scaled)
y_pred = kmeans.predict(df_scaled)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df_scaled, y_pred))
print("Purity: %0.3f" % metrics.adjusted_rand_score(df_scaled['class'], y_pred))

kmeans = KMeans(n_clusters=3, random_state=2).fit(df_scaled)
y_pred = kmeans.predict(df_scaled)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df_scaled, y_pred))
print("Purity: %0.3f" % metrics.adjusted_rand_score(df_scaled['class'], y_pred))

# get the two features with highest variance from the df_scaled dataframe
df_scaled['var'] = df_scaled.var(axis=1)
df_scaled = df_scaled.sort_values(by=['var'], ascending=False)
df_scaled = df_scaled.drop(['var'], axis=1)
df_scaled = df_scaled.iloc[:,0:2]
print(df_scaled)



#kmeans = []
#labels = []
#silhouettes = []
#purity = []

#for i in range(3):
  #  kmeans = KMeans(n_clusters = 3, random_state = i).fit(df_scaled)
    #print("kmeans", kmeans[i])
  #  labels.append(kmeans.labels_)
    #print("labels", labels[i])
   # silhouettes.append(metrics.silhouette_score(df_scaled, labels[i]))
   # purity.append(metrics.adjusted_rand_score(df["class"], labels[i]))


#print("breakdance?",list(set(labels[0]) - set(labels[1])))
#print("Silhouette scores: ", silhouettes)
#print("Purity scores: ", purity)


#What is causing the non-determinism?

#The non-determinism is caused by the random initialization of the centroids. The algorithm will converge to a local minimum, which is why the results are different for different seeds.

#What is the best solution? Why?

#The best solution is the one with the highest silhouette score and purity score. The silhouette score is the average silhouette coefficient for all samples. The silhouette coefficient is a measure of how similar an object is to its own cluster compared to other clusters. The purity score is a measure of how similar the clusters are to the classes. The best solution is the one with the highest silhouette score and purity score, which is the solution with seed 2.




