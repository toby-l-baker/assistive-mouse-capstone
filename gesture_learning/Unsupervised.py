import numpy as np
import numpy.linalg as la
import pandas as pa
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def plotValidation(X, y, title, xlabel, ylabel, filename):
    fig = plt.figure()
    plt.plot(X, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(X)
    fig.savefig(filename)

def tSNE(train, label, filename, k=None):
    '''
    What is t-SNE? And how it work?
    t-SNE [1] is a tool to visualize high-dimensional data. 
    It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence 
    between the joint probabilities of the low-dimensional embedding and the high-dimensional data
    '''
    # embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250).fit_transform(train)
    embedded = TSNE(n_components=2).fit_transform(train)
    df = pa.DataFrame(train)
    df['label'] = label
    df_plot = df.copy()
    df_plot['tsne-2d-one'] = embedded[:, 0]
    df_plot['tsne-2d-two'] = embedded[:, 1]
    plt.figure(figsize=(8,5))
    snsplt = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", k),
        data=df_plot,
        legend="full",
        alpha=1
    )
    snsplt.get_figure().savefig(filename)

'''
This function will convert a set of keypoints to a set of features.

param keypoints: numpy array of 21 (x,y) keypoints
return features: numpy array of 20 features
'''
def keypointsToFeatures(keypoints):
    # construct feature matirx
    features = np.zeros(20)

    # distance ratio features
    for i in range(5):
        denominator = (keypoints[8*(i+1) - 1] - keypoints[0])**2 + (keypoints[8*(i+1)] - keypoints[1])**2  # distance from tip to palm
        for j in range(3):
            numerator = (keypoints[8*i + 2*j + 1] - keypoints[0])**2 + (keypoints[8*i + 2*j + 2] - keypoints[1])**2  # distance from root/mid1/mid2 to palm
            ratio = np.sqrt(numerator) / np.sqrt(denominator)
            features[i*3 + j] = ratio
            # features[i*3 + j] = ratio * 10  # 10 times to make more spearable?
    
    # finger number feature
    for i in range(len(features)):
        features[15] = sum(features[3 : 15 : 4] <= 1) * 10  # stretch finger number, weighted by 10
    
    # angle features
    for i in range(4):  # four pairs
        x1 = np.array([keypoints[8*(i+1) - 1] - keypoints[0], keypoints[8*(i+1)] - keypoints[1]], dtype=np.float32).T
        x2 = np.array([keypoints[8*(i+2) - 1] - keypoints[0], keypoints[8*(i+2)] - keypoints[1]], dtype=np.float32).T
        cos = np.sum(x1*x2) / (np.sqrt(np.sum(x1**2)) * np.sqrt(np.sum(x2**2)))  # caculate cos(theta)
        
        # when zero angle, it is possible
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        features[16 + i] = (np.arccos(cos) / np.pi * 180)
        # features[16 + i] = (np.arccos(cos) / np.pi * 180)**2  # Note: use quadratic here

    # return feature matrix
    return features

'''
This function will read in raw data, return a train matrix after data modification.

Raw data:
Each line is corresponding to one image.
Each line has 21x2 numbers, which indicates (x, y) of 21 joint locations. Note that these are joint CENTRE locations.
Note that (x, y) are already normalized by the palm key point.
The order of 16 joints is Palm, Thumb root, Thumb mid1, Thumb mid2, Thumb tip, Index root, Thumb mid1, Thumb mid2, Index tip, 
Middle root, Middle mid1, Middle mid2, Middle tip, Ring root, Ring mid1, Ring mid2, Ring tip, Pinky root, Pinky mid1, Pinky mid2, Pinky tip.

Revised data:
Each line is corresponding to one image.
Each line has 14 numbers.
The order of 16 joints is ratio of distance of (root, mid) / distance of tip of (Thumb, Index, Middle, Ring, Pinky); angle between fingers.

param file: raw data file path
param n: the number of samples we using; default None, use all samples
return train: revised trained data
'''
def generateTrain(file, n=None):
    # read in dataset and slice
    rawData = np.array(pa.read_csv(file, sep=",", header=None).values[1:])
    label = rawData[:, -1]
    data = rawData[:, :-1]
    if n != None:
        data = data[:n, :]  # Note: only use n lines
    # construct train feature matirx
    train = np.zeros((data.shape[0], 20))
    # distance ratio features
    for i in range(5):
        denominator = (data[:, 8*(i+1) - 1] - data[:, 0])**2 + (data[:, 8*(i+1)] - data[:, 1])**2  # distance from tip to palm
        for j in range(3):
            numerator = (data[:, 8*i + 2*j + 1] - data[:, 0])**2 + (data[:, 8*i + 2*j + 2] - data[:, 1])**2  # distance from root/mid1/mid2 to palm
            ratio = np.sqrt(numerator) / np.sqrt(denominator)
            train[:, i*3 + j] = ratio
            # train[:, i*3 + j] = ratio * 10  # 10 times to make more spearable?
    # finger number feature
    for i in range(len(train)):
        temp = train[i, :]
        train[i, 15] = sum(temp[3 : 15 : 4] <= 1) * 10  # stretch finger number, weighted by 10
    # angle features
    for i in range(4):  # four pairs
        x1 = np.array([data[:, 8*(i+1) - 1] - data[:, 0], data[:, 8*(i+1)] - data[:, 1]], dtype=np.float32).T
        x2 = np.array([data[:, 8*(i+2) - 1] - data[:, 0], data[:, 8*(i+2)] - data[:, 1]], dtype=np.float32).T
        cos = np.sum(x1*x2, axis=1) / (np.sqrt(np.sum(x1**2, axis=1)) * np.sqrt(np.sum(x2**2, axis=1)))  # caculate cos(theta)
        # when zero angle, it is possible
        cos[cos > 1] = 1
        cos[cos < -1] = -1
        train[:, 16 + i] = (np.arccos(cos) / np.pi * 180)
        # train[:, 16 + i] = (np.arccos(cos) / np.pi * 180)**2  # Note: use quadratic here

    # return revised matrix
    return train, label

def getAccuracy(prediction, truth, funcName):
    n = len(prediction)
    acc = sum(prediction == truth) / n * 100
    print("The accuracy of " + funcName + " is " + str(acc) + "%")
    return

'''
K_means algorithms; use validation to find the best K
param train: tarin matrix
return chIndex Calinski-Harabasz Index array
'''
def K_means_validation(train, min, max):
    chIndex = []  # Calinski-Harabasz Index; a higher Calinski-Harabasz score relates to a model with better defined clusters
    dbIndex = []  # Davies-Bouldin index; a lower Davies-Bouldin index relates to a model with better separation between the clusters
    siScode = []  # Silhouette Coefficient score; a higher Silhouette Coefficient score relates to a model with better defined clusters
    for k in range(min, max+1):
        kmeans_model = KMeans(n_clusters=k, random_state=0).fit(train)
        # print(kmeans_model.labels_)
        # print(kmeans_model.cluster_centers_)
        labels = kmeans_model.labels_
        chIndex.append(metrics.calinski_harabasz_score(train, labels))
        dbIndex.append(metrics.davies_bouldin_score(train, labels))
        siScode.append(metrics.silhouette_score(train, labels, metric='euclidean'))

    # plot image here
    plotValidation(range(min, max+1), chIndex, "K_means validation", "K", "Calinski-Harabasz Index", "/Users/sean/GestureLearning/K_means_ch_validation")
    plotValidation(range(min, max+1), dbIndex, "K_means validation", "K", "Davies-Bouldin Index", "/Users/sean/GestureLearning/K_means_db_validation")
    plotValidation(range(min, max+1), siScode, "K_means validation", "K", "Silhouette Coefficient score", "/Users/sean/GestureLearning/K_means_si_validation")
    
    # return Calinski-Harabasz Index array
    return chIndex

'''
Use best K to train.
'''
def K_means_train(train, label, filename, k=None):
    # chIndex = K_means_validation(train, 3, 20)
    # print(chIndex)
    # use the best K to build the model
    # bestK = chIndex.index(max(chIndex))+3
    # print(bestK)
    kmeans_best_model = KMeans(n_clusters=k, random_state=0).fit(train)
    prediction = kmeans_best_model.labels_
    print(prediction)
    # Train accuracy
    getAccuracy(prediction, label, "Kmeans")
    # write to csv
    df = pa.DataFrame(prediction)
    df.to_csv(filename, index=False)

'''
K_means algorithms; use validation to find the best K
param train: tarin matrix
return chIndex Calinski-Harabasz Index array
'''
def EM_validation(train, min, max):
    chIndex = []  # Calinski-Harabasz Index; a higher Calinski-Harabasz score relates to a model with better defined clusters
    dbIndex = []  # Davies-Bouldin index; a lower Davies-Bouldin index relates to a model with better separation between the clusters
    siScode = []  # Silhouette Coefficient score; a higher Silhouette Coefficient score relates to a model with better defined clusters
    for k in range(min, max+1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', init_params='kmeans', max_iter=5000)  # 'spherical', 'diag', 'tied', 'full'
        gmm.fit(train)
        labels = gmm.predict(train)
        chIndex.append(metrics.calinski_harabasz_score(train, labels))
        dbIndex.append(metrics.davies_bouldin_score(train, labels))
        siScode.append(metrics.silhouette_score(train, labels, metric='euclidean'))

    # plot image here
    plotValidation(range(min, max+1), chIndex, "EM validation", "K", "Calinski-Harabasz Index", "/Users/sean/GestureLearning/EM_ch_validation")
    plotValidation(range(min, max+1), dbIndex, "EM validation", "K", "Davies-Bouldin Index", "/Users/sean/GestureLearning/EM_db_validation")
    plotValidation(range(min, max+1), siScode, "EM validation", "K", "Silhouette Coefficient score", "/Users/sean/GestureLearning/EM_si_validation")
    
    # return Calinski-Harabasz Index array
    return chIndex

'''
Use best K to train.
'''
def EM_train(train, label, filename, k=None):
    # chIndex = EM_validation(train, 3, 20)
    # print(chIndex)
    # use the best K to build the model
    # bestK = chIndex.index(max(chIndex))+3
    # print(bestK); 
    gmm = GaussianMixture(n_components=k, covariance_type='spherical', init_params='kmeans', max_iter=250)  # 'spherical', 'diag', 'tied', 'full'
    gmm.fit(train)
    prediction = gmm.predict(train)
    print(prediction)
    # Train accuracy
    getAccuracy(prediction, label, "EM")
    # write to csv
    df = pa.DataFrame(prediction)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    train, label = generateTrain('./twoClass')
    # KMean algorithms
    K_means_train(train, label, './KMeans_2.txt', 2)
    # EM algorithms
    EM_train(train, label, './EM_2.txt', 2)
    # plot tSNE
    tSNE(train, label, './tSNE_2.jpg', 2)
