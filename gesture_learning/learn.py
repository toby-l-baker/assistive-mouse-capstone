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
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

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
    t-SNE is a tool to visualize high-dimensional data. 
    It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence 
    between the joint probabilities of the low-dimensional embedding and the high-dimensional data
    '''
    embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500).fit_transform(train)
    # embedded = TSNE(n_components=2).fit_transform(train)
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
This function will read in raw data, return a train matrix after data modification.

Raw data:
Each line is corresponding to one image.
Each line has 21x2 numbers, which indicates (x, y) of 21 joint locations. Note that these are joint CENTRE locations.
Note that (x, y) are already normalized by the palm key point.
The order of 21 joints is Palm, Thumb root, Thumb mid1, Thumb mid2, Thumb tip, Index root, Thumb mid1, Thumb mid2, Index tip, 
Middle root, Middle mid1, Middle mid2, Middle tip, Ring root, Ring mid1, Ring mid2, Ring tip, Pinky root, Pinky mid1, Pinky mid2, Pinky tip.

Revised data:
Each line is corresponding to one image.
Each line has 20 numbers.
The order of 20 numbers is ratio of distance of (root, mid) / distance of tip of (Thumb, Index, Middle, Ring, Pinky); finger count; angle between fingers.

param file: raw data file path
param n: the number of samples we using; default None, use all samples
return train: revised trained data
'''
def generateTrainAndTest(file, n=None):
    # read in dataset and slice
    rawData = np.array(pa.read_csv(file, sep=",", header=None).values[1:])
    label = rawData[:, -1]
    data = rawData[:, :-1]
    if n != None:
        data = data[:n, :]  # Note: only use n lines
    # construct feature matirx
    feature = np.zeros((data.shape[0], 20))
    # distance ratio features
    for i in range(5):
        denominator = (data[:, 8*(i+1) - 1] - data[:, 0])**2 + (data[:, 8*(i+1)] - data[:, 1])**2  # distance from tip to palm
        for j in range(3):
            numerator = (data[:, 8*i + 2*j + 1] - data[:, 0])**2 + (data[:, 8*i + 2*j + 2] - data[:, 1])**2  # distance from root/mid1/mid2 to palm
            ratio = np.sqrt(numerator) / np.sqrt(denominator)
            feature[:, i*3 + j] = ratio
            # feature[:, i*3 + j] = ratio * 10  # 10 times to make more spearable?
    # finger number feature
    for i in range(len(feature)):
        temp = feature[i, :]
        feature[i, 15] = sum(temp[2 : 15 : 3] <= 1) * 10  # stretch finger number, weighted by 10
    # angle features
    for i in range(4):  # four pairs
        x1 = np.array([data[:, 8*(i+1) - 1] - data[:, 0], data[:, 8*(i+1)] - data[:, 1]], dtype=np.float32).T
        x2 = np.array([data[:, 8*(i+2) - 1] - data[:, 0], data[:, 8*(i+2)] - data[:, 1]], dtype=np.float32).T
        cos = np.sum(x1*x2, axis=1) / (np.sqrt(np.sum(x1**2, axis=1)) * np.sqrt(np.sum(x2**2, axis=1)))  # caculate cos(theta)
        # when zero angle, it is possible
        cos[cos > 1] = 1
        cos[cos < -1] = -1
        feature[:, 16 + i] = (np.arccos(cos) / np.pi * 180)
        # feature[:, 16 + i] = (np.arccos(cos) / np.pi * 180)**2  # Note: use quadratic here
    # shuffle the dataset and divide into 80/20
    indices = np.arange(feature.shape[0])
    np.random.shuffle(indices)
    feature = feature[indices]
    label = label[indices]
    n_train = int(feature.shape[0] * 0.8)
    feature_train, feature_test = feature[:n_train], feature[n_train:]
    label_train, label_test = label[:n_train], label[n_train:]
    # return revised matrix
    return feature_train, label_train, feature_test, label_test

def getAccuracy(prediction, truth, funcName):
    n = len(prediction)
    acc = sum(prediction == truth) / n * 100
    # print("The accuracy of " + funcName + " is " + str(acc) + "%")
    return acc

def unsupervisedMetrics(feature, prediction):
    chIndex = metrics.calinski_harabasz_score(feature, prediction)  # Calinski-Harabasz Index; a higher Calinski-Harabasz score relates to a model with better defined clusters
    dbIndex = metrics.davies_bouldin_score(feature, prediction)  # Davies-Bouldin index; a lower Davies-Bouldin index relates to a model with better separation between the clusters
    siScode = metrics.silhouette_score(feature, prediction, metric='euclidean')  # Silhouette Coefficient score; a higher Silhouette Coefficient score relates to a model with better defined clusters
    # print("ch: " + str(chIndex) + ", db: " + str(dbIndex) + ", si: " + str(siScode))
    return chIndex, dbIndex, siScode

def unsupervisedReorder(feature, label, prediction, k):
    # sort label and prediction
    indices = np.argsort(label)
    label = label[indices]  # 0~4
    prediction = prediction[indices]
    feature = feature[indices]
    # reorder prediction
    convert = {}  # conversion map; convert key to value
    i = 0
    for idx in range(len(prediction)):
        if prediction[idx] not in convert:
            convert[prediction[idx]] = i
            i += 1
            if i == k:
                break
    for idx in range(len(prediction)):
        prediction[idx] = convert[prediction[idx]]
    return feature, label, prediction

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
def K_means_train(feature_train, label_train, feature_test, label_test, k, filename=None):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(feature_train)
    # Train accuracy
    prediction_train = km.predict(feature_train)
    feature_train, label_train, prediction_train = unsupervisedReorder(feature_train, label_train, prediction_train, k)
    train_acc = getAccuracy(prediction_train, label_train, "Kmeans Train")
    # Test accuracy
    prediction_test = km.predict(feature_test)
    feature_test, label_test, prediction_test = unsupervisedReorder(feature_test, label_test, prediction_test, k)
    test_acc = getAccuracy(prediction_test, label_test, "Kmeans Test")
    # check index
    ch, db, si = unsupervisedMetrics(feature_test, prediction_test)
    # save model
    joblib.dump(km, './models/KMEANS.sav')  # dump the Kmean model
    # write to csv
    if filename:
        df = pa.DataFrame(prediction_test)
        df.to_csv(filename, index=False)
    return test_acc, ch, db, si

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
def EM_train(feature_train, label_train, feature_test, label_test, k, filename=None):
    gmm = GaussianMixture(n_components=k, covariance_type='spherical', init_params='kmeans', max_iter=250)  # 'spherical', 'diag', 'tied', 'full'
    gmm.fit(feature_train)
    # Train accuracy
    prediction_train = gmm.predict(feature_train)
    feature_train, label_train, prediction_train = unsupervisedReorder(feature_train, label_train, prediction_train, k)
    train_acc = getAccuracy(prediction_train, label_train, "GMM Train")
    # Test accuracy
    prediction_test = gmm.predict(feature_test)
    feature_test, label_test, prediction_test = unsupervisedReorder(feature_test, label_test, prediction_test, k)
    test_acc = getAccuracy(prediction_test, label_test, "GMM Test")
    # check index
    ch, db, si = unsupervisedMetrics(feature_test, prediction_test)
    # save model
    joblib.dump(gmm, './models/GMM.sav')  # dump the Kmean model
    # write to csv
    if filename:
        df = pa.DataFrame(prediction_test)
        df.to_csv(filename, index=False)
    return test_acc, ch, db, si

'''
Random Forest with aggregated feature
'''
def RF_train(feature_train, label_train, feature_test, label_test, filename=None):
    rf = RandomForestClassifier(max_depth=10, random_state=0)
    rf.fit(feature_train, label_train)
    # Train accuracy
    prediction_train = rf.predict(feature_train)
    trian_acc = getAccuracy(prediction_train, label_train, "RF Train")
    # Test accuracy
    prediction_test = rf.predict(feature_test)
    test_acc = getAccuracy(prediction_test, label_test, "RF Test")
    # save model
    joblib.dump(rf, './models/RF.sav')  # dump the Kmean model
    # write to csv
    if filename:
        df = pa.DataFrame(prediction_test)
        df.to_csv(filename, index=False)
    return test_acc


feature_train, label_train, feature_test, label_test = generateTrainAndTest('./dataset/fiveClass')
epoches = 1
# KMean algorithms
# acc, ch, db, si = 0, 0, 0, 0
# for i in range(epoches):
#     a, c, d, s = K_means_train(feature_train, label_train, feature_test, label_test, 5)
#     acc += a
#     ch += c
#     db += d
#     si += s
# print("The accuracy of KMean is " + str(acc/epoches) + "%")
# print("ch: " + str(ch/epoches) + ", db: " + str(db/epoches) + ", si: " + str(si/epoches))

# EM algorithms
# acc, ch, db, si = 0, 0, 0, 0
# for i in range(epoches):
#     a, c, d, s = EM_train(feature_train, label_train, feature_test, label_test, 5)
#     acc += a
#     ch += c
#     db += d
#     si += s
# print("The accuracy of GMM is " + str(acc/epoches) + "%")
# print("ch: " + str(ch/epoches) + ", db: " + str(db/epoches) + ", si: " + str(si/epoches))

# Random Forest algorithms
acc = 0
for i in range(epoches):
    acc += RF_train(feature_train, label_train, feature_test, label_test)
print("The accuracy of RF is " + str(acc/epoches) + "%")

# plot tSNE
# tSNE(feature_train, label_train, './tSNE_5.jpg', 5)
