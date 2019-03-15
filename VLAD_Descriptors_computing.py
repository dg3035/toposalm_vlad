
import numpy as np
import itertools
from sklearn.cluster import KMeans
import pickle
import glob
import cv2
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap


def getDescriptors(path, functionHandleDescriptor):
    descriptors = list()

    for imagePath in glob.glob(path + "/*.JPG"):
        # print("entered the for vlad_loop_3_390_images")
        im = cv2.imread(imagePath)
        # print("read the image from the path")
        kp, des = functionHandleDescriptor(im)
        # print("found the descriptors")
        # print(des)
        # print(type(des))
        if des is not None:
            descriptors.append(des)

    # flatten list
    descriptors = list(itertools.chain.from_iterable(descriptors))

    # list to array
    descriptors = np.asarray(descriptors)

    return descriptors


def kMeansDictionary(training, k):
    est = KMeans(n_clusters=k, init='k-means++', tol=0.0001, verbose=1).fit(training)
    return est


def getVLADDescriptors(path, functionHandleDescriptor, visualDictionary):
    descriptors = list()
    idImage = list()
    for imagePath in glob.glob(path + "/*.jpg"):
        # print(imagePath)
        im = cv2.imread(imagePath)
        kp, des = functionHandleDescriptor(im)
        if des is not None:
            v = VLAD(des, visualDictionary)# calling the vlad function to create the vlad for each and every image  
            descriptors.append(v)
            idImage.append(imagePath)

    # list to array

    descriptors = np.asarray(descriptors)
    desc_le = int(len(descriptors[0]) / 128)
    descriptors = np.reshape(descriptors, (len(idImage), desc_le, 128))

    return descriptors, idImage



def VLAD(X, visualDictionary):
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_
    k = visualDictionary.n_clusters

    m, d = X.shape
    V = np.zeros([k, d])
    # computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels == i) > 0:
            # add the diferences
            V[i] = np.sum(X[predictedLabels == i, :] - centers[i], axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V) * np.sqrt(np.abs(V))

    V = V / np.sqrt(np.dot(V, V))
    return V



def describeSURF(image):
    surf = cv2.xfeatures2d.SURF_create(400, extended=True)
    # it is better to have this value between 300 and 500
    kp, des = surf.detectAndCompute(image, None)
    return kp, des




if __name__=='__main__': 
    
    k_clusters = 64 #try different number of clusters to get the most out of the high dimensional space
        
    path_to_image_folder = "C:/Users/gabad/Google Drive/TopoVSLAM/Dhruv_ TopoV-Mapping/Videos/markerspace (2-1-2019 6-59-51 PM)"
    # path_to_image_folder = "C:/Users/gabad/Documents/workspace/py_ws/TopoSLAM/vlad_test_py36/Scripts/VLAD Scripts/Rogers_Hall_Video (10-18-2018 2-04-11 PM)"
    

    descriptors = getDescriptors(path_to_image_folder, describeSURF)
    code_book = kMeansDictionary(descriptors, k_clusters)
    VLAD_Descriptors, Image_ID =getVLADDescriptors(path_to_image_folder ,describeSURF, code_book)

    print("VLAD matrix shape",VLAD_Descriptors.shape)
    print("Number of images", len(Image_ID))                      
    # print(type(VLAD_Descriptors))

    VLAD_Descriptors_reshaped = VLAD_Descriptors.reshape(len(Image_ID)*k_clusters,128)
    print("VLAD_Descriptors_reshaped shape: ", VLAD_Descriptors_reshaped.shape)

    # # c=[np.array(item).tolist() for item in c]

    np.savetxt("99_images_from_makerspace",VLAD_Descriptors_reshaped)
    # =======================================================================================#
    

    # VLAD_Descriptors_from_file = np.loadtxt("vlad_txt_file")
    # print("VLAD Descriptors from file shape: ", VLAD_Descriptors_from_file.shape)

    # VLAD_Descriptors_from_file_after_reshaping = VLAD_Descriptors_from_file.reshape(len(Image_ID), k_clusters, 128)
    # print("VLAD_Descriptors_from_file_after_reshaping", VLAD_Descriptors_from_file_after_reshaping.shape)


    # plt.figure()

    # embedding1 = Isomap(n_neighbors = 2, n_components=1)
    # image1 = embedding1.fit_transform(d[0][:1])
    # print('image1 shape', image1.shape)
    # plt.scatter(image1[:,0],image1[:,1],color = 'r', marker = '*', s = 100,label = 'image 1 plot')
    # # plt.plot(image1[:,0],image1[:,1], color = 'r', label = 'image 1 plot')
    # print("image 1 matrix vlad", image1)

  
    # embedding2 = Isomap(n_neighbors = 9, n_components=2)

    # image2 = embedding2.fit_transform(d[1][:10])

    # print('image1 shape', image2.shape)
    # plt.scatter(image2[:,0],image2[:,1],color = 'k', marker = 'o', s = 100, label = 'image 2 plot')
    # # plt.plot(image2[:,0],image2[:,1],color = 'k', label = 'image 2 plot')
    # print("image 2 matrix vlad", image2)

    # plt.legend()
    # plt.show()
