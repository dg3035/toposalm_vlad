
import numpy as np
import itertools
from sklearn.cluster import KMeans
import pickle
import glob
import cv2
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.manifold import (Isomap, SpectralEmbedding, spectral_embedding, LocallyLinearEmbedding)

# def edge_list_creation(Images_no): 
#     E = []
#     edges = Images_no-1
#     for i in range(edges):
#         E.append([i,i+1])
#     return E    


# # print(edge_list_creation(10))

# def adjacency_matrix(Edge_list): 
#     n_vertices = len(Edge_list)
#     A = np.zeros((n_vertices+1, n_vertices+1))

#     for each_vertex in range(len(Edge_list)): 
#         x,y = Edge_list[each_vertex][0], Edge_list[each_vertex][1]

#         A[x][y] = 1
#         A[y][x] = 1

#     return A


if __name__=='__main__': 

    k_clusters = 64
    # Image_ID = 41
    Image_ID = 601
    # n_neighbors = 10
    # n_components = 2
    # n_points = 1000

    VLAD_Descriptors_from_file = np.loadtxt("vlad_601_images")
    print("VLAD Descriptors from file shape: ", VLAD_Descriptors_from_file.shape)

    vlad = VLAD_Descriptors_from_file.reshape(Image_ID, k_clusters*128)
    print("VLAD_Descriptors_from_file_after_reshaping", vlad.shape)


    # fig = plt.figure(figsize=(15,8))
    # fig= plt.figure()
    # plt.subplot("Manifold Learning with %i points, %i neighbors, and %i components" % (n_points, n_neighbors, n_components), fontsize = 14)

    # methods = ['Standard', 'Ltsa', 'Hessain', 'Modified']
    # labels = ['LLE', 'LTSA', 'Hessain LLE', 'Modified LLE']

    # for i, method in enumerate(methods): 
    #     unfolded_data4=LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',method= method).fit_transform(vlad)
    #     ax=fig.add_plot(251 + i)
    #     plt.scatter(unfolded_data4[:,0],unfolded_data4[:,1],color = 'r',marker = '*', 
    #                 s = 300, label = 'image %i plot'.format(i))
    #     plt.plot(unfolded_data4[:,0],unfolded_data4[:,1],color = 'r', label= 'image %i plot' % (i))




    # isomap_embedding = Isomap(n_neighbors = 4, n_components=2)
    # # ax = fig.add_plot(254)
    # unfolded_data1 = isomap_embedding.fit_transform(vlad)
    # print('unfolded_data1 shape', unfolded_data1.shape)
    # plt.scatter(unfolded_data1[:,0],unfolded_data1[:,1],color = 'r', marker = '*', s = 300,label = 'image 1 plot')
    # plt.plot(unfolded_data1[:,0],unfolded_data1[:,1], color = 'r', label = 'image 1 plot')
    


    # print("image 1 matrix vlad", unfolded_data1)

    # A = adjacency_matrix(edge_list_creation(Image_ID))
    # A = np.asarray(A)

    # plt. figure() 

    # spectral_embeddings = spectral_embedding(A, n_components =2, drop_first=False)

    # # unfolded_data2 = spectral_embeddings.fit_transform(vlad)
    # unfolded_data2 = spectral_embeddings
    # print('unfolded_data2 shape', unfolded_data2.shape)
    # plt.scatter(unfolded_data2[:,0],unfolded_data2[:,1],color = 'b', marker = '+', s = 300,label = 'image 1 plot')
    # plt.plot(unfolded_data2[:,0],unfolded_data2[:,1], color = 'k', label = 'image 2 plot')
    # print("image 1 matrix vlad", unfolded_data2)

    plt. figure()

    spectral_embedding = SpectralEmbedding(n_components =2, n_neighbors = 6)

    unfolded_data3 = spectral_embedding.fit_transform(vlad)
    np.savetxt("vlad_601_reduced_points",unfolded_data3)
    print('unfolded_data3 shape', unfolded_data3.shape)
    plt.scatter(unfolded_data3[:,0],unfolded_data3[:,1],color = 'b', marker = '+', s = 300,label = 'image 1 plot')
    plt.plot(unfolded_data3[:,0],unfolded_data3[:,1], color = 'k', label = 'image 2 plot')
    # print("image 1 matrix vlad", unfolded_data3)

    plt.legend()    
    plt.show()
