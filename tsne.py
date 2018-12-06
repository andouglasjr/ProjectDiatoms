# Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2011

print(__doc__)
from time import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from torchvision import models, transforms
from PIL import Image
from DiatomsDataset import DiatomsDataset
import torch

list_of_name_folders = ['train_diatoms_3_class_simulate_all','test_diatoms_3_class']
data_dir = '../data/Dataset_4'
#diatoms = datasets.load_diatoms(n_class=6)

data_transforms = {
     list_of_name_folders[0]: transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(30),
        transforms.CenterCrop(25),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        #transforms.Normalize([0.5017443],[0.09787486])
        #transforms.Normalize([0.5017639, 0.5017639, 0.5017639],[0.08735436, 0.08735436, 0.08735436])
        #transforms.Normalize([0.493], [0.085])
    ]),
     list_of_name_folders[1]: transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(30),
        transforms.CenterCrop(25),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        #transforms.Normalize([0.5017443],[0.09787486])
        #transforms.Normalize([0.5017639, 0.5017639, 0.5017639],[0.08735436, 0.08735436, 0.08735436])
        #transforms.Normalize([0.493], [0.085])
    ])
}

data_show = {
     'data_show': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(30),
        transforms.CenterCrop(25),
        transforms.Grayscale(1),
        transforms.ToTensor()
        #transforms.Normalize([0.5017443],[0.09787486])
        #transforms.Normalize([0.5017639, 0.5017639, 0.5017639],[0.08735436, 0.08735436, 0.08735436])
        #transforms.Normalize([0.493], [0.085])
    ])
}

diatoms = DiatomsDataset(list_of_name_folders[0], data_dir, data_transforms)
diatoms_simulate = DiatomsDataset(list_of_name_folders[1], data_dir, data_transforms)
#toTensor = transforms.ToTensor()
transformToPILImage = transforms.ToPILImage()



#X = [data_transforms['train_'](diatoms[x]['image']) for x in range(len(diatoms))]
#y = [diatoms[x]['diatoms'] for x in range(len(diatoms))]
qtd_of_imgs = len(diatoms) + len(diatoms_simulate)
X1=np.zeros((qtd_of_imgs,25*25))
y=np.zeros((qtd_of_imgs,))
cont = 0

#X1 = [sample['image'] for (i, sample) in enumerate(diatoms)]

for (i, sample) in enumerate(diatoms):    
    X1[i]=sample['image'].reshape(1,-1)
    y[i]=sample['diatoms']
    
for (i, sample) in enumerate(diatoms_simulate):
    X1[len(diatoms)+i]=sample['image'].reshape(1,-1)
    y[len(diatoms)+i]=sample['diatoms']

#for i in range(len(diatoms_simulate)):
#    #rand = random.randrange(0,len(diatoms))
#    x = data_transforms[list_of_name_folders[1]](diatoms_simulate[i]['image'])
#    a = np.zeros(len(x.getdata()))

#    for j in range(len(x.getdata())):
#        a[j] = x.getdata()[j][0]/255
    #x = list(x.getdata())
    #x = np.array(x)
    #X1[len(diatoms)+i]=x.numpy().reshape(1,-1)    
#    X1[len(diatoms)+i]=a.reshape(1,-1)
#    y[len(diatoms)+i]=diatoms_simulate[i]['diatoms']

#Xd = diatoms.data
#yd = diatoms.target
#n_samples, n_features = 90, 10 
n_neighbors = 8


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        if(i < len(diatoms)):
            plt.text(X[i, 0], X[i, 1], str(y[i]) + '-simulate',
                 color=plt.cm.Set1((y[i]+1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
        else:
            plt.text(X[i, 0], X[i, 1], str(y[i]) + '-real',
                 color=plt.cm.Set1((y[i]+1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            if(i < len(diatoms)):
            	x = data_show['data_show'](diatoms['image'])
            else:
            	x = data_show['data_show'](diatoms_simulate[i - len(diatoms)]['image'])
            
            img = x[0]
            #print(img.shape)
            #img = X1[i].reshape(25,25)
            #img = X1[i]
            #print(type(img))
            #img = torch.from_numpy(img)
            #img = np.uint8((img)*255)
            #print(img)
            #img = Image.fromarray(np.uint8((img)*255))
            
            #img = np.uint8(img*255)

            #img = transformToPILImage(img)

            
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(img, cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#----------------------------------------------------------------------
# Plot images of the diatoms
n_img_per_row = 13 
img = np.zeros((28 * n_img_per_row, 28 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 28 * i + 1
    for j in range(n_img_per_row):
        iy = 28 * j + 1
        img[ix:ix + 25, iy:iy + 25] = X1[i * n_img_per_row + j].reshape((25, 25))
        

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 50-dimensional diatoms dataset')


#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X1)
#plot_embedding(X_projected, "Random Projection of the diatoms")


#----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X1)
#plot_embedding(X_pca,
 #              "Principal Components projection of the diatoms (time %.2fs)" %
  #             (time() - t0))

#----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components

print("Computing Linear Discriminant Analysis projection")
X2 = X1.copy()
X2.flat[::X1.shape[1] + 1] += 0.01  # Make X invertible
t0 = time()
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
plot_embedding(X_lda,
               "Linear Discriminant projection of the diatoms (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# Isomap projection of the diatoms dataset
print("Computing Isomap embedding")
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X1)
print("Done.")
#plot_embedding(X_iso,
 #              "Isomap projection of the diatoms (time %.2fs)" %
  #             (time() - t0))


#----------------------------------------------------------------------
# Locally linear embedding of the diatoms dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
t0 = time()
X_lle = clf.fit_transform(X1)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
#plot_embedding(X_lle,
 #              "Locally Linear Embedding of the diatoms (time %.2fs)" %
  #             (time() - t0))


#----------------------------------------------------------------------
# Modified Locally linear embedding of the diatoms dataset
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='modified')
t0 = time()
X_mlle = clf.fit_transform(X1)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
#plot_embedding(X_mlle,
 #              "Modified Locally Linear Embedding of the diatoms (time %.2fs)" %
  #             (time() - t0))


#----------------------------------------------------------------------
# HLLE embedding of the diatoms dataset
print("Computing Hessian LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='hessian')
t0 = time()
X_hlle = clf.fit_transform(X1)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
#plot_embedding(X_hlle,
  #             "Hessian Locally Linear Embedding of the diatoms (time %.2fs)" %
 #              (time() - t0))


#----------------------------------------------------------------------
# LTSA embedding of the diatoms dataset
print("Computing LTSA embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='ltsa')
t0 = time()
X_ltsa = clf.fit_transform(X1)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
#plot_embedding(X_ltsa,
 #              "Local Tangent Space Alignment of the diatoms (time %.2fs)" %
  #             (time() - t0))

#----------------------------------------------------------------------
# MDS  embedding of the diatoms dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X1)
print("Done. Stress: %f" % clf.stress_)
#plot_embedding(X_mds,
 #              "MDS embedding of the diatoms (time %.2fs)" %
  #             (time() - t0))

#----------------------------------------------------------------------
# Random Trees embedding of the diatoms dataset
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
t0 = time()
X_transformed = hasher.fit_transform(X1)
pca = decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)

#plot_embedding(X_reduced,
 #              "Random forest embedding of the diatoms (time %.2fs)" %
  #             (time() - t0))

#----------------------------------------------------------------------
# Spectral embedding of the diatoms dataset
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
t0 = time()
X_se = embedder.fit_transform(X1)

#plot_embedding(X_se,
 #              "Spectral embedding of the diatoms (time %.2fs)" %
  #             (time() - t0))

#----------------------------------------------------------------------
# t-SNE embedding of the diatoms dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X1)

plot_embedding(X_tsne,
               "t-SNE embedding of the diatoms (time %.2fs)" %
               (time() - t0))

plt.show()