import numpy as np
from DataUtils import DataUtils
import argparse
import os
import torch
from torchvision import datasets, models, transforms

if __name__ == "__main__":

    
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Analysis Diatoms Research CNR-ISASI")
    parser.add_argument('--batch_size', default=256, type=int,
                       help="Size of the batch")
    parser.add_argument('--data_dir', default='../data/Dataset_4',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--save_dir', default='results',
                       help="Directory to save the results!")
    parser.add_argument('--dataset', default='results',
                       help="Directory where is the data!")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--images_per_class', default=10, help="how many images will be used per class")
    parser.add_argument('--classes_training', default=50, help="how many classes there are in the training")
    parser.add_argument('--perplexy', default=30, help="TSNE perplexy")
    parser.add_argument('--n_iter', default=300, help="TSNE iterations")
    
    
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    image_size = 224
    
    data_transforms = transforms.Compose([transforms.Resize(256), 
                                          transforms.CenterCrop(224), 
                                       transforms.Grayscale(1),
                                      transforms.ToTensor()])
                                      #transforms.Normalize(_mean, _std)])
    
    
    
    data = DataUtils(transformations=data_transforms, device = device, args = args)
    dataloaders = data.load_data()
    #print(data.train_size, data.valid_size)
    X = np.zeros((data.train_size, image_size*image_size))
    y = np.zeros((data.train_size))
        
    for i, sample in enumerate(dataloaders['train']):
        (inputs, labels),(_,_) = sample
        for j in range(len(inputs)):
            img = inputs[j]
            X[j,:] = img.view(-1, image_size*image_size)
            y[j] = labels[j]
            
    #X = X.numpy()
    #y = y.numpy()
    print(X.shape)
    
    import pandas as pd
    
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    
    df = pd.DataFrame(X,columns=feat_cols)
    df['label'] = y
    df['label'] = df['label'].apply(lambda i: str(i))

    X, y = None, None

    print('Size of the dataframe: {}'.format(df.shape))

    import matplotlib.pyplot as plt

    rndperm = np.random.permutation(df.shape[0])
    # Plot the graph
    #plt.gray()
    #fig = plt.figure( figsize=(16,7) )
    #for i in range(0,15):
    #    ax = fig.add_subplot(3,5,i+1, title='class: ' + str(df.loc[rndperm[i],'label']) )
        
    #    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((224,224)).astype(float))

    #plt.show()
    
    from sklearn.decomposition import PCA


    #pca = PCA(n_components=30)
    #pca_result = pca.fit_transform(df[feat_cols].values)

    #df['pca-one'] = pca_result[:,0]
    #df['pca-two'] = pca_result[:,1] 
    #df['pca-three'] = pca_result[:,2]


    #print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    from ggplot import *


    #chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label') ) + geom_point(size=75,alpha=0.8) + ggtitle("First and Second Principal Components colored by digit")
    #print(chart)
    
    import time

    from sklearn.manifold import TSNE

    n_sne = 6000

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=int(args.perplexy), n_iter=int(args.n_iter))
    tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)


    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    df_tsne = df.loc[rndperm[:n_sne],:].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]

    chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
            + geom_point(size=70,alpha=0.1) \
            + ggtitle("tSNE dimensions colored by digit")
    print(chart)