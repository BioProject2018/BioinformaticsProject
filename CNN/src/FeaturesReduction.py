#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <raheleh.salehi@studenti.polito.it>

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
try:
        import src.myfunctions as mf
except:
        import myfunctions as mf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import shutil
import os

def dimensionality_reduction(TrainFeatures, TestFeatures, Method, params):
        """ It performs dimensionality reduction of a training and a test features matrix
            stored in a .h5 file each.
            It's possible to use 5 different methods for dimensionality reduction.
        _____________________________________________________________________________________

        Parameters:
                - TrainFeatures: string
                        It is the path of an .h5 file of the training features.
                        It contains at least the following datasets:
                        - 'feats':   array-like, shape (n_samples, n_features)
                        - 'labels':  array-like, shape (n_samples, )
                        - 'img_ids': array-like, shape (n_samples, )
                - TestFeatures: string
                        It is the path of an .h5 file of the test features.
                        It contains at least the same datasets.
                - Method: string
                        Possible value are:
                                -'PCA': Principal component analysis
                                -'t-SNE': t-distributed Stochastic Neighbor Embedding
                                -'TruncatedSVD': Truncated SVD
                                -'NMF': Non-Negative Matrix Factorization
                                -'LDA': Linear Discriminant Analysis
                - params: dict
                        It is a dictionary containig parameters for the selected estimator.
                        Keys and possible values are listed on the following websites:
                        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
                        http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
                        http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
                        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
                        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
                        For t-SNE, an additional key is needed: params['reduce'] with possible values 'TruncatedSVD','PCA','None'.
                        It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD
                        for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is
                        very high. This will suppress some noise and speed up the computation of pairwise distances between samples.
                        - params['reduce']='TruncatedSVD' : Truncated SVD --> t-SNE
                        - params['reduce']='PCA' : PCA --> t-SNE
                        - params['reduce']='None' : t-SNE directly

        Returns:
                - X_train: array-like, shape (n_samples, n_components) 
                - X_test:  array-like, shape (n_samples, n_components) 
                - ax: matplotlib.axes._subplots.AxesSubplot object (if n_components<=3) or None (if n_components>3)    
                Furthermore, automatically 2 new .h5 files containing 3 datasets each (one for reduced features, one for labels and one for img_ids)
                are generated in the folder Results/ReducedFeatures and also if n_components is <= 3 a scatter plot is saved in the folder
                Results/Plots

        Example usage:
                import FeaturesReduction as fr
                import matplotlib.pyplot as plt
                params={'n_components':3}
                X_train,X_test,ax=fr.dimensionality_reduction('TrainingFeatures.h5','TestFeatures.h5','PCA',params)
                plt.show()
        """ 

        s=os.sep
        # Load training features file
        train = h5py.File(TrainFeatures,'r')
        train_features = train['feats']
        train_labels= train['labels']
        train_labels=np.squeeze(train_labels)
        train_img_ids=train['img_id']
        
        # Get categories of the training set from features ids
        categories=mf.get_categories(train_img_ids)
                
        # Load test features file
        test = h5py.File(TestFeatures,'r')
        test_features = test['feats']
        test_labels= test['labels']
        test_labels=np.squeeze(test_labels)
        test_img_ids=test['img_id']

        n_comp=params['n_components']
                
        if Method!='NMF':  
                # Standardize features by removing the mean and scaling to unit variance
                scaler=StandardScaler().fit(train_features)
                train_features=scaler.transform(train_features)
                test_features=scaler.transform(test_features)

        if Method=='PCA':
                # Get PCA model
                pca=PCA()
                # Set parameters
                pca.set_params(**params)
                # Fit the model with the training features and
                # apply dimensional reduction to training features	
                X_train = pca.fit_transform(train_features)
                # Apply dimensional reduction to test features			
                X_test = pca.transform(test_features)

        elif Method=='NMF':
                params['verbose']=True
                # Get NMF model
                nmf= NMF()
                # Set parameters
                nmf.set_params(**params)
                # Fit the model with the training features and
                # apply dimensional reduction to training features
                X_train = nmf.fit_transform(train_features)
                # Apply dimensional reduction to test features
                X_test = nmf.transform(test_features)

        elif Method=='LDA':
                # Get LDA model
                lda=LDA()
                # Set parameters
                lda.set_params(**params)
                # Fit the model with the training features
                #lda.fit(train_features,train_labels)
                # apply dimensional reduction to training features	
                #X_train = lda.transform(train_features)
                
                X_train=lda.fit_transform(train_features,train_labels)
                # apply dimensional reduction to training features	
                #X_train = lda.transform(train_features)
                # Apply dimensional reduction to test features
                X_test = lda.transform(test_features)

        elif Method=='t-SNE':
                red=params['reduce']
                del params['reduce']
                print(red)
                params['verbose']=True

                # Use another dimensionality reduction method (PCA for dense
                # data or TruncatedSVD for sparse data) to reduce the number of
                # dimensions to a reasonable amount (e.g. 50) if the number of
                # features is very high. This will suppress some noise and speed
                # up the computation of pairwise distances between samples.
                if n_comp<50:
                        K=50
                else:
                        K=n_comp*2      
                if red=='TruncatedSVD':
                        # Get TruncatedSVD model
                        svd = TruncatedSVD(n_components=K)
                        # Fit the model with the training features and
                        # apply dimensional reduction to training features	
                        train_features = svd.fit_transform(train_features)
                        # Apply dimensional reduction to test features			
                        test_features = svd.transform(test_features)
                elif red=='PCA':
                        # Get PCA model
                        pca=PCA(n_components=K)
                        # Fit the model with the training features and
                        # apply dimensional reduction to training features	
                        train_features = pca.fit_transform(train_features)
                        # Apply dimensional reduction to test features			
                        test_features = pca.transform(test_features)
                else:
                        pass
                
                # Get t-SNE model
                tsne = TSNE()
                # Set parameters
                tsne.set_params(**params)
                # Concatenate training and test set
                n_train=train_features.shape[0]
                features=np.concatenate((train_features,test_features),axis=0)
                
                # Fit the model with the data and apply dimensional reduction
                X=tsne.fit_transform(features)
                
                # Separate training and test set
                X_train = X[:n_train,:]	
                X_test = X[n_train:,:]

        elif Method=='TruncatedSVD':
                # Get TruncatedSVD model
                svd = TruncatedSVD()
                # Set parameters
                svd.set_params(**params)
                # Fit the model with the training features and
                # apply dimensional reduction to training features	
                X_train = svd.fit_transform(train_features)
                # Apply dimensional reduction to test features			
                X_test = svd.transform(test_features)
        
        else:
                raise TypeError("Invalid method: possible methods are 'PCA', 't-SNE', 'TruncatedSVD', 'NMF' and 'LDA'")

        # Create folder in which save reduced features
        mf.folders_creator('Results',['ReducedFeatures'])
                               
        # Create an .h5 file and store in it reduced training set
        name='Results'+s+'ReducedFeatures'+s+Method+str(n_comp)+'_'+TrainFeatures.split(s)[-1].split('.')[0]+'.h5'
        f = h5py.File(name,"w")
        f.create_dataset('img_id', data=train_img_ids[:],dtype="S40")
        f.create_dataset('labels', data=train_labels.T, compression="gzip")
        if Method=='PCA':
                f.create_dataset('pca', data=X_train.T, compression="gzip")
        elif Method=='t-SNE':
                f.create_dataset('tsne', data=X_train.T, compression="gzip")
        elif Method=='TruncatedSVD':
                f.create_dataset('tsvd', data=X_train.T, compression="gzip")
        elif Method=='LDA':
                f.create_dataset('lda', data=X_train.T, compression="gzip")
        elif Method=='NMF':
                f.create_dataset('nmf', data=X_train.T, compression="gzip")
        f.close()

        # Create an .h5 file and store in it reduced test set
        name='Results'+s+'ReducedFeatures'+s+Method+str(n_comp)+'_'+TestFeatures.split(s)[-1].split('.')[0]+'.h5'
        f = h5py.File(name,"w")
        f.create_dataset('img_id', data=test_img_ids[:],dtype="S40")
        f.create_dataset('labels', data=test_labels.T, compression="gzip")
        if Method=='PCA':
                f.create_dataset('pca', data=X_test.T, compression="gzip")
        elif Method=='t-SNE':
                f.create_dataset('tsne', data=X_test.T, compression="gzip")
        elif Method=='TruncatedSVD':
                f.create_dataset('tsvd', data=X_test.T, compression="gzip")
        elif Method=='LDA':
                f.create_dataset('lda', data=X_test.T, compression="gzip")
        elif Method=='NMF':
                f.create_dataset('nmf', data=X_test.T, compression="gzip")
        f.close()

        if n_comp<4:
                
                # Get folders list of the test set from features ids
                test_folders=mf.get_categories(test_img_ids)
                # Get number of folders
                n_folders_test=len(test_folders)
                # Make some names for the plot legend
                tf=[]
                for i in range(n_folders_test):
                        tf.append('Test'+str(i))
                
                # Define a list of colors in exadecimal format
                if len(categories)+n_folders_test<9:
                        colors=['#FF0000','#00FF00','#0000FF','#FFFF00','#00FFFF','#808080','#FF00FF','#000000']
                else:
                        n=250
                        max_value = 255**3
                        interval = int(max_value / n)
                        colors = ['#'+hex(i)[2:].zfill(6) for i in range(0, max_value, interval)]
                        colors=colors[:int((n+1)/10*9)]
                        random.shuffle(colors)
                
                # Create a folder to save images
                mf.folders_creator('Results',['Plots'])

                # Create a name to save image
                name=Method+str(n_comp)+'_'+TrainFeatures.split(s)[-1].split('.')[0]
                name=name.split('_')
                name='_'.join(name[:-1])
                
                
                print(X_train.shape)
                print(X_test.shape)

                if n_comp==1:
                        # Plot 1D Data with different colors
                        fig,ax=plt.subplots()
                        for i in range(len(categories)):
                                ax.scatter(X_train[train_labels==i,0],np.ones(X_train[train_labels==i,0].shape), c=colors[i], label=categories[i])
                        k=len(categories)
                        for i in range(n_folders_test):
                                ax.scatter(X_test[test_labels==i,0], np.ones(X_test[test_labels==i,0].shape), c=colors[k], label=tf[i])
                                k+=1
                        ax.legend()
                        
                        # Save image in .png format
                        plt.savefig('Results'+s+'Plots'+s+name+'.png')
                
                if n_comp==2:
                        # Plot 2D Data with different colors
                        fig,ax=plt.subplots()
                        for i in range(len(categories)):
                                ax.scatter(X_train[train_labels==i,0], X_train[train_labels==i,1], c=colors[i], label=categories[i])
                        k=len(categories)
                        for i in range(n_folders_test):
                                ax.scatter(X_test[test_labels==i,0], X_test[test_labels==i,1], c=colors[k], label=tf[i])
                                k+=1
                        ax.legend()
                        
                        # Save image in .png format
                        plt.savefig('Results'+s+'Plots'+s+name+'.png')

                        # Remove outliers
                        out_train=mf.is_outlier(X_train, thresh=3.5)
                        out_test=mf.is_outlier(X_test, thresh=3.5)
                        out_train=np.logical_not(out_train)
                        out_test=np.logical_not(out_test)

                        X_train2=X_train[out_train,:]
                        X_test2=X_test[out_test,:]

                        if X_train2.shape[0]!=X_train.shape[0] or X_test2.shape[0]!=X_test.shape[0]:

                                train_labels2=train_labels[out_train]
                                test_labels2=test_labels[out_test]
                                
                                # Plot 2D Data without outliers with different colors
                                fig,ax=plt.subplots()
                                for i in range(len(categories)):
                                        ax.scatter(X_train2[train_labels2==i,0], X_train2[train_labels2==i,1], c=colors[i], label=categories[i])
                                k=len(categories)
                                for i in range(n_folders_test):
                                        ax.scatter(X_test2[test_labels2==i,0], X_test2[test_labels2==i,1], c=colors[k], label=tf[i])
                                        k+=1
                                ax.legend()
                                
                                # Save image in .png format
                                plt.savefig('Results'+s+'Plots'+s+name+'_noOutliers.png')
                
                if n_comp==3:
                        mf.folders_creator('Results'+s+'Plots',['tmp'])
                        # Plot 3-D Data with different colors
                        ax=plt.subplot(111, projection='3d')
                        for i in range(len(categories)):
                                ax.scatter(X_train[train_labels==i,0], X_train[train_labels==i,1],X_train[train_labels==i,2], c=colors[i], label=categories[i])
                        k=len(categories)
                        for i in range(n_folders_test):
                                ax.scatter(X_test[test_labels==i,0], X_test[test_labels==i,1],X_test[test_labels==i,2], c=colors[k], label=tf[i])
                                k+=1
                        ax.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0,0))

                        # Rotate for 360° and save every 10° 
                        for angle in range(0,360,10):
                                ax.view_init(30,angle)
                                plt.savefig('Results'+s+'Plots'+s+'tmp'+s+name+str(angle)+'.png')
                        # Save as a .gif image
                        mf.imagesfolder_to_gif('Results'+s+'Plots'+s+name+'.gif','Results'+s+'Plots'+s+'tmp',0.2)
                        shutil.rmtree('Results'+s+'Plots'+s+'tmp')

        else:
                ax=None
                        
        return X_train, X_test, ax
                        

