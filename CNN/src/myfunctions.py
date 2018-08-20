#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <raheleh.salehi@studenti.polito.it>

import warnings
warnings.filterwarnings("ignore")
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import itertools

def list_generator(folder, output, imagesformat, i=0):
        """ This function lets to obtain a .txt file containing
            in the firs column the path of each image stored in
            some subfolders stored in a main folder and in the
            second column their corresponding numeric label.
        Parameters:
                - folder: str
                        It is the path to a folder having the following structure:
                                - Folder
                                        + subfolder1
                                        + ......
                                        - subfolderN
                                                - image1
                                                - image2
                                                - ......
                - output: str
                        It is the name of the output txt file
                - imagesformat: string
                        It must be a valid format, for example 'png'
                - i: int
                        It is the first numeric label
        """
        s=os.sep
        classes_list = sorted(os.listdir(folder))
        f=open(output+'.txt', 'w')
        for c in classes_list:
                path=folder+s+c+'/*.'+imagesformat
                for img_name in glob.glob(path): 
                        f.write(img_name+' '+str(i)+'\n')
                i+=1
        f.close()
        
        
def folders_creator(directory,subdirectories=None):
        """ This function lets to create a directory if it doesn't exist yet
            and also some subdirectories.
        Parameters:
                - directory: str
                        Name of the directory to create
                - subdirectories: list
                        a list contaninig some string, each of them is the name
                        of a subdirectory to create
        """   
            
        try:
                os.stat(directory)
        except:
                os.mkdir(directory)	
        s=os.sep
        if subdirectories!=None:	
                for subdirectory in subdirectories:
                        try:
                                os.stat(directory+s+subdirectory)
                        except:
                                os.mkdir(directory+s+subdirectory)
                                
def visualize_activations(feat):
        """ This function lets to obtain a 2D matrix from a 3D matrix.
        Parameters:
                - feat: 3D array-like (A x B x C)
                        Output of a CNN layer
        Return:
                - M: 2D array-like (B*int(sqrt(A)) x C*int(sqrt(A)))
                        a list contaninig some string, each of them is the name
                        of a subdirectory to create
        """ 
        d,m,n=feat.shape
        rd = int(np.ceil(np.sqrt(d)))
        M=np.zeros((m*rd,n*rd))
        
        count=0
        for i in range(rd):
                for j in range(rd):
                        if count >= d: 
                                break
                        sliceM, sliceN = j * m, i * n
                        M[sliceN:sliceN + n, sliceM:sliceM + m] = feat[count,:, :]
                        count=count+1		
        return M
        

def save_as_image(data,name):
        """ Saves a numpy array as image with the given name.
        Parameters:
                - data: array-like
                        image in numpy array format
                - name: str
                        name to save the image
        """
        #Rescale to 0-255 and convert to uint8
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(name)
        
def make_valid_name(string):
        """ Makes a string a valid name for a file
        Parameters:
                - string: str
        Return:
                - name: str
        """             
        invalid=['<','>',':','"','/','|','?','*']
        l=list(string)
        for i in range(len(l)):
                if l[i] in invalid:
                        l[i]="_"
        name=''.join(l)
        return name
        
def get_categories(train_img_ids):
        """ Returns the list of the last subdirectories of a list of paths.
        Parameters:
                - train_img_ids: array-like
        Return:
                - categories: list
        """  
        categories=[]
        for i in range(train_img_ids.shape[0]):
                tmp=train_img_ids[i].decode('UTF-8')
                tmpl=tmp.replace('/', os.sep)
                tmpl=tmpl.replace('\\', os.sep)
                tmp=''.join(tmpl)
                clas=tmp.split(os.sep)[-2]
                if clas not in categories:
                        categories.append(clas)
        return categories

        
def imagesfolder_to_gif(filename,folder,duration):
        """ Creates a .gif image from a folder containing images.
        Parameters:
                - filename: str
                        name of the output GIF image
                - folder: str
                        path of a folder containing images
                - duration: float
                        duration of the GIF in seconds
        """
        imgFiles = sorted((fn for fn in os.listdir(folder) if fn.endswith('.png')))
        images = [imageio.imread(folder+os.sep+fn) for fn in imgFiles]
        imageio.mimsave(filename,images,duration=duration)


def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
        else:
                print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
def cm_score(cm):
        """ Return the accuracy from a confusion matrix
        Parameters:
                - cm: array-like
                        confusion matrix
        Return:
                - score: float
                        accuracy
        """
        assert cm.shape[0]==cm.shape[1]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        score=0
        for i in range(cm.shape[0]):
                score=score+cm[i,i]
        score=float(score)/cm.shape[0]
        return score
        
def stdev_out_of_diagonal(cm):
        """ Return the standard deviation of the elements outside the diagonal in a confusion matrix
        Parameters:
                - cm: array-like
                        confusion matrix
        Return:
                - s: float
                        standard deviation outside the diagonal
        """
        assert cm.shape[0]==cm.shape[1]
        l=[]
        for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                        if j!=i:
                                l.append(cm[i,j])
        l=np.asarray(l)
        l.astype(np.float32)
        s=np.std(l, ddof=1)
        return s
        
def get_intersection(f,methods):
        """ Return the intersection between a dictionary keys and a list.
        Parameters:
                - f: dict
                - methods: list
        Return:
                - out: list
        """
        datasets=list(f.keys())
        out=list(set(datasets).intersection(methods))
        return out

def is_outlier(points, thresh=3.5):
        """Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
                - points : array-like
                        An numobservations by numdimensions array of observations
                - thresh : float
                           The modified z-score to use as a threshold. Observations with
                           a modified z-score (based on the median absolute deviation) greater
                           than this value will be classified as outliers.

        Returns:
                - mask : A numobservations-length boolean array.

        References:
        ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        if len(points.shape) == 1:
                points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh
