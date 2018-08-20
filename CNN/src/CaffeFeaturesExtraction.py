#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <raheleh.salehi@studenti.polito.it>

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import os
import caffe
try:
        import src.myfunctions as mf
except:
        import myfunctions as mf
from tkinter import Label
from tkinter.ttk import Progressbar
import time
import progressbar
import os


def Caffe_model_path(name):
        """ This function lets, giving as input a Caffe pre-trained CNN model name,
            to get the path of the caffemodel file, the path of the prototxt file, the size
            that input images must have and the mean vector of the data set on which the
            net was pre-trained
        _____________________________________________________________________________________
        
        Parameters:
                - name: string
                        It is the name of a Caffe pre-trained CNN model.
                        Possible value are (it is not case sensitive):
                                        - 'caffe alexnet'
                                        - 'caffe googlenet'
                                        - 'caffe vgg-16'
        Returns:
                - caffemodel: string
                        It is the path for the Caffe pre-trained model file for the selected value of name
                - prototxt: string
                        It is the path for the prototxtfile for the selected value of name
                - mean_v: array-like
                        the mean vector of the data set on which the net was pre-trained
                - crop_size: int
                        It is the size the input images must have (img.shape=(crop_size,crop_size,3))

        Example usage:
                import CaffeFeaturesExtraction as cfe
                caffemodel, prototxt, crop_size, mean_v=cfe.Caffe_model_path('caffe alexnet')
        
        """ 
        s=os.sep
        ''' Return caffemodel, prototxt, mean vector and crop size for the
                input model name (string)'''
                
        if name.lower()=='caffe alexnet':		
                caffemodel='models'+s+'alexnet.caffemodel'
                prototxt='models'+s+'alexnet_deploy.prototxt'
                crop_size=227
                meanfile='models'+s+'ilsvrc_2012_mean.npy'
                mean_v = np.load(meanfile).mean(1).mean(1)

        elif name.lower()=='caffe googlenet':
                caffemodel='models'+s+'googlenet.caffemodel'
                prototxt='models'+s+'googlenet_deploy.prototxt'
                crop_size=224
                mean_v=np.asarray([104.0, 117.0, 123.0])
        elif name.lower()=='caffe vgg-16':
                caffemodel='models'+s+'vgg-16.caffemodel'
                prototxt='models'+s+'vgg-16_deploy.prototxt'
                crop_size=224
                mean_v=np.asarray([103.939, 116.779, 123.68])
        else:
                raise TypeError("Invalid model: possible models are 'caffe alexnet', 'caffe googlenet' and 'caffe vgg-16'")

        return caffemodel, prototxt, crop_size, mean_v

def Caffe_get_net_and_transformer(caffemodel, prototxt, mean_v, crop_size):
        """ This function lets to get a Caffe net and its transformer object
        _____________________________________________________________________________________

        Parameters:
                - caffemodel: string
                        It is the path of a Caffe pre-trained model file
                - prototxt: string
                        It is the path of a prototxt file
                - mean_v: array-like
                        the mean vector of the data set on which the net was pre-trained
                - crop_size: int
                        It is the size the input images must have (img.shape=(crop_size,crop_size,3))
        Returns:
                - net:
                        It is the caffe net for the given inputs
                - transformer:
                        It is the caffe transformer for the net

        Example usage:
                import CaffeFeaturesExtraction as cfe
                caffemodel, prototxt, crop_size, mean_v=cfe.Caffe_model_path('caffe alexnet')
                net,transformer=cfe.Caffe_get_net_and_transformer(caffemodel, prototxt, mean_v, crop_size)
        
        """	
        # define net
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        
        # we process one image at time so we change batch size from 10 to 1
        net.blobs['data'].reshape(1,3,crop_size,crop_size)
        
        # set up transformer - creates transformer object
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        # transpose image from HxWxC to CxHxW
        transformer.set_transpose('data', (2,0,1))
        
        # subtract by mean
        transformer.set_mean("data", mean_v)

        # swap image channels from RGB to BGR
        transformer.set_channel_swap('data', (2,1,0))

        # set raw_scale = 255 to multiply with the values loaded with caffe.io.load_image
        transformer.set_raw_scale('data', 255)
        
        return net,transformer

def Caffe_extract_feature(net, transformer, img, crop_size, layer):
        """ This function lets to extract features from a single image using a Caffe pre-trained CNN model.
        _____________________________________________________________________________________

        Parameters:
                - net: 
                        It is a pre-trained net.
                        It must be obtained as output of Caffe_get_net_and_transformer.
                - transformer:
                        It is transformer object of the net.
                        It must be obtained as output of Caffe_get_net_and_transformer.
                - img: string
                        It is the path to the image to process
                - crop_size: int
                        It is the size the input images must have for the base_model
                - layer: string
                        It is the name of the layer from which extract features.

        Returns:
                - feats: array-like
                        Extracted features

        Example usage:
                import CaffeFeaturesExtraction as cfe
                img='image1.png'
                caffemodel, prototxt, crop_size, mean_v=cfe.Caffe_model_path('caffe alexnet')
                net,transformer=cfe.Caffe_get_net_and_transformer(caffemodel, prototxt, mean_v, crop_size)
                feats=Caffe_extract_feature(net, transformer, img, crop_size, layer)

        """	
        # preprocess image
        preprocessed_img = transformer.preprocess("data", img[:,:crop_size,:crop_size])
        
        # get output from the net
        out = net.forward_all(**{net.inputs[0]: preprocessed_img, "blobs": [layer]})
        
        # get features from a specific layer
        feat = out[layer]
        feat = feat[0]

        return feat


def Caffe_multiple_features_extraction(folder,caffemodel, prototxt, mean_v, layer, crop_size, cuda, save_activations,out_prefix, GUI=None):
        """ This function lets to extract features from an image using a Keras pre-trained CNN model.
        _____________________________________________________________________________________

        Parameters:
                - folder: string
                        It is the path to a folder containing the images from which extract features.
                        For projectual reason imposed by the full project, this folder must have the following
                        structure:
                                - Folder
                                        + subfolder1
                                        + ......
                                        - subfolderN
                                                - image1
                                                - image2
                                                - ......
                - caffemodel: string
                        It is the path of a Caffe pre-trained model file
                - prototxt: string
                        It is the path of a prototxt file
                - mean_v: array-like
                        the mean vector of the data set on which the net was pre-trained
                - crop_size: int
                        It is the size the input images must have (img.shape=(crop_size,crop_size,3)) for the model       
                - layer: string
                        It is the name of the layer from which extract features.
                - cuda: bool
                        If it is True GPU will be used for the extraction, else only CPU will be used.
                        To set this parameter on True you need to have CUDA drivers installed.
                - save_activations: bool
                        if it is True all the activations of all images will be saves as
                        .png files in Results/Activations folder.
                - out_prefix: string
                        It is the prefix of the name of the output file. Infact, during
                        the process a .h5 file will be generated.
                        This file containg extracted features for all images, labels (intended
                        as subfolder names)of all images (intended as subfolder names), and images
                        list of addresses. All these things are stored respectivelly in the following
                        datasets: 'feats','labels','img_ids'.
                - GUI: object (default None)
                        Do not consider this parameter. This parameter was created expressly for the
                        usage of this function with "FeaturesExtraction_GUI" in order to update the
                        progress bar in the GUI while extraction is in progress.

        Returns:
                This function don't return any variable, but during the process a .h5 file will be generated.
                This file containg extracted features for all images, labels (intended as subfolder names)
                of all images (intended as subfolder names), and images list of addresses. All these things
                are stored respectivelly in the following datasets: 'feats','labels','img_ids'.

        Example usage:
                import CaffeFeaturesExtraction as cfe
                model='caffe alexnet'
                layer= 'conv1'
                folder='Training'
                cuda=False
                save_activation=True
                out_prefix=model+'_'+layer
                caffemodel, prototxt, crop_size, mean_v=cfe.Caffe_model_path(model)
                Caffe_multiple_features_extraction(folder,caffemodel, prototxt, mean_v, layer, crop_size, cuda, save_activations,out_prefix, GUI=None)
                
        """	
        s=os.sep
        # Get a proper name for the .txt list of images
        gender=folder.split(s)[-1]
        
        # Create a list of images
        mf.list_generator(folder, gender, 'png')
        
        # Create folders in which store features and activations files
        mf.folders_creator('Results',['Features','Activations'])
        
        # Read .txt list of images and load it as list
        f=open(gender+'.txt','r')
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        f.close()
        
        
        # Take 1st word of each line as an image name, creating a names list
        img_list = [line.split(' ')[0] for line in lines]
        # Take 2nd word of each line as a class, creating a labels list
        lab = [line.split(' ')[1] for line in lines]
        
        if GUI!=None:
                # If GUI mode is ON, initialize a Progress bar
                currentImage=Label(GUI, text=' '*50)
                currentImage.grid(row = 0, column =1)
                progress = Progressbar(GUI, orient="horizontal",length=200, mode="determinate")
                progress.grid(row=1, column=1)
                value = 0
                progress["value"] = 0
                progress["maximum"] = len(lines)
                step=1
        
        if cuda==True:	
                # Set Caffe GPU mode		
                caffe.set_mode_gpu()
        else:
                # Set Caffe CPU only mode		
                caffe.set_mode_cpu()
        
        # Get net and transformer
        net, transformer = Caffe_get_net_and_transformer(caffemodel, prototxt, mean_v, crop_size)
        
        # Initialize some usefull empty lists 
        feats=[]
        img_ids=[]
        labels=[]
        
        # Initialize to 0 some counters and flag
        i=0
        count=0
        file_created=0
        
        
        # Process all images in the list
        for im in img_list:
                
                if i==0:
                        # Clear terminal
                        try:
                                os.system('clear')
                        except:
                                os.system('cls')
                        print('Features Extraction from ' + folder)
                        # Initialize progress bar
                        bar=progressbar.ProgressBar(max_value=len(lines))
                
                # Update progress bar
                bar.update(i)
                
                # Load image
                img=caffe.io.load_image(im)
                
                # Extract features from image
                feat = Caffe_extract_feature(net, transformer, img, crop_size, layer)

                
                if save_activations==True:
                        if i==0:
                                # Create folders in which store features and activations files
                                mf.folders_creator('Results'+s+'Activations'+s+gender)
                                # Visualize activations
                        M=mf.visualize_activations(feat)
                
                # Reshape features matrix to an array
                feat=feat.reshape((1,feat.size))
                feat=np.squeeze(feat)
                
                # Append features array to features list
                feats.append(feat)
                # Append label to labels list
                labels.append(int(lab[i]))
                
                if GUI!=None:
                        # If GUI mode is ON, update Progress bar value
                        a=value/len(lines)*100
                        currentImage["text"]='Processing...'+"{0:.1f}".format(round(a,1)) +' % '
                        currentImage.update()
                        value += step
                        progress["value"] = int(value)
                        progress.update()
                
                # Abbreviete current image path and append it to image ids list
                tmp=im.split(s)
                im=s.join([tmp[-2],tmp[-1]])
                img_ids.append(np.string_(im))
                
                
                if save_activations==True:
                        # Write activations
                        n='Results'+s+'Activations'+s+gender+'/'+out_prefix+'_'+im.split(s)[1]
                        mf.save_as_image(M,n)
        
                if file_created == 0:
                        # Create an Hierarchical Data Format (.h5) file in which
                        # store Features, Labels and Names of the processed images
                        feats_shape = [len(img_list),feat.shape[0]];
                        name='Results'+s+'Features'+s+out_prefix+'_'+gender+'.h5'
                        f = h5py.File(name,"w")
                        feats_dset = f.create_dataset("feats",feats_shape,compression="gzip")
                        img_id_dset = f.create_dataset("img_id",(len(img_list),),dtype="S40")
                        labels_dset = f.create_dataset("labels",(len(img_list),),compression="gzip")
                        file_created = 1
                        
                N=100			
                if (i+1) % N == 0:
                        # Every N processed images save data to the created .h5 file
                        # and clean the list variables to empty the RAM
                        
                        # Convert lists to numpy arrays
                        feats=np.asarray(feats)
                        labels=np.asarray(labels)
                        
                        # Write data 
                        feats_dset[i-N+1:i+1] = feats 
                        img_id_dset[i-N+1:i+1] = img_ids
                        labels_dset[i-N+1:i+1] = labels
                        
                        # Clean list variables
                        feats = []
                        labels = []
                        img_ids=[]
                        
                        count=-1
                
                i+=1
                count+=1
                
        # Convert lists to numpy arrays
        feats=np.asarray(feats)
        labels=np.asarray(labels)
        # Write remaining data
        feats_dset[i-count:i+1] = feats
        img_id_dset[i-count:i+1] = img_ids
        labels_dset[i-count:i+1] = labels
        # Clean list variables
        feats = []
        labels = []
        img_ids=[]
        
        # Close file
        f.close()
        
        if GUI!=None:
                # If GUI mode is ON, update Progress bar value
                currentImage["text"]='         Completed                      '
                currentImage.update()
                value += step
                progress["value"] = int(value)
                progress.update()




if __name__ == "__main__":

        Layers=["conv1/7x7_s2","inception_5b/output","conv2/3x3","pool3/3x3_s2","pool5/7x7_s1","inception_3a/1x1","inception_3a/output_inception_3a/output_0_split_2","inception_3b/output","inception_4a/output","inception_4a/output_inception_4a/output_0_split_0","inception_4e/3x3","inception_5a/3x3","inception_5b/1x1"]

        for layer in Layers:
                try:	
                        model='caffe googlenet'
                        TrainFolder='DataSet/Training'
                        TestFolder='DataSet/Test'
                        
                        save_activation=True
                        GPU_ON=False
                        out_prefix=model+'_'+layer
                                
                        # Get Model paths and settings
                        caffemodel, prototxt, crop_size, mean_v = Caffe_model_path(model)

                        # Start features extraction
                        out_prefix=model+'_'+layer
                        out_prefix=mf.make_valid_name(out_prefix)
                        Caffe_multiple_features_extraction(TrainFolder,caffemodel, prototxt, mean_v, layer, crop_size, GPU_ON, save_activation, out_prefix, GUI=None)
                        Caffe_multiple_features_extraction(TestFolder,caffemodel, prototxt, mean_v, layer, crop_size, GPU_ON, save_activation, out_prefix, GUI=None)
                except:
                        pass
