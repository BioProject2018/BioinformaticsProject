#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <raheleh.salehi@studenti.polito.it>

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import os
try:
        import src.myfunctions as mf
except:
        import myfunctions as mf
from tkinter import Label
from tkinter.ttk import Progressbar
import time
import progressbar

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

from keras.applications.vgg16 import preprocess_input as ppi_vgg16
from keras.applications.vgg19 import preprocess_input as ppi_vgg19
from keras.applications.resnet50 import preprocess_input as ppi_resnet50
from keras.applications.inception_v3 import preprocess_input as ppi_inception_v3

from keras.preprocessing import image
from keras.models import Model


def Keras_model_path(name):
        """ This function lets, giving as input a Keras pre-trained CNN model name,
            to get the model (these models don't include top layers), the preprocessing
            function and the size that input images must have
        _____________________________________________________________________________________

        Parameters:
                - name: string
                        It is the name of a Keras pre-trained CNN model.
                        Possible value are (it is not case sensitive):
                                - 'keras vgg-16'
                                - 'keras vgg-19'
                                - 'keras resnet50'
                                - 'keras inception_v3'

        Returns:
                - base_model: object
                        It is the Keras pre-trained model for the selected name(these models don't include top layers)
                - preprocessing: function
                        It is the function for the preprocessing
                - crop_size: int
                        It is the size the input images must have (img.shape=(crop_size,crop_size,3))

        Example usage:
                import KerasFeaturesExtraction as kfe
                base_model,preprocessing,crop_size=kfe.Keras_model_path('keras vgg-16')
        
        """ 
        s=os.sep		
        if name.lower()=='keras vgg-16':
                base_model = VGG16(weights='models'+s+'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
                preprocessing=ppi_vgg16
                crop_size=224

        elif name.lower()=='keras vgg-19':
                base_model = VGG19(weights='models'+s+'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
                preprocessing=ppi_vgg19
                crop_size=224
                
        elif name.lower()=='keras resnet50':
                base_model = ResNet50(weights='models'+s+'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
                preprocessing=ppi_resnet50
                crop_size=224
                
        elif name.lower()=='keras inception_v3':
                base_model = InceptionV3(weights='models'+s+'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
                preprocessing=ppi_inception_v3
                crop_size=299
                
        else:
                raise TypeError("Invalid model: possible models are 'keras vgg-16', 'keras vgg-19' and 'keras resnet50', 'keras inception_v3'")

        return base_model, preprocessing, crop_size


def Keras_extract_feature(base_model, preprocessing, img, crop_size, layer=''):
        """ This function lets to extract features from a single image using a Keras pre-trained CNN model.
        _____________________________________________________________________________________

        Parameters:
                - base_model: object
                        It is the Keras pre-trained model.
                - preprocessing: function
                        It is the function for the preprocessing.
                - img: string
                        It is the path to the image to process
                - crop_size: int
                        It is the size the input images must have for the base_model
                - layer: string (default: '')
                        It is the name of the layer from which extract features. If layer=''
                        features will be extracted from the last layer

        Returns:
                - feats: array-like
                        Extracted features

        Example usage:
                import KerasFeaturesExtraction as kfe
                img='image1.png'
                base_model,preprocessing,crop_size=kfe.Keras_model_path("keras vgg-16")
                layer="block1_conv2"
                feats=kfe.Keras_extract_feature(base_model, preprocessing, img, crop_size, layer)       

        """
        # Set inputs as attribute of Model class
        if layer=='':
                model=base_model
        else:
                model = Model(input=base_model.input, output=base_model.get_layer(layer).output)
        # Load and resize image
        I = image.load_img(img, target_size=(crop_size, crop_size))
        # Convert image to array
        x = image.img_to_array(I)
        # Expand dimension
        x = np.expand_dims(x, axis=0)
        # Prerform preprocessing
        x = preprocessing(x)
        # Extract features
        feat = model.predict(x)

        return feat


def Keras_multiple_features_extraction(folder,base_model, preprocessing, layer, crop_size, save_activations,out_prefix, GUI=None):
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
                - base_model: object
                        It is the Keras pre-trained model.
                - preprocessing: function
                        It is the function for the preprocessing.
                - crop_size: int
                        It is the size the input images must have for the base_model
                - layer: string (default: '')
                        It is the name of the layer from which extract features. If layer=''
                        features will be extracted from the last layer
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
                
                        #################

        Returns:
                This function don't return any variable, during the process a .h5 file will be generated.
                This file containg extracted features for all images, labels (intended as subfolder names)
                of all images (intended as subfolder names), and images list of addresses. All these things
                are stored respectivelly in the following datasets: 'feats','labels','img_ids'.

        Example usage:
                import KerasFeaturesExtraction as kfe
                model="keras vgg-16"
                base_model,preprocessing,crop_size=kfe.Keras_model_path(model)
                layer="block1_conv2"
                folder="Training"
                out_prefix=model+'_'+layer
                save_activation=True
                GUI=None
                kfe.Keras_multiple_features_extraction(folder,base_model, preprocessing,layer,
                                                       crop_size, save_activations,out_prefix, GUI=None)
        """	
        # Get a proper name for the .txt list of images
        s=os.sep
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
                
                
                # Extract features from image
                feat=Keras_extract_feature(base_model, preprocessing, im, crop_size, layer)
                
                
                if save_activations==True:
                        if i==0:
                                # Create folders in which store features and activations files
                                mf.folders_creator('Results'+s+'Activations'+s+gender)
                                # Visualize activations
                        M=mf.visualize_activations(np.swapaxes(np.squeeze(feat),2,0))
                
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
                        n='Results'+s+'Activations'+s+gender+s+out_prefix+'_'+im.split(s)[1]
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
        import json
        data=open('data/layers.json').read()
        data=json.loads(data)
        Layers=data["Keras VGG-16"]
        
        for layer in Layers:	
                model='keras vgg-16'
                TrainFolder='DataSet/Training'
                TestFolder='DataSet/Test'
                save_activation=True
                
                out_prefix=model+'_'+layer
                        
                # Get Model paths and settings
                base_model, preprocessing, crop_size = Keras_model_path(model)

                # Start features extraction
                out_prefix=model+'_'+layer
                out_prefix=mf.make_valid_name(out_prefix)
                Keras_multiple_features_extraction(TrainFolder,base_model, preprocessing, layer, crop_size, save_activation, out_prefix, GUI=None)
                Keras_multiple_features_extraction(TestFolder,base_model, preprocessing, layer, crop_size, save_activation, out_prefix, GUI=None)
