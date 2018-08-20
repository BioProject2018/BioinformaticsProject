#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <raheleh.salehi@studenti.polito.it>

import warnings
warnings.filterwarnings("ignore")
from tkinter import*
from tkinter.ttk import Button as Button2 
from tkinter import Tk, Entry, Button, Label, IntVar, Checkbutton
from tkinter.ttk import Combobox, Progressbar
from tkinter.filedialog import askdirectory
from tkinter.font import Font
import src.myfunctions as mf
import numpy as np
import json
import h5py
import os
from PIL import Image
global CAFFE
global KERAS
try:
        import src.CaffeFeaturesExtraction as cfe
        CAFFE=1
except:
        CAFFE=0
try:
        import src.KerasFeaturesExtraction as kfe
        KERAS=1
except:
        KERAS=0


class FeaturesExtractionApp(Tk):
        """ It is a GUI to perform Features extraction using Convolutional Neural Network (CNN)
            from a training set and a test set of images stored in appropriate folders.
            Possible pre-trained CNN models are:
                - Caffe AlexNet
                - Caffe VGG-16
                - Caffe GoogLeNet
                - Keras VGG-16
                - Keras VGG-19
                - Keras ResNet50
                - Keras Inception_V3
        __________________________________________________________________________________________

        Things to know:
                        Training image folder and test image folder must mandatory follow this structure:
                        - TrainingSet
                                + folder1
                                + folder2
                                + folder3
                                + ......
                                - folderN
                                        - image1
                                        - image2
                                        - ......
                        - TestSet
                                + folder1
                                + folder2
                                + folder3
                                + ......
                                - folderN
                                        - image1
                                        - image2
                                        - ......
                         The names of the folders can be whatever you want. The only important and mandatory
                         thing is the structure 
        """
           
        def __init__(self):
                self.s=os.sep
                # Set title and its text format
                Tk.__init__(self)
                self.title_font = Font(family='Helvetica', size=18, weight="bold", slant="italic")
                self.title("Features Extractor")
                
                # Import names of CNN layers
                f=open('src'+self.s+'layers.json','r')
                self.layers=f.read()
                f.close()
                self.layers=json.loads(self.layers)
                
                # Initialize variables
                self.TrainFolder=''
                self.TestFolder=''
                self.model=''
                self.layer=''
                self.CUDA=IntVar()
                self.CUDA.set(0)
                self.activations=IntVar()
                self.activations.set(1)
                self.LayersChoices=[[]]
                self.current_activation=0
                self.ids=0
                self.error=0

                # Visualize Home page
                self.HomePage()

        def HomePage(self):
                """ Function that creates the HomePage of the GUI """
                # Update page index
                self.page=0
                # Clear current page
                self.clear()
                
                # Create Training folder button and entry
                self.trainfolder_button=Button(self, text="Training Set folder",fg="blue", command=self.get_trainfolder)
                self.trainfolder_button.grid(row = 0, column =0)
                self.trainfolder_entry = Entry(self, bd =5)
                self.trainfolder_entry.grid(row = 0, column =1)
                self.trainfolder_entry.insert(0,self.TrainFolder)
                self.trainfolder_entry.update()

                # Create Test folder button and entry
                self.testfolder_button=Button(self, text="Test Set folder",fg="blue", command=self.get_testfolder)
                self.testfolder_button.grid(row = 1, column =0)
                self.testfolder_entry = Entry(self, bd =5)
                self.testfolder_entry.grid(row = 1, column =1)
                self.testfolder_entry.insert(0,self.TestFolder)
                self.testfolder_entry.update()

                # Models menu
                if CAFFE==1 and KERAS==0:
                        self.choices = [ 'Caffe AlexNet','Caffe VGG-16','Caffe GoogLeNet']
                elif CAFFE==0 and KERAS==1:
                        self.choices = ['Keras VGG-16', 'Keras VGG-19', 'Keras ResNet50','Keras Inception_V3']
                elif CAFFE==1 and KERAS==1:
                        self.choices = [ 'Caffe AlexNet','Caffe VGG-16','Caffe GoogLeNet','Keras VGG-16', 'Keras VGG-19', 'Keras ResNet50','Keras Inception_V3']
                else:
                        raise RuntimeError('Caffe or Keras installation not detected')
                self.ModelsLabel=Label(self, text="Choose a pre-trained model")
                self.ModelsLabel.grid(row = 4, column =0)
                self.ModelsMenu =Combobox(self, textvariable=self.choices[0], values=self.choices, state= 'readonly')
                self.ModelsMenu.bind("<<ComboboxSelected>>", self.change_dropdown)
                self.ModelsMenu.grid(row = 5, column =0)
                self.ModelsMenu.set(self.model)
                
                # Layers menu
                self.LayersLabel=Label(self, text="Choose a layer")
                self.LayersLabel.grid(row = 6, column =0)
                self.LayersMenu =Combobox(self, textvariable=[], values=[],state='disabled')
                self.LayersMenu.grid(row = 7, column =0)
                self.LayersMenu.set(self.layer)

                # CUDA flag
                self.CudaFlag=Checkbutton(self, text="Use CUDA", variable=self.CUDA)
                self.CudaFlag.grid(row=3, column=1)
                
                # Save activations flag
                self.activationsFlag=Checkbutton(self, text="Save activations", variable=self.activations)
                self.activationsFlag.grid(row=4, column=1)

                # Next page
                self.NextPage = Button(self, text="Extract Features", fg="green",command=self.redirect)
                self.NextPage.grid(row=6, column=1)

                # Close program
                self.close_button = Button(self, text="Close", fg="red", command=quit)
                self.close_button.grid(row=7, column=1)

                # Show detected errors
                if self.error==1:
                        self.error=0
                        self.ErrorLabel1=Label(self, text=self.msgErr1, bg="red")
                        self.ErrorLabel1.grid(row = 8, column =0)
                        self.ErrorLabel1=Label(self, text=self.msgErr2, bg="red")
                        self.ErrorLabel1.grid(row = 9, column =0)


        def get_trainfolder(self):
                """ Callback for Training Folder button """
                # Select and save Training folder
                folder = askdirectory(initialdir=os.getcwd())
                folder=folder.replace('/', os.sep)
                folder=folder.replace('\\', os.sep)
                self.train_suffix=folder.split(self.s)[-1]
                self.TrainFolder=folder
                # Insert Training folder in its entry
                self.trainfolder_entry.destroy()
                self.trainfolder_entry = Entry(self, bd =5)
                self.trainfolder_entry.grid(row = 0, column =1)
                self.trainfolder_entry.insert(0,self.TrainFolder)
                self.trainfolder_entry.update()


        def get_testfolder(self):
                """ Callback for Test Folder button """
                # Select and save Test folder
                folder = askdirectory(initialdir=os.getcwd())
                folder=folder.replace('/', os.sep)
                folder=folder.replace('\\', os.sep)
                self.test_suffix=folder.split(self.s)[-1]
                self.TestFolder=folder
                # Insert Test folder in its entry
                self.testfolder_entry.destroy()
                self.testfolder_entry = Entry(self, bd =5)
                self.testfolder_entry.grid(row = 1, column =1)
                self.testfolder_entry.insert(0,self.TestFolder)
                self.testfolder_entry.update()

        def change_dropdown(self,*args):
                """ Callback for model choice menu.
                    It generates the appropriate list in layers menu """
                # Get the current selected model
                self.model=self.ModelsMenu.get()
                # Update the list of layers
                self.LayersChoices=self.layers[self.model]
                self.LayersMenu.destroy()
                self.LayersMenu =Combobox(self, textvariable=self.LayersChoices[0], values=self.LayersChoices, state= 'readonly')
                self.LayersMenu.grid(row = 7, column =0)

        def validate_dir(self,path):
                """ Function that check if the input folders are valid folders"""
                try:
                        invalid=0
                        dirs=os.listdir(path)
                        a=[]
                        for d in dirs:
                            if os.path.isdir(path+os.sep+d)==True:
                                a.append(d)
                                dirs2=os.listdir(path+os.sep+d)
                                for d2 in dirs2:
                                    if os.path.isdir(path+os.sep+d+os.sep+d2)==True:
                                        invalid=1
                                        break
                        if len(a)!=len(dirs):
                            invalid=1
                except:
                        invalid=1
                return invalid

        def redirect(self):
                """ Function that checks if all fields in HomePage are filled.
                    If it is, this function redirects to the parameters page
                    of the selected method, otherwise it redirects back to
                    HomePage """
                self.OK=1
                if self.TrainFolder=='' and self.OK==1:
                        self.error=1
                        self.msgErr1='Training Folder not selected.'
                        self.msgErr2='All fields are mandatory.'
                        self.OK=0
                if self.validate_dir(self.TrainFolder)==1 and self.OK==1:
                        self.error=1
                        self.msgErr1='Invalid Training Folder structure.'
                        self.msgErr2='See User Manual for more info.'
                        self.OK=0
                if self.TestFolder=='' and self.OK==1:
                        self.error=1
                        self.msgErr1='Test Folder file not selected.'
                        self.msgErr2='All fields are mandatory.'
                        self.OK=0
                if self.validate_dir(self.TestFolder)==1 and self.OK==1:
                        self.error=1
                        self.msgErr1='Invalid Test Folder structure.'
                        self.msgErr2='See User Manual for more info.'
                        self.OK=0
                
                # Import selected model from HomePage
                self.model=self.ModelsMenu.get()
                if self.model=='' and self.OK==1:
                        self.error=1
                        self.msgErr1='Pre-trained model not selected'
                        self.msgErr2='All fields are mandatory.'
                        self.OK=0
                # Import selected layer from HomePage
                self.layer=self.LayersMenu.get()
                if self.layer=='' and self.OK==1:
                        self.error=1
                        self.msgErr1='Layer not selected'
                        self.msgErr2='All fields are mandatory.'
                        self.OK=0

                # If all fields are filled the next page is opened
                # otherwise you will be redirected back to HomePage
                if self.OK==1:
                        self.Page1()
                else:
                        self.HomePage()
                        if self.model!='':
                                self.change_dropdown()


        def Page1(self):
                """ Function that creates the page from which is possible
                    to start the features extraction and view its process status.
                    It also has a button that will be enabled (if save activation
                    check button was enabled) that lets to open Page2"""
                # Update page index	
                self.page=1
                
                # Import selections from HomePage to Page1
                self.model=self.ModelsMenu.get()
                self.layer=self.LayersMenu.get()
                self.GPU_ON=bool(self.CUDA.get())
                self.save_activation=bool(self.activations.get())
                
                # Clear window
                self.clear()

                # Back button
                self.back_button = Button(self, text="Back", fg='red',command=self.HomePage)
                self.back_button.grid(row=3, column=0)

                # Progress bar
                self.progress = Progressbar(self, orient="horizontal",length=200, mode="determinate")
                self.progress.grid(row=1, column=1)

                # Visualize Activations
                self.Plotbutton = Button(self,text="Visualize Activations", command=self.Page2)
                self.Plotbutton.grid(row=2, column=2)
                # Disable visualize button
                self.Plotbutton['state'] = 'disabled'
                self.Plotbutton.update()
                
                
                # Start button
                self.Startbutton = Button(self,text="Start Extraction", command=self.start)
                self.Startbutton.grid(row=1, column=0)
                
                # Close program
                self.close_button2 = Button(self, text="Close", fg="red", command=quit)
                self.close_button2.grid(row=3, column=1)



        def Page2(self):
                """ Function that create a page that lets to visualize activations """
                # Clear window
                self.clear()
                # Update current page index
                self.page=2

                self.Z=500
                
                if self.ids==0:
                        prefix='Results'+self.s+'Features'+self.s+self.model+'_'+self.layer
                        # Load training features ids
                        train = h5py.File(prefix+'_'+self.train_suffix+'.h5','r')
                        train_img_ids=train['img_id']

                        
                        # Load test features ids
                        test = h5py.File(prefix+'_'+self.test_suffix+'.h5','r')
                        test_img_ids=test['img_id']
                        
                        # Get list of names
                        train_ids=['Results'+self.s+'Activations'+self.s+self.train_suffix+self.s+self.model+'_'+self.layer+'_'+train_img_ids[i].decode('UTF-8').split(self.s)[1] for i in range(train_img_ids.shape[0])]
                        test_ids=['Results'+self.s+'Activations'+self.s+self.test_suffix+self.s+self.model+'_'+self.layer+'_'+test_img_ids[i].decode('UTF-8').split(self.s)[1] for i in range(test_img_ids.shape[0])]
                        self.ids=train_ids+test_ids
                        test.close()
                        train.close()
                
                self.current_activation+=1
                if self.current_activation==len(self.ids):
                        self.current_activation=0
                # Set name of the image to load
                name=self.ids[self.current_activation]

                try:
                        self.actname_entry.destroy()
                except:
                        pass

                p=name.split(os.sep)[-2:]
                self.actname_label = Label(self, text=p)
                self.actname_label.grid(row = 0, column =1)
                self.actname_label.update()

                # Zoom -
                self.zoom_M = Button(self, text="-", fg='black',command=self.zoomM)
                self.zoom_M.grid(row=0, column=0)
                # Zoom +
                self.zoom_P = Button(self, text="+", fg='black',command=self.zoomP)
                self.zoom_P.grid(row=1, column=0)
                
                # Back button
                self.back_button2 = Button(self, text="Back", fg='red',command=self.HomePage)
                self.back_button2.grid(row=4, column=0)
                
                # Close program
                self.close_button3 = Button(self, text="Close", fg="red", command=quit)
                self.close_button3.grid(row=5, column=0)
                
                                
                # Visualize plot
                self.flag2=0
                self.b=Button2(self, text="Next Image",command=self.Page2)
                self.b.grid(row=1, column=1)

                # Load image
                self.img=PhotoImage(file=name)
                # Backup image with its original size
                self.img2=self.img
                # Decrease image size
                self.img=self.img.subsample(int(np.ceil(self.img.width()/self.Z)),int(np.ceil(self.img.height()/self.Z)))
                # Merge imge with button
                self.b.config(image=self.img, compound=RIGHT)

        def zoomM(self):
                # resore image with decreased size
                self.b.destroy()
                self.b=Button2(self, text="Next Image",command=self.Page2)
                self.b.grid(row=1, column=1)
                self.b.config(image=self.img, compound=RIGHT)

        def zoomP(self):
                # show image with original size
                self.b.destroy()
                self.b=Button2(self, text="Next Image",command=self.Page2)
                self.b.grid(row=1, column=1)
                self.b.config(image=self.img2, compound=RIGHT)

        def start(self):
                """ Start Button Callback. It effectively performs the extraction with the
                    selected pre-trained model and layer """
                # Disable Start button
                self.Startbutton['state'] = 'disabled'
                self.Startbutton.update()
                
                # Create a prefix for output files
                out_prefix=self.model+'_'+self.layer
                out_prefix=mf.make_valid_name(out_prefix)
                
                # Get info about framework used: caffe or keras (tensorflow)
                framework=self.model.split(' ')[0].lower()

                if framework=='caffe':
                        
                        # Get Model paths and settings
                        caffemodel, prototxt, crop_size, mean_v = cfe.Caffe_model_path(self.model)
                        
                        # Start features extraction
                        cfe.Caffe_multiple_features_extraction(self.TrainFolder,caffemodel, prototxt, mean_v, self.layer, crop_size, self.GPU_ON, self.save_activation, out_prefix, self)
                        cfe.Caffe_multiple_features_extraction(self.TestFolder,caffemodel, prototxt, mean_v, self.layer, crop_size, self.GPU_ON, self.save_activation, out_prefix, self)
                
                elif framework=='keras':
                        base_model, preprocessing, crop_size=kfe.Keras_model_path(self.model)
                        kfe.Keras_multiple_features_extraction(self.TrainFolder,base_model, preprocessing, self.layer, crop_size, self.save_activation, out_prefix, self)
                        kfe.Keras_multiple_features_extraction(self.TestFolder,base_model, preprocessing, self.layer, crop_size, self.save_activation, out_prefix, self)
                
                # If save activation is enabled
                if self.save_activation==True:
                        self.Plotbutton['state'] = 'normal'
                        self.Plotbutton.update()
                else:
                        self.Plotbutton['state'] = 'disable'
                        self.Plotbutton.update()

        def clear(self):
                """ Function that lets to clean a window, deleting all the widgets"""
                # Clear window
                wlist = self.grid_slaves()
                for l in wlist:
                        l.destroy()

if __name__ == "__main__":
        app = FeaturesExtractionApp()
        app.mainloop()
