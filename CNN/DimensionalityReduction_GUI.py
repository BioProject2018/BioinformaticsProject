#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <raheleh.salehi@studenti.polito.it>

import warnings
warnings.filterwarnings("ignore")
from tkinter.font import Font
from tkinter import*
from tkinter.ttk import Button as Button2 
from tkinter import Tk, Entry, Button, Label
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Combobox
import time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import src.FeaturesReduction as fr
import os
import webbrowser

class DimensionalityReductionApp(Tk):
        """ It is a GUI to perform dimensionality reduction of a training and a test features matrix
            stored in a .h5 file each.
            Possible dimensionality reduction method are:
                - Principal component analysis (PCA)
                - t-distributed Stochastic Neighbor Embedding (t-SNE)
                - Truncated SVD (TruncatedSVD)
                - Non-Negative Matrix Factorization (NMF)
                - Linear Discriminant Analysis (LDA)
        __________________________________________________________________________________________
        At the end:
                Automatically 2 new .h5 files containing 3 datasets each (one for reduced features, one for labels and one for img_ids)
                are generated in the folder Results/ReducedFeatures and also if n_components is <= 3 a scatter plot is saved in the folder
                Results/Plots
                
        Things to know:
                        Both .h5 files must mandatory follow this structure:
                        - 'feats':   array-like, shape (n_samples, n_features)
                        - 'labels':  array-like, shape (n_samples, )
                        - 'img_ids': array-like, shape (n_samples, )
        """
        
        def __init__(self):
                self.s=os.sep
                # Set title and its text format
                Tk.__init__(self)
                self.title_font = Font(family='Helvetica', size=18, weight="bold", slant="italic")
                self.title("Dimensionality Reduction")

                # Initialize usefull variables
                self.TrainFeatures=''
                self.TestFeatures=''
                self.Method=''
                self.n_comp=2
                self.error=0

                # Visualize HomePage
                self.HomePage()

        def HomePage(self):
                """ Function that creates the HomePage of the GUI """
                #
                self.flag2=1
                
                # Update current page index
                self.page=0
                
                # Clear window
                self.clear()

                # Training Features button and entry
                self.trainfeats_button=Button(self, text="Training Set Features",fg="blue", command=self.get_trainfeats,)
                self.trainfeats_button.grid(row = 0, column =0)
                self.trainfeats_entry = Entry(self, bd =5)
                self.trainfeats_entry.grid(row = 0, column =1)
                self.trainfeats_entry.insert(0,self.TrainFeatures)

                # Test Features button and entry
                self.testfeats_button=Button(self, text="Test Set Features",fg="blue", command=self.get_testfeats)
                self.testfeats_button.grid(row = 1, column =0)
                self.testfeats_entry = Entry(self, bd =5)
                self.testfeats_entry.grid(row = 1, column =1)
                self.testfeats_entry.insert(0,self.TestFeatures)

                # Methods menu
                self.MethodsChoices = [ 'PCA','LDA','t-SNE','TruncatedSVD','NMF']
                self.MethodsLabel=Label(self, text="Choose features reduction method")
                self.MethodsLabel.grid(row = 4, column =0)
                self.MethodsMenu =Combobox(self, textvariable=self.MethodsChoices[0], values=self.MethodsChoices, state= 'readonly')
                self.MethodsMenu.grid(row = 5, column =0)
                self.MethodsMenu.set(self.Method)

                # Next page
                self.NextPage = Button(self, text="Reduce Features", fg="green",command=self.redirect)
                self.NextPage.grid(row=6, column=1)

                # Close program
                self.close_button = Button(self, text="Close", fg="red", command=quit)
                self.close_button.grid(row=7, column=1)

                # Show detected errors
                if self.error==1:
                        self.error=0
                        self.ErrorLabel1=Label(self, text=self.msgErr1, bg="red")
                        self.ErrorLabel1.grid(row = 6, column =0)
                        self.ErrorLabel1=Label(self, text=self.msgErr2, bg="red")
                        self.ErrorLabel1.grid(row = 7, column =0)

        def get_trainfeats(self):
                """ Callback for Training Features button """
                # Select training features .h5 file
                self.TrainFeatures = askopenfilename(initialdir = os.getcwd(),title = "Select Training features file",filetypes = (("HDF5 files","*.h5"),("all files","*.*")))
                tmp=self.TrainFeatures.replace('/', os.sep)
                tmp=tmp.replace('\\', os.sep)
                self.TrainFeatures=tmp
                self.trainfeats_entry.destroy()
                self.trainfeats_entry = Entry(self, bd =5)
                self.trainfeats_entry.grid(row = 0, column =1)
                self.trainfeats_entry.insert(0,self.TrainFeatures)
                self.trainfeats_entry.update()

        def get_testfeats(self):
                """ Callback for Test Features button """
                # Select test features .h5 file
                self.TestFeatures = askopenfilename(initialdir = os.getcwd(),title = "Select Test features file",filetypes = (("HDF5 files","*.h5"),("all files","*.*")))
                tmp=self.TestFeatures.replace('/', os.sep)
                tmp=tmp.replace('\\', os.sep)
                self.TestFeatures=tmp
                self.testfeats_entry.destroy()
                self.testfeats_entry = Entry(self, bd =5)
                self.testfeats_entry.grid(row = 1, column =1)
                self.testfeats_entry.insert(0,self.TestFeatures)
                self.testfeats_entry.update()


        def redirect(self):
                """ Function that checks if all fields in HomePage are filled.
                    If it is, this function redirects to the parameters page
                    of the selected method, otherwise it redirects back to
                    HomePage """
                self.OK=1
                if self.TrainFeatures=='' and self.OK==1:
                        self.error=1
                        self.msgErr1='Training Features file not selected.'
                        self.msgErr2='All fields are mandatory.'
                        self.OK=0
                if self.TestFeatures=='' and self.OK==1:
                        self.error=1
                        self.msgErr1='Test Features file not selected.'
                        self.msgErr2='All fields are mandatory.'
                        self.OK=0
                
                # Import selected method from HomePage
                self.Method=self.MethodsMenu.get()
                if self.Method=='' and self.OK==1:
                        self.error=1
                        self.msgErr1='Method not selected'
                        self.msgErr2='All fields are mandatory.'
                        self.OK=0

                if self.OK==1:
                # If all fields are filled the selected method
                # parameters selection page is opened

                        if self.Method=='PCA':
                                self.PagePCA()
                        elif self.Method=='t-SNE':
                                self.PageTSNE()
                        elif self.Method=='TruncatedSVD':
                                self.PageTruncatedSVD()
                        elif self.Method=='LDA':
                                self.PageLDA()
                        else:
                                self.PageNMF()
                else:
                        self.HomePage()

        def PagePCA(self):
                """ Function that creates the page of the GUI in which it's possible to
                    select parameters and to perform a PCA """
                # Clear window
                self.clear()
                
                # Update current page index
                self.page=1
                
                self.endofreduction=0

                # Number of Components Label. Entry
                self.nCompLabel=Label(self, text='N_components:')
                self.nCompLabel.grid(row = 0, column =0)
                self.nComp_entry = Entry(self, bd =5)
                self.nComp_entry.grid(row = 0, column =1)
                self.nComp_entry.insert(0,'2')

                # Whiten. Menu
                self.whitenChoices = ['False','True']
                self.whitenLabel=Label(self, text="Whiten:")
                self.whitenLabel.grid(row = 2, column =0)
                self.whitenMenu =Combobox(self, textvariable=self.whitenChoices[0], values=self.whitenChoices, state= 'readonly')
                self.whitenMenu.grid(row = 2, column =1)
                self.whitenMenu.set(self.whitenChoices[0])

                # SVD solver. Menu
                self.svd_solverChoices = ['auto', 'full', 'arpack', 'randomized']
                self.svd_solverLabel=Label(self, text="SVD_solver:")
                self.svd_solverLabel.grid(row = 3, column =0)
                self.svd_solverMenu =Combobox(self, textvariable=self.svd_solverChoices[0], values=self.svd_solverChoices, state= 'readonly')
                self.svd_solverMenu.grid(row = 3, column =1)
                self.svd_solverMenu.set(self.svd_solverChoices[0])

                # Documentation link
                self.link = Label(self, text="Click here to see the documentation", fg="blue", cursor="hand2")
                self.link.grid(row = 4, column =1)
                url="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"
                self.link.bind("<Button-1>", lambda e,url=url:self.openlink(url))

                # Back button
                self.back_button = Button(self, text="Back", fg='red',
                                                                                   command=self.HomePage)
                self.back_button.grid(row=7, column=0)
                
                # Close program
                self.close_button2 = Button(self, text="Close", fg="red", command=quit)
                self.close_button2.grid(row=7, column=1)

                # Start button
                self.Startbutton = Button(self,text="Start Reduction", command=self.start)
                self.Startbutton.grid(row=5, column=0)

                # Visualize plot button
                self.Plotbutton = Button(self,text="Visualize Plot", command=self.ScatterPlotPage)
                self.Plotbutton.grid(row=5, column=2)
                # Disable visualize button
                self.Plotbutton['state'] = 'disabled'
                self.Plotbutton.update()

                # Initialize some variables
                t="                                   "
                self.label2=Label(self, text=t)
                self.label2.grid(row = 5, column =1)
                t="       "
                self.label3=Label(self, text=t)
                self.label3.grid(row = 6, column =1)

        def PageLDA(self):
                """ Function that creates the page of the GUI in which it's possible to
                    select parameters and to perform a LDA """
                # Clear window
                self.clear()
                
                # Update current page index
                self.page=1
                
                self.endofreduction=0

                # Number of Components Label. Entry
                self.nCompLabel=Label(self, text='N_components:')
                self.nCompLabel.grid(row = 0, column =0)
                self.nComp_entry = Entry(self, bd =5)
                self.nComp_entry.grid(row = 0, column =1)
                self.nComp_entry.insert(0,'2')

                # Threshold used for rank estimation in SVD solver. Entry
                self.tolLabel=Label(self, text='Tol:')
                self.tolLabel.grid(row = 2, column =0)
                self.tol_entry = Entry(self, bd =5)
                self.tol_entry.grid(row = 2, column =1)
                self.tol_entry.insert(0,'1e-02')

                # Documentation link
                self.link = Label(self, text="Click here to see the documentation", fg="blue", cursor="hand2")
                self.link.grid(row = 4, column =1)
                url="http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html"
                self.link.bind("<Button-1>", lambda e,url=url:self.openlink(url))

                # Back button
                self.back_button = Button(self, text="Back", fg='red',command=self.HomePage)
                self.back_button.grid(row=7, column=0)
                
                # Close program
                self.close_button2 = Button(self, text="Close", fg="red", command=quit)
                self.close_button2.grid(row=7, column=1)

                # Start button
                self.Startbutton = Button(self,text="Start Reduction", command=self.start)
                self.Startbutton.grid(row=5, column=0)

                # Visualize plot button
                self.Plotbutton = Button(self,text="Visualize Plot", command=self.ScatterPlotPage)
                self.Plotbutton.grid(row=5, column=2)
                # Disable visualize button
                self.Plotbutton['state'] = 'disabled'
                self.Plotbutton.update()

                # Initialize some variables
                t="                                   "
                self.label2=Label(self, text=t)
                self.label2.grid(row = 5, column =1)
                t="       "
                self.label3=Label(self, text=t)
                self.label3.grid(row = 6, column =1)


        def PageTSNE(self):
                """ Function that creates the page of the GUI in which it's possible to
                    select parameters and to perform a t-SNE """
                # Clear window
                self.clear()
                
                # Update current page index
                self.page=1
                
                self.endofreduction=0

                # Number of Components Label. Entry
                self.nCompLabel=Label(self, text='N_components:')
                self.nCompLabel.grid(row = 0, column =0)
                self.nComp_entry = Entry(self, bd =5)
                self.nComp_entry.grid(row = 0, column =1)
                self.nComp_entry.insert(0,'2')

                # Perplexity. Entry
                self.perplexityLabel=Label(self, text='Perplexity:')
                self.perplexityLabel.grid(row = 1, column =0)
                self.perplexity_entry = Entry(self, bd =5)
                self.perplexity_entry.grid(row = 1, column =1)
                self.perplexity_entry.insert(0,'30')

                # Early Aggregation. Entry
                self.early_exaggerationLabel=Label(self, text='Early_exaggeration:')
                self.early_exaggerationLabel.grid(row = 2, column =0)
                self.early_exaggeration_entry = Entry(self, bd =5)
                self.early_exaggeration_entry.grid(row = 2, column =1)
                self.early_exaggeration_entry.insert(0,'12')

                # Learning Rate. Entry
                self.learning_rateLabel=Label(self, text='Learning_rate:')
                self.learning_rateLabel.grid(row = 3, column =0)
                self.learning_rate_entry = Entry(self, bd =5)
                self.learning_rate_entry.grid(row = 3, column =1)
                self.learning_rate_entry.insert(0,'200')

                # Maximum number of iterations for the optimization. Entry
                self.n_iterLabel=Label(self, text='N_iter:')
                self.n_iterLabel.grid(row = 4, column =0)
                self.n_iter_entry = Entry(self, bd =5)
                self.n_iter_entry.grid(row = 4, column =1)
                self.n_iter_entry.insert(0,'5000')

                # Maximum number of iterations without progress before we abort the optimization. Entry
                self.n_iter_without_progressLabel=Label(self, text='N_iter_without_progress:')
                self.n_iter_without_progressLabel.grid(row = 5, column =0)
                self.n_iter_without_progress_entry = Entry(self, bd =5)
                self.n_iter_without_progress_entry.grid(row = 5, column =1)
                self.n_iter_without_progress_entry.insert(0,'300')

                # Threshold for stopping the optimization. Entry
                self.min_grad_normLabel=Label(self, text='Min_grad_norm:')
                self.min_grad_normLabel.grid(row = 6, column =0)
                self.min_grad_norm_entry = Entry(self, bd =5)
                self.min_grad_norm_entry.grid(row = 6, column =1)
                self.min_grad_norm_entry.insert(0,'1e-07')
             
                # Initialization of embedding. Menu
                self.initChoices = ['random','pca']
                self.initLabel=Label(self, text="Init:")
                self.initLabel.grid(row = 7, column =0)
                self.initMenu =Combobox(self, textvariable=self.initChoices[0], values=self.initChoices, state= 'readonly')
                self.initMenu.grid(row = 7, column =1)
                self.initMenu.set(self.initChoices[0])

                # Dimensionality reduction method to pre-reduce dimensionality. Menu
                self.reduceChoices = ['PCA','TruncatedSVD','None']
                self.reduceLabel=Label(self, text="Pre-processing:")
                self.reduceLabel.grid(row = 8, column =0)
                self.reduceMenu =Combobox(self, textvariable=self.reduceChoices[0], values=self.reduceChoices, state= 'readonly')
                self.reduceMenu.grid(row = 8, column =1)
                self.reduceMenu.set(self.reduceChoices[0])

                # Documentation link
                self.link = Label(self, text="Click here to see the documentation", fg="blue", cursor="hand2")
                self.link.grid(row = 9, column =1)
                url="http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"
                self.link.bind("<Button-1>", lambda e,url=url:self.openlink(url))

                # Back button
                self.back_button = Button(self, text="Back", fg='red',
                                                                                   command=self.HomePage)
                self.back_button.grid(row=12, column=0)
                
                # Close program
                self.close_button2 = Button(self, text="Close", fg="red", command=quit)
                self.close_button2.grid(row=12, column=1)

                # Start button
                self.Startbutton = Button(self,text="Start Reduction", command=self.start)
                self.Startbutton.grid(row=10, column=0)

                # Visualize plot button
                self.Plotbutton = Button(self,text="Visualize Plot", command=self.ScatterPlotPage)
                self.Plotbutton.grid(row=10, column=2)
                # Disable visualize button
                self.Plotbutton['state'] = 'disabled'
                self.Plotbutton.update()

                # Initialize some variables
                t="                                   "
                self.label2=Label(self, text=t)
                self.label2.grid(row = 10, column =1)
                t="       "
                self.label3=Label(self, text=t)
                self.label3.grid(row = 11, column =1)


        def PageNMF(self):
                """ Function that creates the page of the GUI in which it's possible to
                    select parameters and to perform a NMF """
                # Clear window
                self.clear()
                
                # Update current page index
                self.page=1
                
                self.endofreduction=0

                # Number of Components Label. Entry
                self.nCompLabel=Label(self, text='N_components:')
                self.nCompLabel.grid(row = 0, column =0)
                self.nComp_entry = Entry(self, bd =5)
                self.nComp_entry.grid(row = 0, column =1)
                self.nComp_entry.insert(0,'2')

                # Method used to initialize the procedure. Menu
                self.initChoices = ['nndsvd','random','nndsvda','nndsvdar']
                self.initLabel=Label(self, text="Init:")
                self.initLabel.grid(row = 1, column =0)
                self.initMenu =Combobox(self, textvariable=self.initChoices[0], values=self.initChoices, state= 'readonly')
                self.initMenu.grid(row = 1, column =1)
                self.initMenu.set(self.initChoices[0])

                # Numerical solver to use: 'cd' is a Coordinate Descent solver. 'mu' is a Multiplicative Update solver. Menu
                self.solverChoices = ['cd','mu']
                self.solverLabel=Label(self, text="Solver:")
                self.solverLabel.grid(row = 2, column =0)
                self.solverMenu =Combobox(self, textvariable=self.solverChoices[0], values=self.solverChoices, state= 'readonly')
                self.solverMenu.grid(row = 2, column =1)
                self.solverMenu.set(self.solverChoices[0])

                # Tolerance of the stopping condition. Entry
                self.tolLabel=Label(self, text='Tol:')
                self.tolLabel.grid(row = 3, column =0)
                self.tol_entry = Entry(self, bd =5)
                self.tol_entry.grid(row = 3, column =1)
                self.tol_entry.insert(0,'1e-04')

                # Maximum number of iterations before timing out. Entry
                self.max_iterLabel=Label(self, text='Max_iter:')
                self.max_iterLabel.grid(row = 4, column =0)
                self.max_iter_entry = Entry(self, bd =5)
                self.max_iter_entry.grid(row = 4, column =1)
                self.max_iter_entry.insert(0,'200')

                # Constant that multiplies the regularization terms. Entry
                self.alphaLabel=Label(self, text='Alpha:')
                self.alphaLabel.grid(row = 5, column =0)
                self.alpha_entry = Entry(self, bd =5)
                self.alpha_entry.grid(row = 5, column =1)
                self.alpha_entry.insert(0,'0')

                # The regularization mixing parameter. Entry
                self.l1_ratioLabel=Label(self, text='l1_ratio:')
                self.l1_ratioLabel.grid(row = 6, column =0)
                self.l1_ratio_entry = Entry(self, bd =5)
                self.l1_ratio_entry.grid(row = 6, column =1)
                self.l1_ratio_entry.insert(0,'0')

                # Documentation link
                self.link = Label(self, text="Click here to see the documentation", fg="blue", cursor="hand2")
                self.link.grid(row = 9, column =1)
                url="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html"
                self.link.bind("<Button-1>", lambda e,url=url:self.openlink(url))

                # Back button
                self.back_button = Button(self, text="Back", fg='red',command=self.HomePage)
                self.back_button.grid(row=12, column=0)
                
                # Close program
                self.close_button2 = Button(self, text="Close", fg="red", command=quit)
                self.close_button2.grid(row=12, column=1)

                # Start button
                self.Startbutton = Button(self,text="Start Reduction", command=self.start)
                self.Startbutton.grid(row=10, column=0)

                # Visualize plot button
                self.Plotbutton = Button(self,text="Visualize Plot", command=self.ScatterPlotPage)
                self.Plotbutton.grid(row=10, column=2)
                # Disable visualize button
                self.Plotbutton['state'] = 'disabled'
                self.Plotbutton.update()

                # Initialize some variables
                t="                                   "
                self.label2=Label(self, text=t)
                self.label2.grid(row = 10, column =1)
                t="       "
                self.label3=Label(self, text=t)
                self.label3.grid(row = 11, column =1)


        def PageTruncatedSVD(self):
                """ Function that creates the page of the GUI in which it's possible to
                    select parameters and to perform a TruncatedSVD """
                # Clear window
                self.clear()
                
                # Update current page index
                self.page=1
                
                self.endofreduction=0

                # Number of Components Label. Entry
                self.nCompLabel=Label(self, text='N_components:')
                self.nCompLabel.grid(row = 0, column =0)
                self.nComp_entry = Entry(self, bd =5)
                self.nComp_entry.grid(row = 0, column =1)
                self.nComp_entry.insert(0,'2')

                #SVD solver to use. Menu
                self.algorithmChoices = ['randomized','arpack']
                self.algorithmLabel=Label(self, text="Algorithm:")
                self.algorithmLabel.grid(row = 1, column =0)
                self.algorithmMenu =Combobox(self, textvariable=self.algorithmChoices[0], values=self.algorithmChoices, state= 'readonly')
                self.algorithmMenu.grid(row = 1, column =1)
                self.algorithmMenu.set(self.algorithmChoices[0])

                # Number of iterations for randomized SVD solver. Entry
                self.n_iterLabel=Label(self, text='N_iter:')
                self.n_iterLabel.grid(row = 2, column =0)
                self.n_iter_entry = Entry(self, bd =5)
                self.n_iter_entry.grid(row = 2, column =1)
                self.n_iter_entry.insert(0,'5')

                # Documentation link                
                self.link = Label(self, text="Click here to see the documentation", fg="blue", cursor="hand2")
                self.link.grid(row = 3, column =1)
                url="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html"
                self.link.bind("<Button-1>", lambda e,url=url:self.openlink(url))

                # Back button
                self.back_button = Button(self, text="Back", fg='red',command=self.HomePage)
                self.back_button.grid(row=6, column=0)
                
                # Close program
                self.close_button2 = Button(self, text="Close", fg="red", command=quit)
                self.close_button2.grid(row=6, column=1)

                # Start button
                self.Startbutton = Button(self,text="Start Reduction", command=self.start)
                self.Startbutton.grid(row=4, column=0)

                # Visualize plot button
                self.Plotbutton = Button(self,text="Visualize Plot", command=self.ScatterPlotPage)
                self.Plotbutton.grid(row=4, column=2)
                # Disable visualize button
                self.Plotbutton['state'] = 'disabled'
                self.Plotbutton.update()

                # Initialize some variables
                t="                                   "
                self.label2=Label(self, text=t)
                self.label2.grid(row = 4, column =1)
                t="       "
                self.label3=Label(self, text=t)
                self.label3.grid(row = 5, column =1)

        def openlink(self,url):
                """ Function that lets to open a web browser showing the required documentation"""
                webbrowser.open_new(url)
      
        def ScatterPlotPage(self):
                """ Function that lets to see the generated plots into the GUI """
                # Clear window
                self.clear()
                # Update current page index
                self.page=2
                # Set name of the image to load
                name=self.Method+str(self.n_comp)+'_'+self.TrainFeatures.split(self.s)[-1].split('.')[0]
                name=name.split('_')
                name='_'.join(name[:-1])
                if self.n_comp<3:
                        name_noOut=name+'_noOutliers.png'
                        name=name+'.png'
                else:
                        name=name+'.gif'
                
                
                # Back button
                self.back_button2 = Button(self, text="Back", fg='red',command=self.HomePage)
                self.back_button2.grid(row=1, column=0)
                
                # Close program
                self.close_button3 = Button(self, text="Close", fg="red", command=quit)
                self.close_button3.grid(row=1, column=1)
                
                                
                # Visualize plot
                self.flag2=0
                b=Button2(self)
                b.grid(row=0, column=0)
                if self.n_comp<3:
                        global img
                        img=PhotoImage(file='Results'+self.s+'Plots'+self.s+name)
                        b.config(image=img, compound=RIGHT)
                        b.update()
                        try:
                                img=PhotoImage(file='Results'+self.s+'Plots'+self.s+name_noOut)
                                time.sleep(7)
                                b.config(image=img, compound=RIGHT)
                                b.update()
                        except:
                                pass
                else:
                        frames = [PhotoImage(file='Results'+self.s+'Plots'+self.s+name,format = 'gif -index %i' %(i)) for i in range(36)]
                        i=0
                        while self.flag2==0:
                                img=frames[i]
                                b.config(image=img, compound=RIGHT)
                                b.update()
                                time.sleep(0.5)
                                i+=1
                                if i==36:
                                        i=0
                

        def start(self):
                """ Start Button Callback. It effectively checks if the inserted parameters are valid, restores
                    the default values if they are not and performs the dimensionality reduction with the
                    selected method """
                if self.page==1:
                        
                        # Disable start button
                        self.Startbutton['state'] = 'disabled'
                        self.Startbutton.update()
                        # Disable plot button while reduction is running
                        self.Plotbutton['state'] = 'disable'
                        self.Plotbutton.update()
                        
                        # Update some labels
                        self.label2['text']="This process could take some minutes."
                        self.label3['text']="Wait..."
                        self.label2.update()
                        self.label3.update()                       

                        # Get all the inserted parameters and restore them to the default value
                        # if an invalid input was inserted                   
                        self.params={}
                        if self.Method=='PCA':
                                # Check n_components value
                                self.params['n_components']=self.nComp_entry.get()
                                try:
                                        self.params['n_components']=float(self.params['n_components'])
                                        self.params['n_components']=int(self.params['n_components'])
                                        if self.params['n_components']<1:
                                                self.params['n_components']=2
                                                self.nComp_entry.destroy()
                                                self.nComp_entry = Entry(self, bd =5)
                                                self.nComp_entry.grid(row = 0, column =1)
                                                self.nComp_entry.insert(0,'2 (default)')
                                                self.nComp_entry.update()
                                except:
                                        self.params['n_components']=2
                                        self.nComp_entry.destroy()
                                        self.nComp_entry = Entry(self, bd =5)
                                        self.nComp_entry.grid(row = 0, column =1)
                                        self.nComp_entry.insert(0,'2 (default)')
                                        self.nComp_entry.update()
                                # Check whiten value
                                self.params['whiten']=self.whitenMenu.get()
                                if self.params['whiten']=='True':
                                        self.params['whiten']=True
                                else:
                                        self.params['whiten']=False
                                self.params['svd_solver']=self.svd_solverMenu.get()

                        elif self.Method=='LDA':
                                # Check n_components value
                                self.params['n_components']=self.nComp_entry.get()
                                try:
                                        self.params['n_components']=float(self.params['n_components'])
                                        self.params['n_components']=int(self.params['n_components'])
                                        if self.params['n_components']<1:
                                                self.params['n_components']=2
                                                self.nComp_entry.destroy()
                                                self.nComp_entry = Entry(self, bd =5)
                                                self.nComp_entry.grid(row = 0, column =1)
                                                self.nComp_entry.insert(0,'2 (default)')
                                                self.nComp_entry.update()
                                except:
                                        self.params['n_components']=2
                                        self.nComp_entry.destroy()
                                        self.nComp_entry = Entry(self, bd =5)
                                        self.nComp_entry.grid(row = 0, column =1)
                                        self.nComp_entry.insert(0,'2 (default)')
                                        self.nComp_entry.update()
                                # Check tol value
                                self.params['tol']=self.tol_entry.get()
                                try:
                                        self.params['tol']=float(self.params['tol'])
                                        if self.params['tol']>0.1 or self.params['tol']<=0:
                                                self.params['tol']=1e-2
                                                self.tol_entry.destroy()
                                                self.tol_entry = Entry(self, bd =5)
                                                self.tol_entry.grid(row = 2, column =1)
                                                self.tol_entry.insert(0,'1e-2 (default)')
                                                self.tol_entry.update()
                                except:
                                        self.params['tol']=1e-2
                                        self.tol_entry.destroy()
                                        self.tol_entry = Entry(self, bd =5)
                                        self.tol_entry.grid(row = 2, column =1)
                                        self.tol_entry.insert(0,'1e-2 (default)')
                                        self.tol_entry.update()

                        elif self.Method=='t-SNE':
                                # Check n_components value
                                self.params['n_components']=self.nComp_entry.get()
                                try:
                                        self.params['n_components']=float(self.params['n_components'])
                                        self.params['n_components']=int(self.params['n_components'])
                                        if self.params['n_components']<1:
                                                self.params['n_components']=2
                                                self.nComp_entry.destroy()
                                                self.nComp_entry = Entry(self, bd =5)
                                                self.nComp_entry.grid(row = 0, column =1)
                                                self.nComp_entry.insert(0,'2 (default)')
                                                self.nComp_entry.update()
                                except:
                                        self.params['n_components']=2
                                        self.nComp_entry.destroy()
                                        self.nComp_entry = Entry(self, bd =5)
                                        self.nComp_entry.grid(row = 0, column =1)
                                        self.nComp_entry.insert(0,'2 (default)')
                                        self.nComp_entry.update()
                                # Check perplexity
                                self.params['perplexity']=self.perplexity_entry.get()
                                try:
                                        self.params['perplexity']=float(self.params['perplexity'])
                                        if self.params['perplexity']<=0:
                                                self.params['perplexity']=30
                                                self.perplexity_entry.destroy()
                                                self.perplexity_entry = Entry(self, bd =5)
                                                self.perplexity_entry.grid(row = 1, column =1)
                                                self.perplexity_entry.insert(0,'30 (default)')
                                                self.perplexity_entry.update()
                                except:
                                        self.params['perplexity']=30
                                        self.perplexity_entry.destroy()
                                        self.perplexity_entry = Entry(self, bd =5)
                                        self.perplexity_entry.grid(row = 1, column =1)
                                        self.perplexity_entry.insert(0,'30 (default)')
                                        self.perplexity_entry.update()
                                # Check early exaggeration
                                self.params['early_exaggeration']=self.early_exaggeration_entry.get()
                                try:
                                        self.params['early_exaggeration']=float(self.params['early_exaggeration'])
                                        if self.params['early_exaggeration']<=0:
                                                self.params['early_exaggeration']=12
                                                self.early_exaggeration_entry.destroy()
                                                self.early_exaggeration_entry = Entry(self, bd =5)
                                                self.early_exaggeration_entry.grid(row = 2, column =1)
                                                self.early_exaggeration_entry.insert(0,'12 (default)')
                                                self.early_exaggeration_entry.update()
                                except:
                                        self.params['early_exaggeration']=12
                                        self.early_exaggeration_entry.destroy()
                                        self.early_exaggeration_entry = Entry(self, bd =5)
                                        self.early_exaggeration_entry.grid(row = 2, column =1)
                                        self.early_exaggeration_entry.insert(0,'12 (default)')
                                        self.early_exaggeration_entry.update()
                                # Check learning rate
                                self.params['learning_rate']=self.learning_rate_entry.get()
                                try:
                                        self.params['learning_rate']=float(self.params['learning_rate'])
                                        if self.params['learning_rate']<=0:
                                                self.params['learning_rate']=200
                                                self.learning_rate_entry.destroy()
                                                self.learning_rate_entry = Entry(self, bd =5)
                                                self.learning_rate_entry.grid(row = 3, column =1)
                                                self.learning_rate_entry.insert(0,'200 (default)')
                                                self.learning_rate_entry.update()
                                except:
                                        self.params['learning_rate']=200
                                        self.learning_rate_entry.destroy()
                                        self.learning_rate_entry = Entry(self, bd =5)
                                        self.learning_rate_entry.grid(row = 3, column =1)
                                        self.learning_rate_entry.insert(0,'200 (default)')
                                        self.learning_rate_entry.update()
                                # Check n_iter
                                self.params['n_iter']=self.n_iter_entry.get()
                                try:
                                        self.params['n_iter']=float(self.params['n_iter'])
                                        self.params['n_iter']=int(self.params['n_iter'])
                                        if self.params['n_iter']<250:
                                                self.params['n_iter']=5000
                                                self.n_iter_entry.destroy()
                                                self.n_iter_entry = Entry(self, bd =5)
                                                self.n_iter_entry.grid(row = 4, column =1)
                                                self.n_iter_entry.insert(0,'5000 (default)')
                                                self.n_iter_entry.update()
                                except:
                                        self.params['n_iter']=5000
                                        self.n_iter_entry.destroy()
                                        self.n_iter_entry = Entry(self, bd =5)
                                        self.n_iter_entry.grid(row = 4, column =1)
                                        self.n_iter_entry.insert(0,'5000 (default)')
                                        self.n_iter_entry.update()
                                # Check n_iter without progress
                                self.params['n_iter_without_progress']=self.n_iter_without_progress_entry.get()
                                try:
                                        self.params['n_iter_without_progress']=float(self.params['n_iter_without_progress'])
                                        self.params['n_iter_without_progress']=int(self.params['n_iter_without_progress'])
                                        if self.params['n_iter_without_progress']<0:
                                                self.params['n_iter_without_progress']=300
                                                self.n_iter_without_progress_entry.destroy()
                                                self.n_iter_without_progress_entry = Entry(self, bd =5)
                                                self.n_iter_without_progress_entry.grid(row = 5, column =1)
                                                self.n_iter_without_progress_entry.insert(0,'300 (default)')
                                                self.n_iter_without_progress_entry.update()
                                except:
                                        self.params['n_iter_without_progress']=300
                                        self.n_iter_without_progress_entry.destroy()
                                        self.n_iter_without_progress_entry = Entry(self, bd =5)
                                        self.n_iter_without_progress_entry.grid(row = 5, column =1)
                                        self.n_iter_without_progress_entry.insert(0,'300 (default)')
                                        self.n_iter_without_progress_entry.update()
                                # Check min_grad_norm
                                self.params['min_grad_norm']=self.min_grad_norm_entry.get()
                                try:
                                        self.params['min_grad_norm']=float(self.params['min_grad_norm'])
                                        if self.params['min_grad_norm']>0.1 or self.params['min_grad_norm']<=0:
                                                self.params['min_grad_norm']=1e-7
                                                self.min_grad_norm_entry.destroy()
                                                self.min_grad_norm_entry = Entry(self, bd =5)
                                                self.min_grad_norm_entry.grid(row = 6, column =1)
                                                self.min_grad_norm_entry.insert(0,'1e-7 (default)')
                                                self.min_grad_norm_entry.update()
                                except:
                                        self.params['min_grad_norm']=1e-7
                                        self.min_grad_norm_entry.destroy()
                                        self.min_grad_norm_entry = Entry(self, bd =5)
                                        self.min_grad_norm_entry.grid(row = 6, column =1)
                                        self.min_grad_norm_entry.insert(0,'1e-7 (default)')
                                        self.min_grad_norm_entry.update()
                                self.params['reduce']=self.reduceMenu.get()
                                self.params['init']=self.initMenu.get()
                        elif self.Method=='TruncatedSVD':
                                # Check n_components
                                self.params['n_components']=self.nComp_entry.get()
                                try:
                                        self.params['n_components']=float(self.params['n_components'])
                                        self.params['n_components']=int(self.params['n_components'])
                                        if self.params['n_components']<1:
                                                self.params['n_components']=2
                                                self.nComp_entry.destroy()
                                                self.nComp_entry = Entry(self, bd =5)
                                                self.nComp_entry.grid(row = 0, column =1)
                                                self.nComp_entry.insert(0,'2 (default)')
                                                self.nComp_entry.update()
                                except:
                                        self.params['n_components']=2
                                        self.nComp_entry.destroy()
                                        self.nComp_entry = Entry(self, bd =5)
                                        self.nComp_entry.grid(row = 0, column =1)
                                        self.nComp_entry.insert(0,'2 (default)')
                                        self.nComp_entry.update()
                                self.params['algorithm']=self.algorithmMenu.get()
                                # Check n_iter
                                self.params['n_iter']=self.n_iter_entry.get()
                                try:
                                        self.params['n_iter']=float(self.params['n_iter'])
                                        self.params['n_iter']=int(self.params['n_iter'])
                                        if self.params['n_iter']<1:
                                                self.params['n_iter']=5
                                                self.n_iter_entry.destroy()
                                                self.n_iter_entry = Entry(self, bd =5)
                                                self.n_iter_entry.grid(row = 2, column =1)
                                                self.n_iter_entry.insert(0,'5 (default)')
                                                self.n_iter_entry.update()
                                except:
                                        self.params['n_iter']=5
                                        self.n_iter_entry.destroy()
                                        self.n_iter_entry = Entry(self, bd =5)
                                        self.n_iter_entry.grid(row = 2, column =1)
                                        self.n_iter_entry.insert(0,'5 (default)')
                                        self.n_iter_entry.update()
                        else:
                                # Check n_components
                                self.params['n_components']=self.nComp_entry.get()
                                try:
                                        self.params['n_components']=float(self.params['n_components'])
                                        self.params['n_components']=int(self.params['n_components'])
                                        if self.params['n_components']<1:
                                                self.params['n_components']=2
                                                self.nComp_entry.destroy()
                                                self.nComp_entry = Entry(self, bd =5)
                                                self.nComp_entry.grid(row = 0, column =1)
                                                self.nComp_entry.insert(0,'2 (default)')
                                                self.nComp_entry.update()
                                except:
                                        self.params['n_components']=2
                                        self.nComp_entry.destroy()
                                        self.nComp_entry = Entry(self, bd =5)
                                        self.nComp_entry.grid(row = 0, column =1)
                                        self.nComp_entry.insert(0,'2 (default)')
                                        self.nComp_entry.update()        
                                self.params['init']=self.initMenu.get()
                                self.params['solver']=self.solverMenu.get()
                                # Check tol value
                                self.params['tol']=self.tol_entry.get()
                                try:
                                        self.params['tol']=float(self.params['tol'])
                                        if self.params['tol']>0.1 or self.params['tol']<=0:
                                                self.params['tol']=1e-4
                                                self.tol_entry.destroy()
                                                self.tol_entry = Entry(self, bd =5)
                                                self.tol_entry.grid(row = 3, column =1)
                                                self.tol_entry.insert(0,'1e-4 (default)')
                                                self.tol_entry.update()
                                except:
                                        self.params['tol']=1e-4
                                        self.tol_entry.destroy()
                                        self.tol_entry = Entry(self, bd =5)
                                        self.tol_entry.grid(row = 3, column =1)
                                        self.tol_entry.insert(0,'1e-4 (default)')
                                        self.tol_entry.update()
                                # Check max_iter value
                                self.params['max_iter']=self.max_iter_entry.get()
                                try:
                                        self.params['max_iter']=float(self.params['max_iter'])
                                        self.params['max_iter']=int(self.params['max_iter'])
                                        if self.params['max_iter']<=0:
                                                self.params['max_iter']=200
                                                self.max_iter_entry.destroy()
                                                self.max_iter_entry = Entry(self, bd =5)
                                                self.max_iter_entry.grid(row = 4, column =1)
                                                self.max_iter_entry.insert(0,'200 (default)')
                                                self.max_iter_entry.update()
                                except:
                                        self.params['max_iter']=200
                                        self.max_iter_entry.destroy()
                                        self.max_iter_entry = Entry(self, bd =5)
                                        self.max_iter_entry.grid(row = 4, column =1)
                                        self.max_iter_entry.insert(0,'200 (default)')
                                        self.max_iter_entry.update()
                                # Check alpha value
                                self.params['alpha']=self.alpha_entry.get()
                                try:
                                        self.params['alpha']=float(self.params['alpha'])
                                        if self.params['alpha']<0:
                                                self.params['alpha']=0
                                                self.alpha_entry.destroy()
                                                self.alpha_entry = Entry(self, bd =5)
                                                self.alpha_entry.grid(row = 5, column =1)
                                                self.alpha_entry.insert(0,'0 (default)')
                                                self.alpha_entry.update()
                                except:
                                        self.params['alpha']=0
                                        self.alpha_entry.destroy()
                                        self.alpha_entry = Entry(self, bd =5)
                                        self.alpha_entry.grid(row = 5, column =1)
                                        self.alpha_entry.insert(0,'0 (default)')
                                        self.alpha_entry.update()
                                # Check l1_ratio value
                                self.params['l1_ratio']=self.l1_ratio_entry.get()
                                try:
                                        self.params['l1_ratio']=float(self.params['l1_ratio'])
                                        if self.params['l1_ratio']>1 or self.params['l1_ratio']<0:
                                                self.params['l1_ratio']=0
                                                self.l1_ratio_entry.destroy()
                                                self.l1_ratio_entry = Entry(self, bd =5)
                                                self.l1_ratio_entry.grid(row = 6, column =1)
                                                self.l1_ratio_entry.insert(0,'0 (default)')
                                                self.l1_ratio_entry.update()
                                except:
                                        self.params['l1_ratio']=0
                                        self.l1_ratio_entry.destroy()
                                        self.l1_ratio_entry = Entry(self, bd =5)
                                        self.l1_ratio_entry.grid(row = 6, column =1)
                                        self.l1_ratio_entry.insert(0,'0 (default)')
                                        self.l1_ratio_entry.update()
                        
                        print(self.params)
                        self.n_comp=self.params['n_components']
                        
                        # Start Features Reduction
                        try:
                                fr.dimensionality_reduction(self.TrainFeatures, self.TestFeatures, self.Method, self.params)
                                # Enable start button
                                self.Startbutton['state'] = 'normal'
                                self.Startbutton.update()
                                
                                # Update some labels
                                self.label2['text']="         Process completed!          " 
                                self.label3['text']="        "
                                self.label2.update()
                                self.label3.update()
                                
                                # If number of components is minor than 4 enable plot button
                                if self.n_comp<4:
                                        self.Plotbutton['state'] = 'normal'
                                        self.Plotbutton.update()
                                else:
                                        self.Plotbutton['state'] = 'disable'
                                        self.Plotbutton.update()
                        except:
                       
                                self.error=1
                                self.msgErr1='At least one of the input files is invalid.'
                                self.msgErr2='Read on User Manual how inputs files must be.'
                                self.HomePage()
                        
                else:
                        pass
                        
        def clear(self):
                """ Function that lets to clean a window, deleting all the widgets"""
                # Clear window
                wlist = self.grid_slaves()
                for l in wlist:
                        l.destroy()


if __name__ == "__main__":
        app = DimensionalityReductionApp()
        app.mainloop()
