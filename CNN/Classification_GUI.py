#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <raheleh.salehi@studenti.polito.it>

import warnings
warnings.filterwarnings("ignore")
from tkinter.font import Font
from tkinter import*
from tkinter.ttk import Button as Button2 
from tkinter import Tk, Entry, Button, Label, IntVar, Checkbutton
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Combobox
import os
import numpy as np
import src.Classification as cl
import webbrowser


class ClassificationApp(Tk):
		""" It is a GUI to the classification of a test set using 2 features matrixes
			(one for the training set and one for the test set) stored in a .h5 file each.
			Possible classification methods are:
						- Support vector machine (SVM) 
						- RandomForest (RandomForest)
						- Naive Bayes classifier (NaiveBayes)
						- K-nearest neighbors (k-NN)
		__________________________________________________________________________________________
		At the end:
				- An .xlsx file containing results of the classification is created in Classification/Predictions folder
				- if confusion matrix check box is set on 1 an image of the computed confusion matrix is saved in the
				  folder Classification/ConfusionMatrix

		Things to know:
						Both .h5 files must mandatory follow this structure:
						- 'pca','tsne','tsvd','lda','nmf' or 'feats':   array-like, shape (n_samples, n_features)
						- 'labels':  array-like, shape (n_samples, )
						- 'img_ids': array-like, shape (n_samples, )
		"""
		
		def __init__(self):
				self.s=os.sep
				# Set the title of the App and its font
				Tk.__init__(self)
				self.title_font = Font(family='Helvetica', size=18, weight="bold", slant="italic")
				self.title("Classification - HomePage")

				# Initialize some usefull variables
				self.TrainFeatures=''
				self.TestFeatures=''
				self.Classifier=''
				self.error=0

				# Visualize HomePage
				self.HomePage()

		def HomePage(self):
				self.title("Classification - HomePage")
				
				# Update page index
				self.page=0
				
				# Clear window
				self.clear()

				# Training Features button
				self.trainfeats_button=Button(self, text="Training Set Features",fg="blue", command=self.get_trainfeats)
				self.trainfeats_button.grid(row = 0, column =0)
				self.trainfeats_entry = Entry(self, bd =5)
				self.trainfeats_entry.grid(row = 0, column =1)
				self.trainfeats_entry.insert(0,self.TrainFeatures)

				# Test Features button
				self.testfeats_button=Button(self, text="Test Set Features",fg="blue", command=self.get_testfeats)
				self.testfeats_button.grid(row = 1, column =0)
				self.testfeats_entry = Entry(self, bd =5)
				self.testfeats_entry.grid(row = 1, column =1)
				self.testfeats_entry.insert(0,self.TestFeatures)

				# Methods menu
				self.choices = [ 'SVM','RandomForest','NaiveBayes','k-NN']
				self.label=Label(self, text="Choose classification method")
				self.label.grid(row = 4, column =0)
				self.Menu =Combobox(self, textvariable=self.choices[0], values=self.choices, state= 'readonly')
				self.Menu.grid(row = 5, column =0)
				self.Menu.set(self.Classifier)
				
				# Confusion matrix flag
				self.conf_matrix=IntVar()
				self.cm_flag=Checkbutton(self, text="Compute Confusion Matrix", variable=self.conf_matrix)
				self.cm_flag.grid(row=3, column=1)

				# Next page
				self.next_page = Button(self, text="Classify", fg="green",command=self.redirect)
				self.next_page.grid(row=6, column=1)

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
				# Select reduced training features file
				self.TrainFeatures = askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("HDF5 files","*.h5"),("all files","*.*")))
				self.trainfeats_entry.insert(0,self.TrainFeatures)
				tmp=self.TrainFeatures.replace('/', os.sep)
				tmp=tmp.replace('\\', os.sep)
				self.TrainFeatures=tmp
				self.trainfeats_entry.destroy()
				self.trainfeats_entry = Entry(self, bd =5)
				self.trainfeats_entry.grid(row = 0, column =1)
				self.trainfeats_entry.insert(0,self.TrainFeatures)
				self.trainfeats_entry.update()

		def get_testfeats(self):
				# Select reduced test features file
				self.TestFeatures = askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("HDF5 files","*.h5"),("all files","*.*")))
				self.testfeats_entry.insert(0,self.TestFeatures)
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
				self.Classifier=self.Menu.get()
				if self.Classifier=='' and self.OK==1:
						self.error=1
						self.msgErr1='Method not selected'
						self.msgErr2='All fields are mandatory.'
						self.OK=0

				if self.OK==1:
				# If all fields are filled the selected method
				# parameters selection page is opened

						if self.Classifier=='SVM':
								self.PageSVM()
						elif self.Classifier=='RandomForest':
								self.PageRandomForest()
						elif self.Classifier=='NaiveBayes':
								self.PageNaiveBayes()
						else:
								self.PagekNN()
				else:
						self.HomePage()




		def PageSVM(self):
				self.title("Classification - SVM")
				
				# Get selected classifier from HomePage and CM config
				self.Classifier=self.Menu.get()
				self.cm=int(self.conf_matrix.get())
				
				# Clear window
				self.clear()
				
				# Update page index
				self.page=1
								
				self.endofclassification=0


				# Penalty parameter C of the error term. Entry
				self.CLabel=Label(self, text='C:')
				self.CLabel.grid(row = 0, column =0)
				self.C_entry = Entry(self, bd =5)
				self.C_entry.grid(row = 0, column =1)
				self.C_entry.insert(0,'0.001, 0.01, 0.1, 1, 10, 100, 1000')


				# Penalty parameter Gamma of the error term. Entry
				self.GammaLabel=Label(self, text='Gamma:')
				self.GammaLabel.grid(row = 1, column =0)
				self.Gamma_entry = Entry(self, bd =5)
				self.Gamma_entry.grid(row = 1, column =1)
				self.Gamma_entry.insert(0,'0.001, 0.01, 0.1, 1')
				
				# Kernel. Menu
				self.kernelChoices = ['rbf','linear', 'poly', 'sigmoid']
				self.kernelLabel=Label(self, text="SVD_solver:")
				self.kernelLabel.grid(row = 2, column =0)
				self.kernelMenu =Combobox(self, textvariable=self.kernelChoices[0], values=self.kernelChoices, state= 'readonly')
				self.kernelMenu.grid(row = 2, column =1)
				self.kernelMenu.set(self.kernelChoices[0])

				# Degree. Entry
				self.degreeLabel=Label(self, text='Degree:')
				self.degreeLabel.grid(row = 3, column =0)
				self.degree_entry = Entry(self, bd =5)
				self.degree_entry.grid(row = 3, column =1)
				self.degree_entry.insert(0,'3')

				# Coef0. Entry
				self.coef0Label=Label(self, text='Coef0:')
				self.coef0Label.grid(row = 4, column =0)
				self.coef0_entry = Entry(self, bd =5)
				self.coef0_entry.grid(row = 4, column =1)
				self.coef0_entry.insert(0,'0.0')

				# probability. Menu
				self.probabilityChoices = ['False','True']
				self.probabilityLabel=Label(self, text="Probability:")
				self.probabilityLabel.grid(row = 5, column =0)
				self.probabilityMenu =Combobox(self, textvariable=self.probabilityChoices[0], values=self.probabilityChoices, state= 'readonly')
				self.probabilityMenu.grid(row = 5, column =1)
				self.probabilityMenu.set(self.probabilityChoices[0])

				# Shrinking. Menu
				self.shrinkingChoices = ['True','False']
				self.shrinkingLabel=Label(self, text="Shrinking:")
				self.shrinkingLabel.grid(row = 6, column =0)
				self.shrinkingMenu =Combobox(self, textvariable=self.shrinkingChoices[0], values=self.shrinkingChoices, state= 'readonly')
				self.shrinkingMenu.grid(row = 6, column =1)
				self.shrinkingMenu.set(self.shrinkingChoices[0])

				# Tolerance for stopping criterion. Entry
				self.tolLabel=Label(self, text='Tol:')
				self.tolLabel.grid(row = 7, column =0)
				self.tol_entry = Entry(self, bd =5)
				self.tol_entry.grid(row = 7, column =1)
				self.tol_entry.insert(0,'1e-03')

				# Decision function shape. Menu
				self.decision_function_shapeChoices = ['ovr','ovo']
				self.decision_function_shapeLabel=Label(self, text="Decision_function_shape:")
				self.decision_function_shapeLabel.grid(row = 8, column =0)
				self.decision_function_shapeMenu =Combobox(self, textvariable=self.decision_function_shapeChoices[0], values=self.decision_function_shapeChoices, state= 'readonly')
				self.decision_function_shapeMenu.grid(row = 8, column =1)
				self.decision_function_shapeMenu.set(self.decision_function_shapeChoices[0])

				# Documentation link                
				self.link = Label(self, text="Click here to see the documentation", fg="blue", cursor="hand2")
				self.link.grid(row = 9, column =1)
				url="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"
				self.link.bind("<Button-1>", lambda e,url=url:self.openlink(url))

				# Back button
				self.back_button = Button(self, text="Back", fg='red',command=self.HomePage)
				self.back_button.grid(row=12, column=0)

				# Start button
				self.Startbutton = Button(self,text="Start Classification", command=self.start)
				self.Startbutton.grid(row=10, column=0)
				
				# Visualize Confusion Matrix
				self.Plotbutton = Button(self,text="Visualize Confusion Matrix", command=self.Page2)
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

		def PageRandomForest(self):
				self.title("Classification - Random Forest")
				
				# Get selected classifier from HomePage and CM config
				self.Classifier=self.Menu.get()
				self.cm=int(self.conf_matrix.get())
				
				# Clear window
				self.clear()
				
				# Update page index
				self.page=1
								
				self.endofclassification=0

				# The number of trees in the forest. Entry
				self.n_estimatorsLabel=Label(self, text='N_estimators:')
				self.n_estimatorsLabel.grid(row = 0, column =0)
				self.n_estimators_entry = Entry(self, bd =5)
				self.n_estimators_entry.grid(row = 0, column =1)
				self.n_estimators_entry.insert(0,'10, 20, 30')

				# Criterion. Menu
				self.criterionChoices = ['gini','entropy']
				self.criterionLabel=Label(self, text="Criterion:")
				self.criterionLabel.grid(row = 1, column =0)
				self.criterionMenu =Combobox(self, textvariable=self.criterionChoices[0], values=self.criterionChoices, state= 'readonly')
				self.criterionMenu.grid(row = 1, column =1)
				self.criterionMenu.set(self.criterionChoices[0])

				# max_depth. Entry
				self.max_depthLabel=Label(self, text='Max_depth:')
				self.max_depthLabel.grid(row = 2, column =0)
				self.max_depth_entry = Entry(self, bd =5)
				self.max_depth_entry.grid(row = 2, column =1)
				self.max_depth_entry.insert(0,'None')

				# Documentation link                
				self.link = Label(self, text="Click here to see the documentation", fg="blue", cursor="hand2")
				self.link.grid(row = 3, column =1)
				url="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
				self.link.bind("<Button-1>", lambda e,url=url:self.openlink(url))

				# Back button
				self.back_button = Button(self, text="Back", fg='red',command=self.HomePage)
				self.back_button.grid(row=6, column=0)

				# Start button
				self.Startbutton = Button(self,text="Start Classification", command=self.start)
				self.Startbutton.grid(row=4, column=0)
				
				# Visualize Confusion Matrix
				self.Plotbutton = Button(self,text="Visualize Confusion Matrix", command=self.Page2)
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

		def PageNaiveBayes(self):
				self.title("Classification - Naive Bayes")
				
				# Get selected classifier from HomePage and CM config
				self.Classifier=self.Menu.get()
				self.cm=int(self.conf_matrix.get())
				
				# Clear window
				self.clear()
				
				# Update page index
				self.page=1
								
				self.endofclassification=0

				# Back button
				self.back_button = Button(self, text="Back", fg='red',command=self.HomePage)
				self.back_button.grid(row=4, column=0)

				# Start button
				self.Startbutton = Button(self,text="Start Classification", command=self.start)
				self.Startbutton.grid(row=2, column=0)
				
				# Visualize Confusion Matrix
				self.Plotbutton = Button(self,text="Visualize Confusion Matrix", command=self.Page2)
				self.Plotbutton.grid(row=2, column=2)
				# Disable visualize button
				self.Plotbutton['state'] = 'disabled'
				self.Plotbutton.update()

				# Initialize some variables
				t="                                   "
				self.label2=Label(self, text=t)
				self.label2.grid(row = 2, column =1)
				t="       "
				self.label3=Label(self, text=t)
				self.label3.grid(row = 3, column =1)

		def PagekNN(self):
				self.title("Classification - k-NN")
				
				# Get selected classifier from HomePage and CM config
				self.Classifier=self.Menu.get()
				self.cm=int(self.conf_matrix.get())
				
				# Clear window
				self.clear()
				
				# Update page index
				self.page=1
								
				self.endofclassification=0


				# Number of neighbors to use. Entry
				self.n_neighborsLabel=Label(self, text='N_neighbors:')
				self.n_neighborsLabel.grid(row = 0, column =0)
				self.n_neighbors_entry = Entry(self, bd =5)
				self.n_neighbors_entry.grid(row = 0, column =1)
				self.n_neighbors_entry.insert(0,'1, 2, 3, 4, 5, 6, 7, 8, 9')

				# Weights. Menu
				self.weightsChoices = ['uniform','distance']
				self.weightsLabel=Label(self, text="Weights:")
				self.weightsLabel.grid(row = 1, column =0)
				self.weightsMenu =Combobox(self, textvariable=self.weightsChoices[0], values=self.weightsChoices, state= 'readonly')
				self.weightsMenu.grid(row = 1, column =1)
				self.weightsMenu.set(self.weightsChoices[0])

				# Algorithm. Menu
				self.algorithmChoices = ['auto', 'ball_tree', 'kd_tree', 'brute']
				self.algorithmLabel=Label(self, text="Algorithm:")
				self.algorithmLabel.grid(row = 2, column =0)
				self.algorithmMenu =Combobox(self, textvariable=self.algorithmChoices[0], values=self.algorithmChoices, state= 'readonly')
				self.algorithmMenu.grid(row = 2, column =1)
				self.algorithmMenu.set(self.algorithmChoices[0])

				# Leaf size passed to BallTree or KDTree. Entry
				self.leaf_sizeLabel=Label(self, text='Leaf_size:')
				self.leaf_sizeLabel.grid(row = 3, column =0)
				self.leaf_size_entry = Entry(self, bd =5)
				self.leaf_size_entry.grid(row = 3, column =1)
				self.leaf_size_entry.insert(0,'30')

				# Power parameter for the Minkowski metric. Entry
				self.pLabel=Label(self, text='P:')
				self.pLabel.grid(row = 4, column =0)
				self.p_entry = Entry(self, bd =5)
				self.p_entry.grid(row = 4, column =1)
				self.p_entry.insert(0,'2')

				# Documentation link                
				self.link = Label(self, text="Click here to see the documentation", fg="blue", cursor="hand2")
				self.link.grid(row = 5, column =1)
				url="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier"
				self.link.bind("<Button-1>", lambda e,url=url:self.openlink(url))
				
				# Back button
				self.back_button = Button(self, text="Back", fg='red',command=self.HomePage)
				self.back_button.grid(row=8, column=0)

				# Start button
				self.Startbutton = Button(self,text="Start Classification", command=self.start)
				self.Startbutton.grid(row=6, column=0)
				
				# Visualize Confusion Matrix
				self.Plotbutton = Button(self,text="Visualize Confusion Matrix", command=self.Page2)
				self.Plotbutton.grid(row=6, column=2)
				# Disable visualize button
				self.Plotbutton['state'] = 'disabled'
				self.Plotbutton.update()

				# Initialize some variables
				t="                                   "
				self.label2=Label(self, text=t)
				self.label2.grid(row = 6, column =1)
				t="       "
				self.label3=Label(self, text=t)
				self.label3.grid(row = 7, column =1)

		def openlink(self,url):
				""" Function that lets to open a web browser showing the required documentation"""
				webbrowser.open_new(url)
				
		def Page2(self):
				self.title("Classification - Confusion Matrix")

				# Clear window
				self.clear()
				# Update current page index
				self.page=2
				# Set name of the image to load
				name=self.Classifier+'_'+self.TestFeatures.split(self.s)[-1].split('.')[0]+'.png'
		
				
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

				global img
				img=PhotoImage(file='Results'+self.s+'Classification'+self.s+'ConfusionMatrix'+self.s+name)
				img.subsample(2,2)
				b.config(image=img, compound=RIGHT)



		   

		def start(self):
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
						if self.Classifier=='SVM':
								self.params['C']=self.C_entry.get()
								self.params['C']=self.params['C'].split(',')
								try:
										self.params['C']=[float(i) for i in self.params['C']]
										v=[0 if i>0 else 1 for i in self.params['C']]
										if sum(v)>0:
												self.params['C']=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
												self.C_entry.destroy()
												self.C_entry = Entry(self, bd =5)
												self.C_entry.grid(row = 0, column =1)
												self.C_entry.insert(0,'0.001, 0.01, 0.1, 1, 10, 100, 1000 (default)')
												self.C_entry.update()
								except:
										self.params['C']=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
										self.C_entry.destroy()
										self.C_entry = Entry(self, bd =5)
										self.C_entry.grid(row = 0, column =1)
										self.C_entry.insert(0,'0.001, 0.01, 0.1, 1, 10, 100, 1000 (default)')
										self.C_entry.update()

								self.params['gamma']=self.Gamma_entry.get()
								self.params['gamma']=self.params['gamma'].split(',')
								try:
										self.params['gamma']=[float(i) for i in self.params['gamma']]
										v=[0 if i>0 else 1 for i in self.params['gamma']]
										if sum(v)>0:
												self.params['gamma']=[0.001, 0.01, 0.1, 1]
												self.Gamma_entry.destroy()
												self.Gamma_entry = Entry(self, bd =5)
												self.Gamma_entry.grid(row = 1, column =1)
												self.Gamma_entry.insert(0,'0.001, 0.01, 0.1, 1 (default)')
												self.Gamma_entry.update()
								except:
										self.params['gamma']=[0.001, 0.01, 0.1, 1]
										self.Gamma_entry.destroy()
										self.Gamma_entry = Entry(self, bd =5)
										self.Gamma_entry.grid(row = 1, column =1)
										self.Gamma_entry.insert(0,'0.001, 0.01, 0.1, 1 (default)')
										self.Gamma_entry.update()
								self.params['kernel']=self.kernelMenu.get()      
								self.params['degree']=self.degree_entry.get()
								try:
										self.params['degree']=float(self.params['degree'])
										self.params['degree']=int(self.params['degree'])
										if self.params['degree']<=0:
												self.params['degree']=3
												self.degree_entry.destroy()
												self.degree_entry = Entry(self, bd =5)
												self.degree_entry.grid(row = 3, column =1)
												self.degree_entry.insert(0,'3 (default)')
												self.degree_entry.update()
								except:
										self.params['degree']=3
										self.degree_entry.destroy()
										self.degree_entry = Entry(self, bd =5)
										self.degree_entry.grid(row = 3, column =1)
										self.degree_entry.insert(0,'3 (default)')
										self.degree_entry.update()
								self.params['coef0']=self.coef0_entry.get()
								try:
										self.params['coef0']=float(self.params['coef0'])
										if self.params['coef0']<0:
												self.params['coef0']=0
												self.coef0_entry.destroy()
												self.coef0_entry = Entry(self, bd =5)
												self.coef0_entry.grid(row = 4, column =1)
												self.coef0_entry.insert(0,'0.0 (default)')
												self.coef0_entry.update()
								except:
										self.params['coef0']=0
										self.coef0_entry.destroy()
										self.coef0_entry = Entry(self, bd =5)
										self.coef0_entry.grid(row = 4, column =1)
										self.coef0_entry.insert(0,'0.0 (default)')
										self.coef0_entry.update()

								self.params['probability']=self.probabilityMenu.get()
								if self.params['probability']=='True':
										self.params['probability']=True
								else:
										self.params['probability']=False

								self.params['shrinking']=self.shrinkingMenu.get()
								if self.params['shrinking']=='True':
										self.params['shrinking']=True
								else:
										self.params['shrinking']=False

								self.params['tol']=self.tol_entry.get()
								try:
										self.params['tol']=float(self.params['tol'])
										if self.params['tol']>0.1 or self.params['tol']<=0:
												self.params['tol']=1e-3
												self.tol_entry.destroy()
												self.tol_entry = Entry(self, bd =5)
												self.tol_entry.grid(row = 7, column =1)
												self.tol_entry.insert(0,'1e-3 (default)')
												self.tol_entry.update()
								except:
										self.params['tol']=1e-3
										self.tol_entry.destroy()
										self.tol_entry = Entry(self, bd =5)
										self.tol_entry.grid(row = 7, column =1)
										self.tol_entry.insert(0,'1e-3 (default)')
										self.tol_entry.update()
								self.params['decision_function_shape']=self.decision_function_shapeMenu.get()
								
															
						elif self.Classifier=='RandomForest':
								self.params['n_estimators']=self.n_estimators_entry.get()
								self.params['n_estimators']=self.params['n_estimators'].split(',')
								try:
										self.params['n_estimators']=[float(i) for i in self.params['n_estimators']]
										self.params['n_estimators']=[int(i) for i in self.params['n_estimators']]
										v=[0 if i>0 else 1 for i in  self.params['n_estimators']]
										if sum(v)>0:
												self.params['n_estimators']=[10, 20, 30]
												self.n_estimators_entry.destroy()
												self.n_estimators_entry = Entry(self, bd =5)
												self.n_estimators_entry.grid(row = 0, column =1)
												self.n_estimators_entry.insert(0,'10, 20, 30 (default)')
												self.n_estimators_entry.update()
								except:
										self.params['n_estimators']=[10, 20, 30]
										self.n_estimators_entry.destroy()
										self.n_estimators_entry = Entry(self, bd =5)
										self.n_estimators_entry.grid(row = 0, column =1)
										self.n_estimators_entry.insert(0,'10, 20, 30 (default)')
										self.n_estimators_entry.update()
								self.params['criterion']=self.criterionMenu.get()
								self.params['max_depth']=self.max_depth_entry.get()
								try:
										if self.params['max_depth']=='None':
												self.params['max_depth']=None
										else:
												self.params['max_depth']=float(self.params['max_depth'])
												self.params['max_depth']=int(self.params['max_depth'])
												if self.params['random_state']<1:
														self.params['max_depth']=None
														self.max_depth_entry.destroy()
														self.max_depth_entry = Entry(self, bd =5)
														self.max_depth_entry.grid(row = 2, column =1)
														self.max_depth_entry.insert(0,'None (default)')
														self.max_depth_entry.update()
								except:
										self.params['max_depth']=None
										self.max_depth_entry.destroy()
										self.max_depth_entry = Entry(self, bd =5)
										self.max_depth_entry.grid(row = 2, column =1)
										self.max_depth_entry.insert(0,'None (default)')
										self.max_depth_entry.update()
								
						elif self.Classifier=='k-NN':
								self.params['n_neighbors']=self.n_neighbors_entry.get()
								self.params['n_neighbors']=self.params['n_neighbors'].split(',')
								try:
										self.params['n_neighbors']=[float(i) for i in self.params['n_neighbors']]
										self.params['n_neighbors']=[int(i) for i in self.params['n_neighbors']]
										v=[0 if i>0 else 1 for i in self.params['n_neighbors']]
										if sum(v)>0:
												self.params['n_neighbors']=[1,2,3,4,5,6,7,8,9]
												self.n_neighbors_entry.destroy()
												self.n_neighbors_entry = Entry(self, bd =5)
												self.n_neighbors_entry.grid(row = 0, column =1)
												self.n_neighbors_entry.insert(0,'1,2,3,4,5,6,7,8,9 (default)')
												self.n_neighbors_entry.update()
								except:
										self.params['n_neighbors']=[1,2,3,4,5,6,7,8,9]
										self.n_neighbors_entry.destroy()
										self.n_neighbors_entry = Entry(self, bd =5)
										self.n_neighbors_entry.grid(row = 0, column =1)
										self.n_neighbors_entry.insert(0,'1,2,3,4,5,6,7,8,9 (default)')
										self.n_neighbors_entry.update()
								self.params['weights']=self.weightsMenu.get()
								self.params['algorithm']=self.algorithmMenu.get()
								self.params['leaf_size']=self.leaf_size_entry.get()
								try:
										self.params['leaf_size']=float(self.params['leaf_size'])
										self.params['leaf_size']=int(self.params['leaf_size'])
										if self.params['leaf_size']<1:
												self.params['leaf_size']=30
												self.leaf_size_entry.destroy()
												self.leaf_size_entry = Entry(self, bd =5)
												self.leaf_size_entry.grid(row = 3, column =1)
												self.leaf_size_entry.insert(0,'30 (default)')
												self.leaf_size_entry.update()
								except:
										self.params['leaf_size']=30
										self.leaf_size_entry.destroy()
										self.leaf_size_entry = Entry(self, bd =5)
										self.leaf_size_entry.grid(row = 3, column =1)
										self.leaf_size_entry.insert(0,'30 (default)')
								self.params['p']=self.p_entry.get()
								try:
										self.params['p']=float(self.params['p'])
										self.params['p']=int(self.params['p'])
										if self.params['p']<1:
												self.params['p']=2
												self.p_entry.destroy()
												self.p_entry = Entry(self, bd =5)
												self.p_entry.grid(row = 4, column =1)
												self.p_entry.insert(0,'2 (default)')
												self.p_entry.update()
								except:
										self.params['p']=2
										self.p_entry.destroy()
										self.p_entry = Entry(self, bd =5)
										self.p_entry.grid(row = 4, column =1)
										self.p_entry.insert(0,'2 (default)')
							   
						else:
								pass              
						

						print(self.params)
						# Start Classification
						#try:
						kk=0
						if kk==0:       
								cl.Classification(TrainFeatures=self.TrainFeatures, TestFeatures=self.TestFeatures, classifier=self.Classifier,params=self.params, conf_matrix=self.cm)

								# Enable start button
								self.Startbutton['state'] = 'normal'
								self.Startbutton.update()
								
								# Update some labels
								self.label2['text']="         Process completed!          " 
								self.label3['text']="        "
								self.label2.update()
								self.label3.update()
								
								# If confusion matrix flag is enabled
								if self.cm==1:
										self.Plotbutton['state'] = 'normal'
										self.Plotbutton.update()
								else:
										self.Plotbutton['state'] = 'disable'
										self.Plotbutton.update()
						#except:
						else:
								self.error=1
								self.msgErr1='At least one of the input files is invalid.'
								self.msgErr2='Read on User Manual how inputs files must be.'
								self.HomePage()

		def clear(self):
				wlist = self.grid_slaves()
				for l in wlist:
						l.destroy()




if __name__ == "__main__":
		app = ClassificationApp()
		app.mainloop()
