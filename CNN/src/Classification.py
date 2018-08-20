#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <s226014@studenti.polito.it>

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats
import h5py
import pandas as pd
import xlsxwriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
try:
		import src.myfunctions as mf
except:
		import myfunctions as mf
import json
import os
import itertools

def Classification(TrainFeatures,TestFeatures,classifier,params={},conf_matrix=0):
		""" It performs dimensionality reduction of a training and a test features matrix
			stored in a .h5 file each.
			It's possible to use 5 different methods for dimensionality reduction.
		_____________________________________________________________________________________

		Parameters:
				- TrainFeatures: string
						It is the path of an .h5 file of the training features.
						It contains at least the following datasets:
						- One between 'pca','tsne','tsvd','lda','nmf','feats':
						  array-like, shape (n_samples, n_features)
						- 'labels':  array-like, shape (n_samples, )
						- 'img_ids': array-like, shape (n_samples, )
						For more information see FeaturesReduction.py
				- TestFeatures: string
						It is the path of an .h5 file of the test features.
						It contains at least the same datasets.
				- classifier: string
						Possible value are:
								- 'SVM': Support vector machine
								- 'RandomForest': Random forest
								- 'NaiveBayes': Naive Bayes classifier
								- 'k-NN': K-nearest neighbors
				- params: dict
						It is a dictionary containig parameters for the selected classifier.
						Keys and possible values are listed on the following websites:
						http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
						http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
						http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
						http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
				- conf_matrix: int (0 or 1)
						If conf_matrix=1 confusion matrix is performed else it is not performed. Set this parameter to 1 only
						when TrainFeatures and TestFeatures are both labelled with significant label, otherwise set it to 0 
				
		Returns:
				- predicted_labels: list
						It is a list containing the predicted labels
				- cnf_matrix: array-like or None
						If conf_matrix=0 cnf_matrix is None else if conf_matrix=1 it is the computed confusion matrix
				- categories: list
						It is the list of the classes of the data set
				
				Furthermore, automatically an .xlsx file containing results of the classification is created in Classification/Predictions
				folder and if conf_matrix is set on 1 an image of the computed confusion matrix is saved in the folder Classification/ConfusionMatrix

		Example usage:
				import Classification as cl
				TrainFeatures='Training.h5'
				TestFeatures='Test.h5'
				classifier='RandomForest'
				params={'n_estimators':10}
				conf_matrix=1
				predicted_labels,cnf_matrix,categories=cl.Classification(TrainFeatures,TestFeatures,classifier,params,conf_matrix)
		""" 
		s=os.sep
		# Load training reduced features, labels and names of the images
		train = h5py.File(TrainFeatures, 'r')
		train_img_ids=train['img_id'][:]
		method=mf.get_intersection(train,['pca','tsne','tsvd','lda','nmf','feats'])
		train_features = np.transpose(np.array(train[method[0]]))
		train_labels = np.array(train['labels'])
		train_labels = np.squeeze(train_labels)	
		train.close()
		
		# Load test reduced features and names of the images
		test = h5py.File(TestFeatures, 'r')
		test_img_ids=test['img_id'][:]
		method=mf.get_intersection(test,['pca','tsne','tsvd','lda','nmf','feats'])
		test_features = np.transpose(np.array(test[method[0]]))
		test.close()

		# Check that number of features is equal
		if train_features.shape[1]!=test_features.shape[1]:
				train_features = np.transpose(train_features)
				test_features = np.transpose(test_features)
				
		# Initialize the selected classifier
		if classifier=='SVM':	
				clf=SVC()
				#params['verbose']=True
				Gamma=list(params['gamma'])
				C=list(params['C'])
				del params['gamma']
				del params['C']
				params1=[Gamma,C]
				params_grid=list(itertools.product(*params1))
				for i in range(len(params_grid)):
						tmp=tuple(params_grid[i])
						params_grid[i]={}
						params_grid[i]['C']=tmp[1]
						params_grid[i]['gamma']=tmp[0]
		elif classifier=='RandomForest':
				clf=RandomForestClassifier()
				#params['verbose']=1
				params['n_jobs']=-1
				NE=list(params['n_estimators'])
				del params['n_estimators']
				params1=[NE]
				params_grid=list(itertools.product(*params1))
				for i in range(len(params_grid)):
						tmp=tuple(params_grid[i])
						params_grid[i]={}
						params_grid[i]['n_estimators']=tmp[0]
		elif classifier=='NaiveBayes':
				clf=GaussianNB() 
		elif classifier=='k-NN':
				clf=KNeighborsClassifier()
				NN=list(params['n_neighbors'])
				del params['n_neighbors']
				params1=[NN]
				params_grid=list(itertools.product(*params1))
				for i in range(len(params_grid)):
						tmp=tuple(params_grid[i])
						params_grid[i]={}
						params_grid[i]['n_neighbors']=tmp[0]
		else:
				raise TypeError("Invalid classifier: possible models are 'SVM', 'RandomForest','NaiveBayes' and k-NN")
		# Set parameters of classifier
		clf.set_params(**params)
		
		
		# Initialize output filenames	
		outputname=TestFeatures.split(s)[-1]
		outputname=classifier+'_'+(outputname.split('.')[0])

		# Create folder in which save results
		mf.folders_creator('Results',['Classification'])
		mf.folders_creator('Results'+s+'Classification',['ConfusionMatrix','Predictions'])
				
		# TUNING OF PARAMETERS
		# Perform cross validation to evaluate estimator performance
		# and set the best parameters
		if classifier!='NaiveBayes':
			best_accuracy=0
			pos=-1
			for ind,p in enumerate(params_grid):
					clf.set_params(**p)
					scores = cross_val_score(clf, train_features, train_labels, cv=10)
					cross_accuracy=scores.mean()
					if cross_accuracy>best_accuracy:
							best_accuracy=cross_accuracy
							pos=ind
					cross_std=scores.std()
			clf.set_params(**params_grid[pos])
		else:
			scores = cross_val_score(clf, train_features, train_labels, cv=10)
			cross_accuracy=scores.mean()
			cross_std=scores.std()
			
		print('final params')
		print(clf.get_params())
		
		# Fit the classifier on the training set
		clf.fit(train_features, train_labels)
		
		# Perform Predictions 
		predicted_labels = clf.predict(test_features)
		print('predicted labels')
		print(predicted_labels)
		
		# If the selected classifier uses random methods we have to break down
		# fortuity performing more times the prediction and to make the mode
		# the mode of the prediction outputs
		if classifier == 'RandomForest':
				p=[predicted_labels]
				for i in range(9):
						clf=RandomForestClassifier()
						clf.set_params(**params)
						clf.set_params(**params_grid[pos])
						clf.fit(train_features, train_labels)
						p.append(clf.predict(test_features))
				predicted_labels=np.array(p)
				predicted_labels,_=stats.mode(predicted_labels,axis=0)
				predicted_labels=np.ndarray.tolist(predicted_labels)[0]
		
		# Get categories of the training set from its image ids
		categories=mf.get_categories(train_img_ids)

		# Check that everything worked properly
		counter=[0]*len(categories)	
		for i in range(int(len(predicted_labels))):
				counter[int(predicted_labels[i])]=counter[int(predicted_labels[i])]+1		
		assert sum(counter) == len(predicted_labels)
		
		# Calculate the percentage of images classified with a certain label
		# on the number of images of test set
		for i in range(len(counter)):
				counter[i]=counter[i]/len(predicted_labels)*100
		
		# If you set conf_matrix flag to 1 a confusion matrix will be
		# performed considering the classification as supervised
		# (considering training set folders names = to test set folders names
		# as classes)
		if conf_matrix==1:
				test_categories=mf.get_categories(test_img_ids)
				y_test=[]
				for i in range(test_img_ids.shape[0]):
						tmp=test_img_ids[i].decode('UTF-8')
						tmpl=tmp.replace('/', os.sep)
						tmp=tmpl.replace('\\', os.sep)
						clas=tmp.split(s)[-2]
						y_test.append(clas)

				# Verify that effectivelly training set folders names = to test set folders names
				assert sorted(test_categories)==sorted(categories)	
				
				# Convert string test folder names to the corrispondent numeric
				# value 	
				for i in range(len(y_test)):
						for j in range(len(categories)):
								if categories[j]==y_test[i]:
										y_test[i]=j
				y_test=np.asarray(y_test)
				
				# Perform confusion matrix
				cnf_matrix = confusion_matrix(y_test, predicted_labels)
				
				# Get accuracy
				accuracy=mf.cm_score(cnf_matrix)
				# Get standard deviation of the value outside the diagonal
				standard_dev=mf.stdev_out_of_diagonal(cnf_matrix)
				
				try:
						f=open('Results'+s+'Classification'+s+'ConfusionMatrix'+s+'CM_database.json','r')
						json_flag=1
				except:
						f=open('Results'+s+'Classification'+s+'ConfusionMatrix'+s+'CM_database.json','w')
						f.write('{"":""}')
						f.close()
						json_flag=0
				if json_flag==0:
						f=open('Results'+s+'Classification'+s+'ConfusionMatrix'+s+'CM_database.json','r')	
				CMs=f.read()
				f.close()
				CMs=json.loads(CMs)
				CMs[outputname]=[np.ndarray.tolist(cnf_matrix), categories,clf.get_params()]
				CMs=json.dumps(CMs)
				f=open('Results'+s+'Classification'+s+'ConfusionMatrix'+s+'CM_database.json','w')
				f.write(CMs)
				f.close
								
				# Save Confusion matrix coloured plot as image
				fig,ax=plt.subplots()
				mf.plot_confusion_matrix(cm=cnf_matrix, classes=categories,normalize=True)
				plt.savefig('Results'+s+'Classification'+s+'ConfusionMatrix'+s+outputname+'.png')
		else:
				cnf_matrix=None

		# Convert test images ids from array to list 
		image_paths=[line.decode("utf-8") for line in test_img_ids]
		# Convert predicted numerical label corrispondent string class
		labels=[categories[int(label)] for label in predicted_labels]
		assert len(image_paths) == len(predicted_labels)


		# Prepare results to write in a .xlsx file 
		c=['']*len(labels)
		c[0]=cross_accuracy
		data={'0) Image path': image_paths, '1) Predicted class':labels}
		data['2) '+classifier+' Cross-validation accuracy']=c
		if conf_matrix==1:
				a=['']*len(labels)
				a[0]=accuracy
				data['3) TestSet Classification accuracy']=a
				b=['']*len(labels)
				b[0]=standard_dev
				data['4) Standard Deviation of elements outside CM diagonal']=b
		
		# Convert data results to panda DataFrame
		df = pd.DataFrame(data)
		
		# Create a Pandas Excel writer using XlsxWriter as the engine
		writer = pd.ExcelWriter('Results'+s+'Classification'+s+'Predictions'+s+outputname+'.xlsx', engine='xlsxwriter')

		# Convert the dataframe to an XlsxWriter Excel object.
		df.to_excel(writer, sheet_name='Sheet1')

		# Close the Pandas Excel writer and output the Excel file.
		writer.save()
		return predicted_labels, cnf_matrix, categories
				
		
if __name__=="__main__":

		Layers=['Keras VGG-19_block1_conv2']
		
		for layer in Layers:
				for classifier in ['SVM','RandomForest','NaiveBayes']:
						for n in [2,3]:
								for m in ['PCA','t-SNE']:

										TrainFeatures='Results'+os.sep+'SelectedFeatures'+os.sep+m+str(n)+'_'+ layer+'_Training-RGB2.h5'
										TestFeatures='Results'+os.sep+'SelectedFeatures'+os.sep+m+str(n)+'_'+ layer+'_Test-RGB2.h5'
										Classification(TrainFeatures,TestFeatures,classifier,conf_matrix=1)	
