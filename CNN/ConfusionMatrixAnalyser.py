#!/usr/bin/env python3

# Authors: Boretto Luca    <luca.boretto@studenti.polito.it>
#          Salehi  Raheleh <raheleh.salehi@studenti.polito.it>

""" This script taking as input a .json file having the structure {"":"","name1":[ConfusionMatrix1,Classes1],"name2":[ConfusionMatrix2,Classes2],...}
    can return a .xlsx file containing statistics about each classification task (each key in the dictionary).
    - ConfusionMatrix: list, size nclasses x nclasses -> example with 3 classes [[.,.,.],[.,.,.],[.,.,.]]
    - Classes: list, size nclasses -> example with 3 classes [.,.,.]
    This .json file is obtained in Results/Classification/ConfusionMatrix folder as output file of Classification.py script (or Classification_GUI.py).
    An output file called report.xlsx will be generated.
"""
import json
import numpy as np
import src.myfunctions as mf
import pandas as pd
import xlsxwriter
import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()
# Select JSON file
file_path = filedialog.askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("JSON files","*.json"),("all files","*.*")))
try:
        # Read file and loads values
        f=open(file_path,'r')
        data=f.read()
        data=json.loads(data)
        names=list(data.keys())
        names=names[1:]
        categories=data[names[1]][1]

        # Initialization of the output structure
        out={}
        out["name"]=[]
        out["accuracy"]=[]
        out["std_out_diag"]=[]
        for i in range(len(categories)):
                out["pcc"+categories[i]]=[]
        for i in range(len(categories)):
                for j in range(len(categories)):
                        if i!=j:
                                out["p"+categories[i]+"c"+categories[j]]=[]
            
        for name in names:
                out["name"].append(name)
                cm=data[name][0]
                # Accuracy of classification
                out["accuracy"].append(mf.cm_score(np.asarray(cm)))
                # Standard deviation of of out of diagonal elements
                out["std_out_diag"].append(mf.stdev_out_of_diagonal(np.asarray(cm)))
                cat=data[name][1]
                # Percentages of correct classified for each class
                for i in range(len(cat)):
                        out["pcc"+cat[i]].append(float(cm[i][i])/sum(cm[i]))
                # Percentages of wrong classified as ... for each class       
                for i in range(len(cat)):
                        for j in range(len(cat)):
                                if i!=j:
                                        out["p"+cat[i]+"c"+cat[j]].append(float(cm[i][j])/sum(cm[i]))
                        
                        
        # Convert data results to panda DataFrame
        df = pd.DataFrame(out)

        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter('report.xlsx', engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1')

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        f.close()
        print('Process Complete')
except:
        print('Invalid json file')
