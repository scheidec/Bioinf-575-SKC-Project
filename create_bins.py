# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 19:48:33 2017

@author: Kaiwen Lin
"""
import Data_processing as dp
import numpy as np


record_file = open("file_records.txt")
lines = record_file.read().split('\n')
file_records = []
for i in xrange(len(lines)):
    if lines[i]: 
        file_records.append(lines[i])
record_file.close()


data_ary_names = []
for i in xrange(len(file_records)):
    data_fa =  open(file_records[i])
    features = dp.ft_ext(data_fa)
    features = np.delete(features, 0, 1) #delete the name column
                                         # so it would be all numbers
    sequence_count = len(features)
    label = i + 1
    label_col = np.full((sequence_count, 1), label) # create label vector
    data = np.c_[features, label_col] # combine the labels into data
    
    data_ary_names[i] = file_records[i] + "_features.npy"
    np.save(data_ary_names[i], data)
    
##########################################################
'''
Create 2-class classification bins

'''




##########################################################
'''
Create 4-class classification bins

'''




##########################################################
'''
Create 8-class classification bins

'''






    
