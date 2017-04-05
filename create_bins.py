# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 19:48:33 2017

@author: Kaiwen Lin
"""
import Data_processing as dp
import numpy as np
from sklearn.model_selection import train_test_split

def mesh_data_two_class(dataname1, dataname2):
    data1 = np.load(dataname1)
    data2 = np.load(dataname2)
    row_length = min(len(data1), len(data2))
    data1 = np.random.permutation(data1)
    data1 = data1[0:row_length, :]
    data2 = np.random.permutation(data2)
    data2 = data2[0:row_length, :]
    data_two_class = np.random.permutation(np.concatenate((data1, data2), axis = 0))
    data_two_class_train,  data_two_class_test = train_test_split(data_two_class, test_size=0.33, random_state=42)
    return data_two_class_train,  data_two_class_test


def mesh_data_four_class(dataname1, dataname2, dataname3, dataname4):
    data1 = np.load(dataname1)
    data2 = np.load(dataname2)
    data3 = np.load(dataname3)
    data4 = np.load(dataname4)
    row_length = min(len(data1), len(data2), len(data3), len(data4))
    data1 = np.random.permutation(data1)
    data1 = data1[0:row_length, :]
    data2 = np.random.permutation(data2)
    data2 = data2[0:row_length, :]
    data3 = np.random.permutation(data3)
    data3 = data3[0:row_length, :]
    data4 = np.random.permutation(data4)
    data4 = data4[0:row_length, :]
    data_four_class = np.random.permutation(np.concatenate((data1, data2, 
                                                            data3, data4), axis = 0))
    data_four_class_train,  data_four_class_test = train_test_split(data_four_class, test_size=0.33, random_state=42)
    return data_four_class_train,  data_four_class_test

def mesh_data_eight_class(dataname1, dataname2, dataname3, dataname4, 
                          dataname5, dataname6, dataname7, dataname8):
    data1 = np.load(dataname1)
    data2 = np.load(dataname2)
    data3 = np.load(dataname3)
    data4 = np.load(dataname4)
    data5 = np.load(dataname5)
    data6 = np.load(dataname6)
    data7 = np.load(dataname7)
    data8 = np.load(dataname8)
    row_length = min(len(data1), len(data2), len(data3), len(data4), 
                     len(data5), len(data6), len(data7), len(data8))
    data1 = np.random.permutation(data1)
    data1 = data1[0:row_length, :]
    data2 = np.random.permutation(data2)
    data2 = data2[0:row_length, :]
    data3 = np.random.permutation(data3)
    data3 = data3[0:row_length, :]
    data4 = np.random.permutation(data4)
    data4 = data4[0:row_length, :]
    data5 = np.random.permutation(data5)
    data5 = data5[0:row_length, :]
    data6 = np.random.permutation(data6)
    data6 = data6[0:row_length, :]
    data7 = np.random.permutation(data7)
    data7 = data7[0:row_length, :]
    data8 = np.random.permutation(data8)
    data8 = data8[0:row_length, :]
    data_eight_class = np.random.permutation(np.concatenate((data1, data2, 
                                                            data3, data4, 
                                                            data5, data6, 
                                                            data7, data8
                                                            ), axis = 0))
    data_eight_class_train, data_eight_class_test = train_test_split(data_eight_class, test_size=0.33, random_state=42)
    return data_eight_class_train, data_eight_class_test
    

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
    
    data_ary_names.append(file_records[i] + "_features.npy")
    np.save(data_ary_names[i], data)
    
##########################################################
'''
Create 2-class classification bins

'''
two_class_12_train, two_class_12_test = mesh_data_two_class(data_ary_names[0], data_ary_names[1])
two_class_34_train, two_class_34_test = mesh_data_two_class(data_ary_names[2], data_ary_names[3])
two_class_56_train, two_class_56_test = mesh_data_two_class(data_ary_names[4], data_ary_names[5])
two_class_78_train, two_class_78_test = mesh_data_two_class(data_ary_names[6], data_ary_names[7])
np.save('two_class_12_train', two_class_12_train)
np.save('two_class_12_test', two_class_12_test)
np.save('two_class_34_train', two_class_34_train)
np.save('two_class_34_test', two_class_34_test)
np.save('two_class_56_train', two_class_56_train)
np.save('two_class_56_test', two_class_56_test)
np.save('two_class_78_train', two_class_78_train)
np.save('two_class_78_test',two_class_78_test)


##########################################################
'''
Create 4-class classification bins

'''
four_class_1234_train, four_class_1234_test = mesh_data_four_class(data_ary_names[0], data_ary_names[1], 
                                                                   data_ary_names[2], data_ary_names[3])
four_class_5678_train, four_class_5678_test = mesh_data_four_class(data_ary_names[4], data_ary_names[5], 
                                                                   data_ary_names[6], data_ary_names[7])
np.save('four_class_1234_train', four_class_1234_train)
np.save('four_class_1234_test', four_class_1234_test)
np.save('four_class_5678_train', four_class_5678_train)
np.save('four_class_5678_test', four_class_5678_test)


##########################################################
'''
Create 8-class classification bins

'''
eight_class_train, eight_class_test = mesh_data_eight_class(data_ary_names[0], data_ary_names[1], 
                                                            data_ary_names[2], data_ary_names[3], 
                                                            data_ary_names[4], data_ary_names[5], 
                                                            data_ary_names[6], data_ary_names[7])
np.save('eight_class_train', eight_class_train)
np.save('eight_class_test', eight_class_test)





    
