# Bioinf-575-SKC-Project
Riboswitch Classification Machine Learning Project

Data preparation: from fasta files at ftp://ftp.ebi.ac.uk/pub/databases/Rfam/12.2/fasta_files/

Feature extraction: extract 20 features (%A, %T, %C, %G, %AT, %TC, ...)

Algorithms to use:
SVM

Multilayer Perceptron (Neural Network)

Random Forest

Naive Bayes

Decision Tree

KNN (Nearest Neighbor)



**Feature Extraction Usage:**

Code in Python:  
import Data_processing as dp  
file = open("RF00059.fa")  
print(dp.ft_ext(file))  

Return:  
a list of lists  
e.g    
[['>BA000032.2/1323634-1323774' '0.25' '0.23' '0.21' '0.31' '0.04' '0.02'
  '0.06' '0.1' '0.05' '0.05' '0.09' '0.06' '0.07' '0.06' '0.11' '0.04'
  '0.03' '0.04' '0.07' '0.04']  
 ['>ACZB01000034.1/21513-21373' '0.25' '0.23' '0.21' '0.31' '0.04' '0.02'
  '0.06' '0.1' '0.05' '0.05' '0.09' '0.06' '0.07' '0.06' '0.11' '0.04'
  '0.03' '0.04' '0.07' '0.04']]

note: each list is in the format of  
['ID', 'A%', 'C%', 'T%', 'G%', 'AA%', 'AC%', 'GT%', 'AG%', 'CC%', 'TT%', 'CG%', 'GG%', 'GC%', 'AT%', 'GA%', 'TG%', 'CT%', 'CA%', 'TC%', 'TA%']  

Labels:  
RF00059 - 1   
RF00174 - 2   
RF00162 - 3   
RF00504 - 4  
RF00168 - 5  
RF00050 - 6  
RF00167 - 7  
RF01051 - 8  

