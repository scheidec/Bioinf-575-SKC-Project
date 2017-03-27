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

Arg:  
Python Data_processing.py file.fa

Return:  
a list of lists
e.g    
[['>BA000032.2/1323634-1323774' '0.248226950355' '0.22695035461'
  '0.212765957447' '0.312056737589' '0.0428571428571' '0.0214285714286'
  '0.0571428571429' '0.1' '0.05' '0.05' '0.0928571428571' '0.0642857142857'
  '0.0714285714286' '0.0642857142857' '0.107142857143' '0.0428571428571'
  '0.0285714285714' '0.0428571428571' '0.0714285714286' '0.0357142857143']
 ['>ACZB01000034.1/21513-21373' '0.248226950355' '0.22695035461'
  '0.212765957447' '0.312056737589' '0.0428571428571' '0.0214285714286'
  '0.0571428571429' '0.1' '0.05' '0.05' '0.0928571428571' '0.0642857142857'
  '0.0714285714286' '0.0642857142857' '0.107142857143' '0.0428571428571'
  '0.0285714285714' '0.0428571428571' '0.0714285714286' '0.0357142857143']]

note: each list is  
['ID', 'A%', 'C%', 'T%', 'G%', 'AA%', 'AC%', 'GT%', 'AG%', 'CC%', 'TT%', 'CG%', 'GG%', 'GC%', 'AT%', 'GA%', 'TG%', 'CT%', 'CA%', 'TC%', 'TA%']
