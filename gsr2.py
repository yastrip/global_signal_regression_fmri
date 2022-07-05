
### This is a code for Global Signal Regression of fMRI data.
### Based on:
### Fox et al., 2019. The Global Signal and Observed Anticorrelated Resting State Brain Networks. J. Neurophysiol.

# We need to import the following Python libraries before running the code: I think it's numpy and matplotlib

import numpy as np
import math
import matplotlib

### The code below creates an empty matrix containing two hundred sixty four rows (as many as rois) and eight hundred
### fifty columns (as many as timepoints),
### then the matrix is filled with the values taken from two hundred sixty four different .txt files using a for loop.

#subj139
matrix139 = np.zeros([264,850])
for i in range(1,265):
    myfile = open("//Users//TechnionLab//Documents//Research//fmri_analysis//globalsignal//timeseries//timeseries_dyslexia//139//subj139.roi" + str(i) + ".txt", 'r')
    myline = (myfile.read())
    mytextrow = myline.split('  \n')
    #mytextrow_array = np.array (mytextrow)
    #print(mytextrow_array[1])
    #print(type(mytextrow_array[1]))
    for x in range(0,850):
        # print (float(mytextrow[x]))
        matrix139[i-1, x] = float(mytextrow[x])

# Here, we will save the raw matrix as a .txt file

np.savetxt('sub139_matrix.txt', matrix139, fmt='%.2f')

print('Timeseries found.')

### Now we create an empty array in which there is eight fifty rows and 1 column. We will then fill this
### vector with the global signal (from a .txt file). That is the signal
### we will regress out of the subject's matrix (matrix139).

gsr = np.zeros([850,1])

gsrfile = open("//Users//TechnionLab//Documents//Research//fmri_analysis//globalsignal//Dyslexia//globalsignalsubject_139//globalsignal.txt")
gsrline = (gsrfile.read())
gsrrow = gsrline.split('  \n')
#print(gsrrow)
for h in range(0, 850):
    gsr[h] = float(gsrrow[h])

print('Global signal found.')

#print (gsr)

# In the next lines I am just checking the dimensions of the gsr or g matrix (the global signal for subj87), as well
# as the lenght of the matrix139 (raw matrix before regression).

# rows = len(gsr) # Height.
# columns = len(gsr[0]) # Width.
# print("rows gsr = ", rows)
# print("columns gsr = ",  columns)
# rowsmat = len(matrix139) # Height.
# columnsmat = len(matrix139[0]) # Width.
# print("rows matrix = ", rowsmat)
# print("columns matrix = ",  columnsmat)

# In the valid code the result of this print is:
# ('rows gsr = ', "eight five zero")
# ('columns gsr = ', "one")
# ('rows matrix = ', "two six four")
# ('columns matrix = ', "eight five zero")

### Now we need to implement the Global Signal Regression formula, which is defined as:
### B' = B - g * B_sub_g
### B_sub_g = g.pseudoinverse * B
### g.pseudoinverse = gplus = ([g.T * g] ** -1) * g.T
### B is a matrix with n or eight hundred fifty rows (timepoints) and m or two hundred sixty four columns (rois)
### g is the global signal vector with eight fifty rows (timepoints) and 1 column
### This formula was taken from:
### Fox et al., 2009. The Global Signal and Observed Anticorrelated Resting State Brain Networks. J. Neurophysiol.

# gsr can be easily transposed using gsr.T

gsr_T_times_gsr = np.dot(gsr.T, gsr)
#print(gsr_T_times_gsr)

#    np.tensordot(gsr, gsr)
#print(gsr_T_times_gsr)

# CHECKING MATRIX PROPERTIES/OPERATIONS
# a = np.array([[1, 2, 3], [4, 5, 6]])
# print(a)
# b = np.array([[1, 2, 3]])
# print(b)
# # c = np.tensordot(a, a)
# # print(c)
# d = np.dot(a,b)
# print(d)

totheminusone = gsr_T_times_gsr ** -1
#print(totheminusone)
gplus = totheminusone*gsr
#print(gplus)

### The formula below verifies the property g.plus * g = 1 or g.pseudoinverse * g = 1 :
### The resul of np.dot(gsr.T,gplus) should be equal to one. Else, there has been an error in the definition of
### gplus, g.plus or g.pseudoinverse, however you want to call it.
###### Note than I am using gsr.T here instead of gsr because of the specific properties of vectors in Python.

#print(np.dot(gsr.T,gplus))

betasubg = np.dot(gplus.T, matrix139.T)
#print(betasubg)

### Here, we can check the size of betasubg
### The lenght of betasubg should be one row, two hundred sixty four columns

#get the number of rows.
rows2 = len(betasubg)
#get the number of columns.
cols2 = len(betasubg[ 0 ])
#print( 'Length of betasubg is' , rows2)
#print( 'Number of columns' , cols2)
#print( 'Total number of elements' , rows2 * cols2)


g_times_betasubg = np.dot(gsr, betasubg)
#print(g_times_betasubg)

### Here, we can check the size of g_times_betasubg
### The lenght of g_times_betasubg should be eight hundred fifty rows, two hundred sixty four columns

#get the number of rows.
rows3 = len(g_times_betasubg)
#get the number of columns.
cols3 = len(g_times_betasubg[ 0 ])
#print( 'Length of g times betasubg is' , rows3)
#print( 'Number of columns' , cols3)
#print( 'Total number of elements' , rows3 * cols3)

regressedmatrix139trans = matrix139.T - betasubg
#print(regressedmatrix139trans)

### The code belows checks the size of matrix 'regressedmatrix139trans'
### Should be eight hundred fifty rows (timepoints) and two hundred sixty four columns (rois).

#get the number of rows.
rows = len(regressedmatrix139trans)
#get the number of columns.
cols = len(regressedmatrix139trans[ 0 ])
#print( 'Length of regressedmatrix139 is' , rows)
#print( 'Number of columns' , cols)
#print( 'Total number of elements' , rows * cols)

regressedmatrix139 = regressedmatrix139trans.T

### The code belows checks the size of matrix 'regressedmatrix139'.
### Should be two hundred sixty four columns (rois) x eight hundred fifty columns (timepoints).
### This is the final output of the code that we can save as a .txt file using np.savetxt

#get the number of rows.
rows = len(regressedmatrix139)
#get the number of columns.
cols = len(regressedmatrix139[ 0 ])
#print( 'Length of regressedmatrix4 is' , rows, 'rows', cols, 'columns')
#print( 'Total number of elements' , rows * cols)

#print(regressedmatrix139)

np.savetxt('sub139_regressedmatrix.txt', regressedmatrix139, fmt='%.2f')

print('The matrix for subject 139 has been globally regressed and saved.')

