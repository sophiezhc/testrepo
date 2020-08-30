############################## lecture7
import numpy as np


mylist1 = [1,2,3,4]
myarray1 = np.array(mylist1)

mylist2 = [11,22,33,44]
mylists = [mylist1,mylist2]

myarray2 = np.array(mylists)

myarray2.shape
myarray2.dtype

my_zeros_array = np.zeros(5)
my_zeros_array.dtype

my_ones_array = np.ones([5,5])
np.ones([2,3])
np.empty(5) # =np.zeros(5)

np.eye(5) # identity matrix

np.arange(5) # = array([0,1,2,3,4])
np.arange(5,50,2) # = array([5,7,9,...,45,47,49])


############################## lecture8

# from __future__ import division # python3.x already included

arr1 = np.array([[1,2,3,4],[8,9,10,11]])

arr1 * arr1
arr1 - arr1
1 / arr1
arr1 ** 3 #power3

############################ lecture9

arr = np.arange(0,11) # array([0,1,2,3,4,5,6,7,8,9,10])
arr[8] # 8
arr[1:5] # array([1,2,3,4])
arr[0:5] # array([0,1,2,3,4])
len(arr)
arr[0:11:2] # array([0,2,4,6,8,10])
arr[1:11:2] # array([1,3,5,7,9])
arr[2:10:2] # array([2,4,6,8])

arr[0:5] = 100 # array([100, 100, 100, 100, 100, 5, 6, 7, 8, 9, 10])

arr = np.arange(0,11) # array([0,1,2,3,4,5,6,7,8,9,10])
slice_of_arr = arr[0:6] # array([0,1,2,3,4,5])
slice_of_arr[:] = 99

arr_copy = arr.copy()

arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45])) #array([[ 5, 10, 15],[20, 25, 30],[35, 40, 45]])
arr_2d[1] #array([20,25,30])
arr_2d[1][0] #20
arr_2d[:2,1:]

arr2d = np.zeros((10,10))
arr2d_length = arr2d.shape[1]

for i in range(arr2d_length):
    arr2d[i,:] = i

############################ lecture10

arr = np.arange(50).reshape((10,5))
#array([[ 0,  1,  2,  3,  4],
#       [ 5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14],
#       ..., 
#       [35, 36, 37, 38, 39],
#       [40, 41, 42, 43, 44],
#       [45, 46, 47, 48, 49]])
    
arr[:,[2,4]]
#array([[ 2,  4],
#       [ 7,  9],
#       [12, 14],
#       ..., 
#       [37, 39],
#       [42, 44],
#       [47, 49]])
    
arr[[3,5,7]]
#array([[15, 16, 17, 18, 19],
#       [25, 26, 27, 28, 29],
#       [35, 36, 37, 38, 39]])
    
arr[[3,5,7],2:4]
#array([[17, 18],
#       [27, 28],
#       [37, 38]])

arr[[3,5,7]][2:4]
#array([[35, 36, 37, 38, 39]])

arr[[3,5,7]][:,[2,4]]
#array([[17, 19],
#       [27, 29],
#       [37, 39]])
    

arr.T #transpose
#array([[ 0,  5, 10, ..., 35, 40, 45],
#       [ 1,  6, 11, ..., 36, 41, 46],
#       [ 2,  7, 12, ..., 37, 42, 47],
#       [ 3,  8, 13, ..., 38, 43, 48],
#       [ 4,  9, 14, ..., 39, 44, 49]])
    
np.dot(arr.T, arr) 

arr3d = np.arange(50).reshape((5,5,2))
arr3d.transpose((1,0,2))

arr = np.array([[1,2,3]])
arr.swapaxes(0,1)

############################################ lecture11

arr = np.arange(10)
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.sqrt(arr) #taking square root of every value in the array
#array([ 0.        ,  1.        ,  1.41421356,  1.73205081,  2.        ,
#        2.23606798,  2.44948974,  2.64575131,  2.82842712,  3.        ])

np.exp(arr) #taking exponential of every value in the array
#array([  1.00000000e+00,   2.71828183e+00,   7.38905610e+00,
#         2.00855369e+01,   5.45981500e+01,   1.48413159e+02,
#         4.03428793e+02,   1.09663316e+03,   2.98095799e+03,
#         8.10308393e+03])

A = np.random.randn(10)
#array([ 0.08672012, -0.09407779, -0.66770868,  0.37547263,  0.31110819,
#       -0.88775505, -0.96525686, -2.71480423, -1.52325153, -0.70833612])
    
B = np.random.randn(10)
#array([-0.86092215,  0.44851192,  0.54336308, -0.09485297, -0.86373302,
#        1.40470089,  1.08652835,  1.69949359, -1.12132535, -1.21255151])

# Binary Functions
np.add(A,B)
#array([-0.77420203,  0.35443413, -0.12434559,  0.28061965, -0.55262483,
#        0.51694584,  0.12127149, -1.01531064, -2.64457688, -1.92088763])

np.maximum(A,B)
#array([ 0.08672012,  0.44851192,  0.54336308,  0.37547263,  0.31110819,
#        1.40470089,  1.08652835,  1.69949359, -1.12132535, -0.70833612])
np.minimum(A,B)


### a list of math operation universal functions 

import webbrowser
website = 'https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs'
# webbrowser.open(website)


############################################ lecture12

import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline

points = np.arange(-5,5,0.01)
dx,dy = np.meshgrid(points,points)

z = (np.sin(dx) + np.sin(dy))

#plt.imshow(z)
#plt.colorbar()
#plt.title('Plot for sin(x)+sin(y)')


#numpy where
A = np.array([1,2,3,4])
B = np.array([100,200,300,400])

condition = np.array([True,True,False,False])

answer = [(A_val if cond else B_val) for A_val,B_val,cond in zip(A,B,condition)]

answer2 = np.where(condition,A,B)

from numpy.random import randn
arr = randn(5,5)
np.where(arr<0,0,arr)

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr.sum()
arr.sum(0) #sum by the columns
arr.mean()
arr.std()
arr.var() #variance

bool_arr = np.array([True,False,True])
bool_arr.any() #return True if any value is true
bool_arr.all() #returen True if all the values are true


# Sort
arr = randn(5)
arr.sort() # sort from least to greatest

countries = np.array(['France','Germany','USA','Russia','USA','Mexico','Germany'])
np.unique(countries)
#array(['France', 'Germany', 'Mexico', 'Russia', 'USA'], 
#      dtype='<U7')
np.in1d(['France','USA','Sweden'],countries)
#array([ True,  True, False], dtype=bool)


############################################ lecture13

arr = np.arange(5)
#arr = array([0, 1, 2, 3, 4])
np.save('myarray',arr)
arr = np.arange(5)
#arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.load('myarray.npy')
#array([0, 1, 2, 3, 4])
arr1 = np.load('myarray.npy')
#array([0, 1, 2, 3, 4])
arr2 = arr












