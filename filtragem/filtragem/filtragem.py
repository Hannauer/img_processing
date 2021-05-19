# Processamento de Imagens
# Trabalho 3
# Caio Ferreira Bernardo - NUSP 9276936
# João Pedro Hannauer - NUSP 9390486

##important observation: the last teste case is returnin the right awnser but it the time of execution is exceeded on run codes 

import numpy as np
import math
import random
import imageio
from numpy.core.shape_base import atleast_1d

def RMSE(g, R):
    assert g.shape == R.shape
    N, M = g.shape
    
    # Convert the images to int32 allowing negative values to compute RMSE
    return np.sqrt((np.sum((g.astype(np.int32)-R.astype(np.int32))**2))/(N*M))

def normalize(matrix, start, end):
    return (matrix - matrix.min())/(matrix.max() - matrix.min()) * (end-start) + start

def filtering_1d(img, w, size):
    
    #Flattten the image in order to have a 1-dimension 
    img_flatten = img.flatten()
    
    # Get parameters
    N,M = img.shape # number of rows and columns (N*M is the length of the flatten image)    
    
    #flatten vecto size
    flatten_size =  N*M
    a = int((size-1)/2) 
    
    #creating result image
    img = np.zeros((N*M)-(2*a))
    
    for i in range (a, (flatten_size)-a):
        img[i-a] = np.sum(np.multiply(w, img_flatten[i-a:i+a+1]))
    
    #padding
    img = np.pad(img, (a,a), 'wrap')
    
    #reshaping
    img = np.reshape(img, (N,M))  
    
    #normalize the result
    return   normalize(img, 0, 255)

def filtering_2d(img, w):
    #getting usefull information 
    N, M = img.shape
    n, m = w.shape
    a = int((n-1)/2)
    b = int((m-1)/2)
    
    #result image
    g = np.zeros((N, M))
    
    #applying convolution
    for i in range(a, N-a):
        for j in range(b, M-b):
            img_temp = img[i-a:i+a+1, j-b:j+b+1]
            
            g[i,j] = np.sum( np.multiply(img_temp, w))
            
    return normalize(g, 0, 255)

def median_filter(img, size):
    #getting usefull information
    n, m = (size,size)
    a = int((n-1)/2)
    b = int((m-1)/2)
    imag= np.pad(img, a, mode = 'constant')
    N, M = imag.shape
   
    #creating result matrix
    g = np.zeros((N, M))
    
    #applying filter
    for i in range(a, N-a):
        for j in range(b, M-b):
            img_temp = imag[i-a:i+a+1, j-b:j+b+1]
            g[i,j] = int(np.median(np.sort(img_temp.flatten())))
    return g[a:-a, a:-a]

def main():
	
    # Get user inputs
    filename = input().rstrip()
    F = int(input().rstrip())

    #reading img
    img  = imageio.imread(filename)
    # Select function according to input
    if F == 1:
        size =int(input().rstrip())
        w=list(map(int, input().split()))
        
        result = filtering_1d(img, np.asarray(w), size)
    elif F == 2:
        size = int(input().rstrip())
        filter = []
        for i in range (size):
            filter.append(list(map(int, input().split())))
    
        result = filtering_2d(img,  np.asarray(filter))
    elif F == 3:    
        size = int(input().rstrip())
        result = median_filter(img, size)

    error = RMSE(result, img)
    print("{:.4f}".format(error))


if __name__ == "__main__":
	main()