#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math 
import numpy as np  

from MyConvolution import convolve 

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
     
    # Low Pass Image
    low_pass_filter = makeGaussianKernel(lowSigma)
    low_pass_Image = convolve(lowImage, low_pass_filter)
    
    # High Pass Image
    high_pass_filter = makeGaussianKernel(highSigma)
    low_pass_Image2 = convolve(highImage, high_pass_filter)
    high_pass_Image = low_pass_Image2 - highImage
    
    # Hybrid Image:
    hybrid_image = low_pass_Image + high_pass_Image
    
    return hybrid_image


def makeGaussianKernel(sigma: float) -> np.ndarray: 
    
    kernel_size = math.floor(8 * sigma + 1)
    if (kernel_size % 2) == 0:
        kernel_size += 1
    
    kernel_sum = 0
    kernel_centre = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Gaussian Filter
    for i in range(kernel_size):
        for j in range(kernel_size):
            term1 = math.exp(-(abs(i - kernel_centre) ** 2 + abs(j - kernel_centre) ** 2) / (2 * (sigma ** 2)))
            term2 = 1 / (2 * math.pi * (sigma ** 2))
            kernel[i][j] = term1 * term2
            kernel_sum += kernel[i][j]
    
    # Flipping the Kernel
    kernel_flipped_x = np.zeros((kernel_size, kernel_size))
    kernel_flipped = np.zeros((kernel_size, kernel_size))
    
    # Flipping the Kernel along the x-axis
    # When flipped along the x-axis, the row values will change and the column values will remain the same
    kernel_flipped_x[:,:] = kernel[::-1,:]
    
    # Flipping the Kernel along the y-axis
    # When flipped along the y-axis, the column values will change and the row values will remain the same
    kernel_flipped[:,:] = kernel_flipped_x[:, ::-1]
    
    return kernel_flipped

