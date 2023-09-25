#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray: 
    
    # If an image has only 2 dimensions (Grey Scale image)
    # Convert it into 3 dimensions so it has a shape of (*, *, 1)
    no_dimensions = len(image.shape)
    if no_dimensions == 2:
        image = np.expand_dims(image, axis = 2)
    
    # Checking the number of rows and columns that need to be padded 
    # This will ensure that the edges are captured as well
    # Assuming the Stride = 1 by default
    new_height = (image.shape[0] + kernel.shape[0]) - 1
    new_width = (image.shape[1] + kernel.shape[1]) - 1
    padded_image = np.zeros((new_height, new_width, image.shape[2]))
    
    # We will pad (say 3) rows on top of the image and bottom of the image
    # We will pad (say 3) columns to the left and right of the image
    no_rows_to_pad = int((new_height - image.shape[0]) / 2)
    no_cols_to_pad = int((new_width - image.shape[1]) / 2)
    
    # Padding the image to ensure that the borders are included in the Convolved image
    # Adding the zeros to the image along the rows (at the top and bottom) and columns (Left and Right)
    x1 = 0 + no_rows_to_pad
    x2 = (padded_image.shape[0]) - no_rows_to_pad
    y1 = 0 + no_cols_to_pad
    y2 = (padded_image.shape[1]) - no_cols_to_pad
    padded_image[x1:x2,y1:y2,:] = image[:,:,:]
    
    # Convolution
    convolved_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    for i in range(0, ((padded_image.shape[0] - kernel.shape[0]) + 1)):
         for j in range(0, ((padded_image.shape[1] - kernel.shape[1]) + 1)):
                 for k in range(0, padded_image.shape[2]):
                        window_image = padded_image[i : (i + kernel.shape[0]), j : (j + kernel.shape[1]), k]
                        convolved_image[i, j, k] = np.sum(window_image * kernel)
            
    return convolved_image

