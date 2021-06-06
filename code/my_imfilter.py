# -*- coding: utf-8 -*-
"""
% This function that you will implement is intended to behave like the built-in function 
% imfilter() in Matlab or equivalently the same function implemented as part of scipy.misc module
% in Python. You will implement imfilter from first principles, i.e., without using 
% any library functions. 

% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, we will essentially perform 2D correlation 
% between the filter and image. Referring to 'proj1_test_filtering.py' would help you with
% your implementation. 
  
% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' in Matlab, you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. A better
% approach would be to mirror the image content over the boundaries for padding.

% % Uncomment if you want to simply call library imfilter so you can see the desired
% % behavior. When you write your actual solution, **you can't use imfilter,
% % correlate, convolve commands, but implement the same using matrix manipulations**. 
% % Simply loop over all the pixels and do the actual
% % computation. It might be slow.
"""

import numpy as np

""" Exemplar Gaussian 3x3 filter shown below-- see filters defined in proj1_test_filtering.py """
# filter = np.asarray(dtype=np.float32, a=[[0.1019,0.1154,0.1019],[0.1154,0.1308,0.1154],[0.1019,0.1154,0.1019]]) 

# Defining the function my_imfilter
def my_imfilter(image,my_filter):
	"""
	Input : 
		image: A 3d image array representing the input image
		my_filter: Filter to be applied to the image
	
	Output : 
		filtered_image
  	"""

	# import scipy.ndimage as ndimage
	# output = np.zeros_like(image)

	# for ch in range(image.shape[2]):
	# 	output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], filter, mode='constant')
	
	# return output


	# Get the image dimensions
	output = image.copy()
	im_dim = image.shape

	# Get the individual height, width of the image and the filter being used
	flt_dim = my_filter.shape
	img_dim1 = im_dim[0]
	img_dim2 = im_dim[1]

	flt_dim1 = flt_dim[0]
	flt_dim2 = flt_dim[1]

	# Get the padding dimensions
	pad_dim1 = int((flt_dim1-1)/2)
	pad_dim2 = int((flt_dim2-1)/2)

	# Get the final padded matrix

	# Get the dimension of the final padded matrix
	final_shape = (img_dim1 + 2*pad_dim1, img_dim2 + 2*pad_dim2, 3)
	pad_mat = np.zeros(shape=final_shape)

	# Align our image to the centre of the new padded matrix
	pad_mat[pad_dim1: img_dim1 + pad_dim1, pad_dim2 : img_dim2 + pad_dim2] = image

	# print(f"Image shape : {image.shape}")
	# print(len(image[0][0]))
	# print(len(image))
	# print(len(image[0]))

	num_channels = len(image[0][0])
	width = len(image)
	height = len(image[0])

	# Convolute/Correlate

	# Apply filter for each channel of the image
	for d in range(num_channels):

		# Apply to the 2D pixel matrix at channel 'd'
		for i in range(width):
			for j in range(height):

				# Get the dimensions of the image to be filtered and filter it
				output[i][j][d] = sum(sum(np.multiply(my_filter, pad_mat[i:i+flt_dim1, j:j+flt_dim2, d])))

	return output
