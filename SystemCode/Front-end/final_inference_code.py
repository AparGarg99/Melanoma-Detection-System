import numpy as np
import tensorflow as tf
import os
import sys
import cv2
import skimage.io as io
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Input,Dropout,Conv2D,BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose,concatenate,MaxPooling2D,Activation,Flatten,Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD,Adam
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from collections import Counter
from glob import glob
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from scipy import ndimage
import math
import imageio
from PIL import Image
from skimage.io import imread,imshow
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import resize
"""# Define UNET Model """

#define orginal image shape
ORG_IMG_WIDTH = 600
ORG_IMG_HEIGHT = 450
#define UNET input shape
IMG_HEIGHT = 192
IMG_WIDTH = 256


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    conv_final = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    outputs = Reshape((IMG_HEIGHT,IMG_WIDTH))(conv_final)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

"""# Load UNET Model"""

def load_Unet_model(oModelPath):
    InputImg = Input((IMG_HEIGHT,IMG_WIDTH, 3), name='img')
    UnetModel= get_unet(InputImg, n_filters=32, dropout=0.3, batchnorm=True)
    UnetModel.compile(optimizer=Adam(), loss="binary_crossentropy")
    UnetModel.load_weights(oModelPath)
    return UnetModel

"""# Perform image segmentation to obtain mask"""

#Using the segmentation model to obtain seg mask array from image array

def pre_process(image):
    oResize = cv2.resize(image,(IMG_WIDTH,IMG_HEIGHT))
    return oResize
def enhance(img):
    sub = img.flatten()
    count = 0
    threshold = 0.5
    total = 0
    for i in range(len(sub)):
        total = total + sub[i]
        if sub[i] > threshold:
            count = count + 1
    ### Added this part to handle light coloured lesions that are not segmented with threshold 0.5 ###
    if count == 0:
        threshold = 1.6 * total/len(sub)
    for i in range(len(sub)):
        if sub[i] > threshold:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub
def post_process_mask(oMask):
    #perform closing
    kernel = np.ones((5, 5), 'uint8')
    oClosedMask = cv2.dilate(oMask, kernel, iterations=2)
    oClosedMask = ndimage.binary_fill_holes(oClosedMask, structure=np.ones((5,5)))
    return oClosedMask
def perform_segmentation(model,image):
    oMask = model.predict(image.reshape(1,IMG_HEIGHT,IMG_WIDTH,3))
    #threshold the mask to make it either 1 or 0
    oEnhancedBinaryMask = enhance(oMask)
    #enlarge binary mask to orginal image size
    oEnhancedBinaryMask = oEnhancedBinaryMask.reshape(IMG_HEIGHT,IMG_WIDTH)
    oEnhancedBinaryMask = cv2.resize(oEnhancedBinaryMask, (ORG_IMG_WIDTH,ORG_IMG_HEIGHT), interpolation = cv2.INTER_AREA)
    return oEnhancedBinaryMask

"""# Set path, names of base models and load base models"""

#set path to model folder
def load_all_models(model_path,model_segmentation_name,model_arbitrator_name):
    #model_path = 'D:\\PRS_project\Model\\final_models\\'
    #input name of model to be tested
    #model_segmentation_name = 'unet_large_jaccard_with3Augmentations.hdf5'
    #model_arbitrator_name = 'Final_Arbitrator_6lyr_binaryCross_Weight5_YesAug_0.5Dropout.hdf5'
    UnetModel = load_Unet_model(os.path.join(model_path,model_segmentation_name))
    model_arbitrator =load_model(os.path.join(model_path,model_arbitrator_name))
    return UnetModel,model_arbitrator

"""# ASYMMETRY PREPROCESSING FUNCTION"""

#this function gets the binary perimeter outline, used in both asymmetry and border preprocessing
def get_perimeter(seg, seg_border, width, height):
#getting the image perimeter from the segmentation mask
    
    # getting border pixels in the left to right direction
    for i in range(height):
        for j in range(width - 1):
            try:
                if seg[i][j] == 0 and seg[i][j + 1] == 1:
                    seg_border[i][j] = 1
                elif seg[i][j] == 1 and seg[i][j + 1] == 0:
                    seg_border[i][j + 1] = 1
            except:
                #print('Pixel out of range')
                pass
                
    # getting border pixels in the right to left direction
    for i in range(height):
        for j in range(width - 1):
            try:
                if seg[i][width - j - 1] == 0 and seg[i][width - j - 2] == 1:
                    seg_border[i][width - j - 1] = 1
                elif seg[i][width - j - 1] == 1 and seg[i][width - j - 2] == 0:
                    seg_border[i][width - j - 2] = 1
            except:
                #print('Pixel out of range')
                pass

    # getting border pixels in the up to down direction
    for j in range(width):
        for i in range(height - 1):
            try:
                if seg[i][j] == 0 and seg[i + 1][j] == 1:
                    seg_border[i][j] = 1
                elif seg[i][j] == 1 and seg[i + 1][j] == 0:
                    seg_border[i + 1][j] = 1
            except:
                #print('Pixel out of range')
                pass

    # getting border pixels in the down to up direction
    for j in range(width):
        for i in range(height - 1):
            try:
                if seg[height - i - 1][j] == 0 and seg[height - i - 2][j] == 1:
                    seg_border[height - i - 1][j] = 1
                elif seg[height - i - 1][j] == 1 and seg[height - i - 2][j] == 0:
                    seg_border[height - i - 2][j] = 1
            except:
                #print('Pixel out of range')
                pass
                
    return seg_border

#get the image perimeter pixel values as a list with arrays of length 3
def get_px_values(img, perimeter):
    #this is a list that will contain arrays of length 3
    perimeter_px_values = []
    
    height = img.shape[0]
    width = img.shape[1]
    
    for i in range(height):
        for j in range(width):
            if perimeter[i][j] == 1:
                perimeter_px_values.append(img[i][j])
    
    return perimeter_px_values

#calculate the average perimeter pixel value
def avg_pixel_values(perimeter_px_values):
    length = len(perimeter_px_values)
    sum_red = 0
    sum_green = 0
    sum_blue = 0
    
    for i in perimeter_px_values:
        sum_red = sum_red + i[0]
        sum_green = sum_green + i[1]
        sum_blue = sum_blue + i[2]
    return sum_red/length, sum_green/length, sum_blue/length

#step 1 of the asymmetry preprocessing: replace non-lesion pixels(outside the seg mask) with the avg perimeter colour
def asym_step_one(img, seg, avg_red, avg_green, avg_blue):
    step_1_img = np.empty(img.shape, dtype=np.uint8)
    
    height = img.shape[0]
    width = img.shape[1]
    
    for i in range(height):
        for j in range(width):
            if seg[i][j] == 0:
                step_1_img[i][j][0] = avg_red
                step_1_img[i][j][1] = avg_green
                step_1_img[i][j][2] = avg_blue
            else:
                step_1_img[i][j][:] = img[i][j][:]
          
    return step_1_img

#gets the eqn parameters of major and minor axes of segmentation mask, using tangent and orientation angle(in radian)
def get_axes_eqns(orientation, centroid_0, centroid_1):
    
    gradient_mj = math.tan((math.pi/2)-orientation)
    intercept_mj = centroid_0 - gradient_mj * centroid_1
    
    gradient_mn = 1/(math.tan((math.pi/2)+orientation))
    intercept_mn = centroid_0 - gradient_mn * centroid_1
    
    return gradient_mj, gradient_mn, intercept_mj, intercept_mn

#gets the pixel difference of the reflection over both major and minor axes
def get_px_diff(img, gradient_mj, intercept_mj, gradient_mn, intercept_mn):
        
    height = img.shape[0]
    width = img.shape[1]
    
    #create empty 3d arrays to store the difference values
    diff_arr_mj = np.empty(img.shape, dtype=np.uint8)
    diff_arr_mn = np.empty(img.shape, dtype=np.uint8)
    diff_arr_mjmn = np.empty(img.shape, dtype=np.uint8)
    
    #looping through each pixel in the image 
    for i in range(height):
        for j in range(width):
            #in case the gradient of the axis is zero
            if gradient_mj !=0 and gradient_mn !=0:
                x_dist_from_mj_axis = int(j - ((i-intercept_mj)/gradient_mj))
                y_dist_from_mj_axis = int(i - (gradient_mj * j + intercept_mj)) 
                x_dist_from_mn_axis = int(j - ((i-intercept_mn)/gradient_mn))
                y_dist_from_mn_axis = int(i - (gradient_mn * j + intercept_mn))
                reflected_y_dist_from_mj_axis = x_dist_from_mj_axis
                reflected_x_dist_from_mj_axis = y_dist_from_mj_axis
                reflected_y_dist_from_mn_axis = x_dist_from_mn_axis
                reflected_x_dist_from_mn_axis = y_dist_from_mn_axis
                reflected_x_coordinate_mj = j + reflected_x_dist_from_mj_axis
                reflected_y_coordinate_mj = i + reflected_y_dist_from_mj_axis
                reflected_x_coordinate_mn = j + reflected_x_dist_from_mn_axis
                reflected_y_coordinate_mn = i + reflected_y_dist_from_mn_axis            
                reflected_x_coordinate_mjmn = j + reflected_x_dist_from_mj_axis + reflected_x_dist_from_mn_axis
                reflected_y_coordinate_mjmn = i + reflected_y_dist_from_mj_axis + reflected_y_dist_from_mn_axis
            elif gradient_mj == 0:
                reflected_x_coordinate_mj = j
                y_dist_from_mj_axis = int(i - intercept_mj)
                reflected_y_coordinate_mj = i - 2*y_dist_from_mj_axis
                reflected_y_coordinate_mn = i
                x_dist_from_mn_axis = int(j - intercept_mn)
                reflected_x_coordinate_mn = j - 2*x_dist_from_mn_axis
                reflected_x_coordinate_mjmn = reflected_x_coordinate_mn
                reflected_y_coordinate_mjmn = reflected_y_coordinate_mj
            #else if gradient_mn==0
            else:
                reflected_x_coordinate_mn = j
                y_dist_from_mn_axis = int(i - intercept_mn)
                reflected_y_coordinate_mn = i - 2*y_dist_from_mn_axis
                reflected_y_coordinate_mj = i
                x_dist_from_mj_axis = int(j - intercept_mj)
                reflected_x_coordinate_mj = j - 2*x_dist_from_mj_axis
                reflected_x_coordinate_mjmn = reflected_x_coordinate_mj
                reflected_y_coordinate_mjmn = reflected_y_coordinate_mn
            
            if 0 <= reflected_x_coordinate_mj < width and 0 <= reflected_y_coordinate_mj < height:
                diff_arr_mj[i][j][:] = img[i][j][:] - img[reflected_y_coordinate_mj][reflected_x_coordinate_mj][:]
            else:
                diff_arr_mj[i][j][:] = 0
                
            if 0 <= reflected_x_coordinate_mn < width and 0 <= reflected_y_coordinate_mn < height:
                diff_arr_mn[i][j][:] = img[i][j][:] - img[reflected_y_coordinate_mn][reflected_x_coordinate_mn][:]
            else:
                diff_arr_mn[i][j][:] = 0     
                
            if 0 <= reflected_x_coordinate_mjmn < width and 0 <= reflected_y_coordinate_mjmn < height:
                diff_arr_mjmn[i][j][:] = img[i][j][:] - img[reflected_y_coordinate_mjmn][reflected_x_coordinate_mjmn][:]
            else:
                diff_arr_mjmn[i][j][:] = 0
                
    return diff_arr_mj, diff_arr_mn, diff_arr_mjmn

def get_asym_img(img, seg):

    #get the perimeter outline of the lesion in a 2d binary array
    height = seg.shape[0]
    width = seg.shape[1]
    
    seg_border = np.empty(seg.shape, dtype=np.uint8)
    perimeter = get_perimeter(seg, seg_border, width, height)

    #perimeter_px_values is a list containing arrays of length 3
    #use perimeter outline to get the perimeter_px_values from original image
    perimeter_px_values = get_px_values(img, perimeter)
    
    if len(perimeter_px_values) > 0:
    
        #get the average value of perimeter pixels
        avg_red, avg_green, avg_blue = avg_pixel_values(perimeter_px_values)

        #replace the non-lesion pixels with the average RGB values of the lesion perimeter
        step_1_img = asym_step_one(img, seg, avg_red, avg_green, avg_blue)
    
    else:
        step_1_img = img
    
    #using skimage regionprops to get orientation, centroid_0, centroid_1, to calculate the axes eqn parameters
    label_seg = label(seg)
    props = regionprops(label_seg)
    gradient_mj, gradient_mn, intercept_mj, intercept_mn = get_axes_eqns(props[0].orientation, 
                                                                         props[0].centroid[0], 
                                                                         props[0].centroid[1])
    
    #get the 3 set of difference values
    diff_mj, diff_mn, diff_mjmn = get_px_diff(step_1_img, gradient_mj, intercept_mj, gradient_mn, intercept_mn)

    #calculate the average for each channel in each pixel
    asym_img = np.empty(img.shape,dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            asym_img[i][j][:] = (diff_mj[i][j][:] + diff_mn[i][j][:] + diff_mjmn[i][j][:])/3
    
    return asym_img

"""# BORDER PREPROCESSING FUNCTION"""

def get_outward_pixels(img, img_border, width, height):
#get 5 pixels outwards, left right up down direction

    out_px = 5

    #Getting leftward 5 pixels
    for i in range(height):
        for j in range(width - 1):
            try:
                if img[i][j] == 0 and img[i][j + 1] == 1:
                    img_border[i][(j - out_px):j] = 1
            except:
                #print('Pixel out of range')
                pass

    #Getting rightward 5 pixels
    for i in range(height):
        for j in range(width - 1):
            try:
                if img[i][width - j - 1] == 0 and img[i][width - j - 2] == 1:
                    img_border[i][(width-j-1) : (width-j-1+out_px)] = 1
            except:
                #print('Pixel out of range')
                pass

    #Getting upward 5 pixels
    for j in range(width):
        for i in range(height - 1):
            try:
                if img[i][j] == 0 and img[i + 1][j] == 1:
                    for p in range(1, out_px+1):
                        img_border[i-p][j] = 1
            except:
                #print('Pixel out of range')
                pass
                
    #Getting downward 5 pixels
    for j in range(width):
        for i in range(height - 1):
            try:
                if img[height - i - 1][j] == 0 and img[height - i - 2][j] == 1:
                    for p in range(out_px):
                        img_border[height-i-1+p][j] = 1  
            except:
                #print('Pixel out of range')
                pass
                
    return(img_border)

def get_inward_pixels(img, img_border, width, height):
#get 20 pixels inwards, right left down up direction

    inPixels = 20

    #Getting rightward 20 pixels
    for i in range(height):
        for j in range(width - 1):
            try:
                if img[i][j] == 0 and img[i][j + 1] == 1:
                    img_border[i][j+1:j+inPixels+1] = 1
            except:
                #print('Pixel out of range')
                pass

    #Getting leftward 20 pixels
    for i in range(height):
        for j in range(width - 1):
            try:
                if img[i][width - j - 1] == 0 and img[i][width - j - 2] == 1:
                    img_border[i][(width-j-1-inPixels) : (width-j-1)] = 1
            except:
                #print('Pixel out of range')
                pass

    #Getting downward 20 pixels
    for j in range(width):
        for i in range(height - 1):
            try:
                if img[i][j] == 0 and img[i + 1][j] == 1:
                    for p in range(i+1, i+1+inPixels):
                        img_border[p][j] = 1
            except:
                #print('Pixel out of range')
                pass
                
    #Getting upward 20 pixels
    for j in range(width):
        for i in range(height - 1):
            try:
                if img[height - i - 1][j] == 0 and img[height - i - 2][j] == 1:
                    for p in range(inPixels):
                        img_border[height-i-2-p][j] = 1
            except:
                #print('Pixel out of range')
                pass
            
    return img_border

def create_border_mask(img,seg):

    width = img.shape[1]
    height = img.shape[0]

    # img_border will store the border imformation from the mask
    seg_border = np.empty(seg.shape, dtype=int)
    horizontal = seg.shape[1]
    vertical = seg.shape[0]
    
    seg_border = get_perimeter(seg, seg_border, width, height)
    seg_border = get_outward_pixels(seg, seg_border, width, height)
    seg_border = get_inward_pixels(seg, seg_border, width, height)
    
    return seg_border

def crop_border(img, img_crop, mask):
    height = img.shape[0]
    width = img.shape[1]
    
    for i in range(height):
        for j in range(width):
            if int(mask[i][j]) != 0:
                #to make the RGB channels correspond properly as original is loaded in BGR
                img_crop[i][j][:] = img[i][j][:]
            else:
                img_crop[i][j][:] = 0
    return img_crop

def get_border_img(img, seg):
    
    #create the border mask
    seg_border = create_border_mask(img,seg)
    
    #create am empty nd array with same shape as img to store the cropped img
    img_border = np.empty(img.shape, dtype=np.uint8)
    
    #apply the border cropping
    img_border = crop_border(img, img_border, seg_border)

    return img_border

"""# CENTER PREPROCESSING FUNCTION"""

#function to get the center coordinates of a single segmentation mask
def get_center_coordinates(seg):
    
    label_seg = label(seg)
    props = regionprops_table(label_seg, properties=('centroid',))

    y_max = seg.shape[0]
    x_max = seg.shape[1]

    y_range = np.array([props['centroid-0'][0]-112,props['centroid-0'][0]+112])
    x_range = np.array([props['centroid-1'][0]-112,props['centroid-1'][0]+112])

    #if y_range is outside of the image, shift it back in
    if y_range[0] < 0:
        y_range = y_range - y_range[0]
    if y_range[1] > y_max:
        y_range = y_range - (y_range[1] - y_max)

    #if x_range is outside of the image, shift it back in
    if x_range[0] < 0:
        x_range = x_range - x_range[0]
    if x_range[1] > x_max:
        x_range = x_range - (x_range[1] - x_max)

    #join y_range and x_range into a 2D array, which will be stored in a list for all images in the folder
    center_px_range = np.array([y_range,x_range])
    
    return center_px_range

#function to crop the center 256x256 pixels of a single image and save it to the output folder
def center_crop(img, center_coordinate_range):

    left = int(center_coordinate_range[1][0])
    top = int(center_coordinate_range[0][0])
    right = int(center_coordinate_range[1][1])
    bottom = int(center_coordinate_range[0][1])

    # Crop image of above dimension (It will not change original image)
    img_cropped = img[top:bottom, left:right]

    return img_cropped

def get_center_img(img, seg):
    
    center_coordinate_range = get_center_coordinates(seg)
    
    #crop the center 256x256 pixels
    center_cropped_img = center_crop(img, center_coordinate_range)
    
    return center_cropped_img

"""# IMAGE RESIZE FUNCTION"""

def image_resize(img):
    image_resized = resize(img, (224,224,3),preserve_range=True, anti_aliasing=False).astype('uint8')
    return image_resized

"""# Preprocessing image(s) to be fed into Asymmetry, Border, Center, Whole models"""

def perform_all_preprocessing(img,UnetModel):
    oErrorFlag = 0
    asym_img= []
    border_img = []
    center_img = []
    whole_img = []
    #get the segmentation mask of the lesion image using the segmentation model
    #preprocess input image 
    oResizedImage = pre_process(img)
    oBinaryMask = perform_segmentation(UnetModel,oResizedImage)
    # post processing the predicted mask
    oBinaryMask = post_process_mask(oBinaryMask)
    #converting binary mask to 1-0 array
    seg = np.empty(oBinaryMask.shape, dtype=np.uint8)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if oBinaryMask[i][j] == True:
                seg[i][j] = 1
            else:
                seg[i][j] = 0
    
    #if the lesion contrast is too little and segmentation did not occur (even with using modified threshold), we do not perform prediction on that image
    if 1 not in seg:
        oErrorFlag = 1
        return oErrorFlag,asym_img,border_img,center_img,whole_img
        
    else:       
        #perform ABC preprocessing and resize
        asym_img = image_resize(get_asym_img(img, seg))
        border_img = image_resize(get_border_img(img, seg))
        center_img = image_resize(get_center_img(img, seg))
        whole_img = image_resize(img)
        #convert lists to arrays for feeding into corresponding model    
        asym_img = np.array(asym_img)
        border_img = np.array(border_img)
        center_img = np.array(center_img)
        whole_img = np.array(whole_img)
        #reshape it to (1,image size)
        asym_img = np.reshape(asym_img,(1,asym_img.shape[0],asym_img.shape[1],asym_img.shape[2]))
        border_img = np.reshape(border_img,(1,border_img.shape[0],border_img.shape[1],border_img.shape[2]))
        center_img = np.reshape(center_img,(1,center_img.shape[0],center_img.shape[1],center_img.shape[2]))
        whole_img = np.reshape(whole_img,(1,whole_img.shape[0],whole_img.shape[1],whole_img.shape[2]))
        
        return oErrorFlag,asym_img,border_img,center_img,whole_img
