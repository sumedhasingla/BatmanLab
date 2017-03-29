"""
Author: Sumedha Singla
This script will go through the pre-processing pipeline for the Kaggle Data Science Bowl 2017 challenge. The script assumes for each of the samples, a partial lung label map has been generated using the CIP (Chest Imaging Platform) command line tool: "GeneratePartialLungLabelMap". Below is the list of Python dependencies to run this program.
"""

import os
import sys
import argparse
import SimpleITK as sitk
import math
import pdb
import numpy as np
from cip_python.input_output.image_reader_writer import ImageReaderWriter
from cip_python.segmentation.grid_segmenter import GridSegmenter

'''
The partial lung label map generated using the commad line tool "GeneratePartialLungLabelMap" assigns multiple labels to various regions of the lung. The below function takes the label map as input and convert it into a binary label map where a label 1 is assign to any region marked as lung and a label of 0 is assigned to background.
'''
def computeBinaryLungLabelMap(fileRoot, fileName):
    imageFilePath = os.path.join(fileRoot, fileName)
    inputMap = sitk.ReadImage(imageFilePath)
    #Binary Threshold Image Filter
    thresholdFilter = sitk.BinaryThresholdImageFilter()
    thresholdFilter.SetLowerThreshold(1)
    thresholdFilter.SetUpperThreshold(1024)
    thresholdFilter.SetOutsideValue(0)
    thresholdFilter.SetInsideValue(1)
    thresholdInputMap = thresholdFilter.Execute(inputMap)

    outputFileName = fileName.split('_')[0] + '_BinaryLungLabelMap.nrrd'
    outputFilePath = os.path.join(fileRoot, outputFileName)
    sitk.WriteImage(thresholdInputMap, outputFilePath)
    return outputFileName

'''
The below function, takes an image volume as input and threshold it such that the intensity values are between lower and upper threshold. All the voxels with value less than or equal to "lower" are assigned a new value equal to lower. All the voxels with value more than equal to "upper" are assigned a new value equal to upper.
'''
def thresholdImageVolumne(fileRoot, inputVolumeName, lower, upper):
    imageFilePath = os.path.join(fileRoot, inputVolumeName)
    inputVolume = sitk.ReadImage(imageFilePath)
    image_io = ImageReaderWriter()
    ct_array = image_io.sitkImage_to_numpy(inputVolume)
    ct_array[ct_array >= upper] = upper
    ct_array[ct_array <= lower] = lower
    metainfo=dict()
    metainfo['space origin']=inputVolume.GetOrigin()
    metainfo['spacing']=inputVolume.GetSpacing()
    metainfo['space directions']=inputVolume.GetDirection()
    outputVolume = image_io.numpy_to_sitkImage(ct_array, metainfo)
    outputFileName = inputVolumeName.split('.')[0] + '_threshold.nrrd'
    outputFilePath = os.path.join(fileRoot, outputFileName)
    image_io.write(outputVolume,outputFilePath)
    return outputFileName

'''
It is still very common to find medical image datasets that have been acquired with large inter-slice spacings that result in voxels with anisotropic shape. A scan may have a pixel spacing of [0.5, 0.5, 2.0], which means that the distance between slices is 2.0 mm. For a different scan this may be [0.725, 0.725, 1.75], this can be problematic for automatic analysis.
Before doing any kind of analysis, we first make the spacing in all the 3 directions same by performing isotropic resampling.
'''    
def convertToIsotropicVolume(fileRoot, fileName, interpolationMethod):
    '''
    Referrence: https://itk.org/Doxygen/html/Examples_2Filtering_2ResampleVolumesToBeIsotropic_8cxx-example.html
    '''
    imageFilePath = os.path.join(fileRoot, fileName)
    inputVolume = sitk.ReadImage(imageFilePath)
    inputSpacing = inputVolume.GetSpacing()
    inputSize = inputVolume.GetSize()
    #Resample the images to make them iso-tropic
    resampleFilter = sitk.ResampleImageFilter()
    T = sitk.Transform()
    T.SetIdentity()
    resampleFilter.SetTransform(T)
    resampleFilter.SetInterpolator(interpolationMethod)
    resampleFilter.SetDefaultPixelValue( 255 );
    isoSpacing = 1 #math.sqrt(inputSpacing[2] * inputSpacing[0])
    resampleFilter.SetOutputSpacing((isoSpacing,isoSpacing,isoSpacing))
    resampleFilter.SetOutputOrigin(inputVolume.GetOrigin())
    resampleFilter.SetOutputDirection(inputVolume.GetDirection())
    dx = int(inputSize[0] * inputSpacing[0] / isoSpacing)
    dy = int(inputSize[1] * inputSpacing[1] / isoSpacing)
    dz = int((inputSize[2] - 1 ) * inputSpacing[2] / isoSpacing)
    resampleFilter.SetSize((dx,dy,dz))
    resampleVolumne = resampleFilter.Execute(inputVolume)
    
    outputFileName = fileName.split('.')[0] + '_isotropic.nrrd'
    outputFilePath = os.path.join(fileRoot, outputFileName)
    sitk.WriteImage(resampleVolumne, outputFilePath)
    return outputFileName

'''
Given a binary mask and an input volume, the below function creates a masked output, where the input volume values are retain in regions where mask is 1. In the regions where mask is 0, the output volume will have value = outside.
'''
def maskImageVolume(fileRoot, inputVolumeName, maskVolumeName, outside):
    imageFilePath = os.path.join(fileRoot, inputVolumeName)
    inputVolume = sitk.ReadImage(imageFilePath)
    imageFilePath = os.path.join(fileRoot, maskVolumeName)
    maskVolume = sitk.ReadImage(imageFilePath)
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetOutsideValue(outside)
    maskedOutputVolume = maskFilter.Execute(inputVolume,maskVolume )
    outputFileName = inputVolumeName.split('.')[0] + '_Masked.nrrd'
    outputFilePath = os.path.join(fileRoot, outputFileName)
    sitk.WriteImage(maskedOutputVolume, outputFilePath)
    return outputFileName

'''
Given an input volume, the below function creats an output volume where each voxel in input volume is assigned to a patch id. The input volume is divided into a grid and each grid cell is assigned a unique patch id.
The size of the grid cell is specified by xSize, ySize and zOffset
'''    
def gridSegmentation(fileRoot, inputVolumeName, xSize=31, ySize=31, zOffset=10):
    imageFilePath = os.path.join(fileRoot, inputVolumeName)
    inputVolume = sitk.ReadImage(imageFilePath)
    image_io = ImageReaderWriter()
    ct_array = image_io.sitkImage_to_numpy(inputVolume)
    grid_segmenter = GridSegmenter(input_dimensions=None, ct=ct_array, x_size=xSize, y_size=ySize, z_offset=zOffset)
    grid_segmentation = grid_segmenter.execute()
    metainfo=dict()
    metainfo['space origin']=inputVolume.GetOrigin()
    metainfo['spacing']=inputVolume.GetSpacing()
    metainfo['space directions']=inputVolume.GetDirection()
    outputVolume = image_io.numpy_to_sitkImage(grid_segmentation, metainfo)
    outputFileName = inputVolumeName.split('.')[0] + '_GridSegmentation_' + str(xSize) + '.nrrd'
    outputFilePath = os.path.join(fileRoot, outputFileName)
    image_io.write(outputVolume,outputFilePath)
    return outputFileName

    
'''
Given an input volume, and a grid label map, this function divides the input volume in to patches. Each patch corresponds to the grid cell with same patch id. We consider only those patches which have useful information, i.e the percentage of volume of the patch which is background is below some threshold. We also consider only those patches that are of same size as defined by the patchSize.
'''
def createPatches(fileRoot, inputVolumeName, gridSegmentationVolume, outside, threshold, patchSize=9610):  #9610 = 31*31 * 10
    image_io = ImageReaderWriter()
    imageFilePath = os.path.join(fileRoot, inputVolumeName)
    ct_array_image,metainfo = image_io.read_in_numpy(imageFilePath)
    
    imageFilePath = os.path.join(fileRoot, gridSegmentationVolume)
    ct_array_grid,metainfo1 = image_io.read_in_numpy(imageFilePath)
    
    minPatchId = int(np.min(ct_array_grid))
    maxPatchId = int(np.max(ct_array_grid))
    all_patches = []
    for i in range(minPatchId, maxPatchId+1):
        index_grid = np.where(ct_array_grid == i)
        patch =  ct_array_image[index_grid]
        #Find the volume of patch that is outside the useful content
        noOfOutsidePixels = len(np.where(patch == outside)[0])
        totalNumberOfPixels = patch.size
        percentage = float(noOfOutsidePixels)/totalNumberOfPixels
        n = patch.size
        if percentage <= threshold and n == patchSize: 
            all_patches.append(patch)
    all_patches = np.array(all_patches)
    print all_patches.shape
    outputFileName = inputVolumeName.split('.')[0] + '_patches.npy'
    outputFilePath = os.path.join(fileRoot, outputFileName)
    np.save(outputFilePath, all_patches)
    
    return outputFileName
   
'''
One of the clinical parameter for evaluating the CT image of the lung, is to consider the percentage of volume of the lung CT with intenisty value less than -910. The below function evaluates the percentage of input volume that has intensity less than or equal to given threshold. Here we only consider those pixels which are part of the lung segmentation.
''' 
def baselineValue(fileRoot, inputVolumeName, threshold, outside):
    image_io = ImageReaderWriter()
    imageFilePath = os.path.join(fileRoot, inputVolumeName)
    ct_array_image, metainfo = image_io.read_in_numpy(imageFilePath)
    noOfOutsidePixels = len(np.where((ct_array_image <= threshold) & (ct_array_image > outside))[0])
    totalNumberOfPixels = len(np.where(ct_array_image > outside)[0])
    percentage = float(noOfOutsidePixels)/totalNumberOfPixels
    outputFileName = inputVolumeName.split('.')[0] + '_ClinicalThreshold.txt'
    outputFilePath = os.path.join(fileRoot, outputFileName)
    line = "Percentage of volume with intensity below the clinical threshold " + str(threshold) + " is: " + str(percentage)
    outfile = open(outputFilePath, 'w')
    outfile.write(line)
    outfile.close()    
     
    
def main():
    #directory = '/pylon2/ms4s88p/jms565/projects/KaggleLungCancer/'
    directory = '/pylon2/ms4s88p/singla/K0026/'
    #Parse through all the files in the training dataset.
    for fileRoot, directories, files in os.walk(directory):
        inputCT = ""
        partialLungLabelMap = ""
        for f in files:  
            if "original_3D" in f:
                inputCT = f
                print inputCT
            if "partialLungLabelMap" in f:
                print f
                partialLungLabelMap = f
        if inputCT != "" and partialLungLabelMap != "":
            # Convert the input CT to isotropic volume with same spacing in all three directions
            isotropicCT = convertToIsotropicVolume(fileRoot, inputCT, sitk.sitkCosineWindowedSinc) 

            # Threshold the isotropic CT to have intensity values between [-1024, 200]
            threshold_isotropicCT = thresholdImageVolumne(fileRoot, isotropicCT, -1024, 200) 

            # Convert the partial lung label map into a binary label map for lung vs not lung.
            binaryLabelMap = computeBinaryLungLabelMap(fileRoot, partialLungLabelMap)

            # Convert the binary label map to be isotropic to match the CT in spacing. 
            isotropic_BinaryLabelMap = convertToIsotropicVolume(fileRoot, binaryLabelMap, sitk.sitkNearestNeighbor)

            # Use the binary label map to maked out the lung region from the chest CT. the outside pixels are set to -1030
            maskedCT = maskImageVolume(fileRoot, threshold_isotropicCT, isotropic_BinaryLabelMap, -1030) 

            # Calculate the precentage of volume of masked lung that have intensity value below threshold: -910
            baselineValue(fileRoot, maskedCT, -910, -1030)

            # Divide the masked lung region of the CT into grids, each grid cell have a unique patch id.
            gridSegmentedMaskedCT = gridSegmentation(fileRoot, maskedCT)

            #Extract individual patches from the masked Lung CT and create a numpy array.
            #Now each subject is represented in terms of a number of patches of same dimension.
            patches = createPatches(fileRoot, maskedCT, gridSegmentedMaskedCT, -1030, 0.6)
            
            # Divide the masked lung region of the CT into larger grids, each grid cell have a unique patch id.
            gridSegmentedMaskedCTLarge = gridSegmentation(fileRoot, maskedCT, xSize=51, ySize=51, zOffset=15)

            #Extract individual patches from the masked Lung CT and create a numpy array.
            #Now each subject is represented in terms of a number of patches of same dimension.
            patches = createPatches(fileRoot, maskedCT, gridSegmentedMaskedCTLarge, -1030, 0.6, patchSize=51*51*15)
            
            
if __name__ == "__main__":
    main()