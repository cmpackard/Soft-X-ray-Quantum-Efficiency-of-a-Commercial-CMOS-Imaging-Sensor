# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:13:18 2022

@author: colin
"""
import sys
import os
import time
import math as math
from datetime import datetime
import numpy as np
import astropy as ast
from astropy.io import fits
from astropy.table import Table, hstack, vstack, Column
from astropy.stats import sigma_clipped_stats
import photutils as Phot
from photutils.detection import DAOStarFinder
from photutils.aperture import EllipticalAnnulus, aperture_photometry
#from photutils.detection import detect_threshold
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
#from photutils.segmentation import detect_threshold, source_properties, detect_sources, SourceCatalog
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate
import scipy.constants
import scipy.special
from scipy import integrate
from scipy.optimize import curve_fit
import scipy.interpolate
from scipy.interpolate import CubicSpline
from scipy.stats import norm


import astropy.io.fits as pyfits
import glob2

from tqdm.notebook import tqdm as tqdm

import CU as cu


import gc
from astropy.coordinates import Angle
from astropy.time import Time


from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable







def MakeFileNamesList(datadirectory,filepattern):
    FileNamesList = glob2.glob(datadirectory + filepattern)
    
    return FileNamesList

def MakeImagesList(FileNamesList,IndexOfFitsImage):
    
    ImagesList = []
    
    for file in FileNamesList:
        
        #File = directory + file
        
        h = pyfits.open(file)
        #img_data = h[0].data
        #header = h[0].header
        img_data = h[IndexOfFitsImage].data
        header = h[IndexOfFitsImage].header
        #because steve saves his fits files wrong
        '''
        exposure = header['EXPOSURE']
        gain = header['GAIN']
        DateAndTime = header['DATETIME']
        CamID = header['CAM_ID']
        CamFreeName = header['CAMFREE']
        '''
        image_and_metadata = [0,0,0,0,0,0]
        
        image_and_metadata[0] = img_data
        image_and_metadata[1] = header['EXPOSURE']
        image_and_metadata[2] = header['GAIN']
        image_and_metadata[3] = header['DATE']
        image_and_metadata[4] = header['TEMP']
        image_and_metadata[5] = file
        #image_and_metadata[4] = header['CAM_ID']
        #image_and_metadata[5] = header['CAMFREE']
        
        ImagesList.append(image_and_metadata)
        
    return ImagesList

def MakeMasterDarkFrames(datadirectory,filepattern,DimX,DimY,hotsig,hotfrac,SaveDirectory,SaveName,SaveHotPix = True, SaveAsFits = True,SaveAsCSV = True):
    #DimY,DimX the dimensions of the images eg for 1080x1920 DimY=1080, DimX=1920
    
    #This Currently doesn't make a median dark frame because I havent figured out a way of doing that 
    #without making this code take roughly one-million times longer to run.
    
    #FileNameList = MakeFileNamesList(datadirectory,filepattern)
    FileNameList = glob2.glob(datadirectory + filepattern)
    ImagesList = MakeImagesList(FileNameList,1)

    NumberOfFrames = len(ImagesList)
    SumDark = np.zeros((DimY,DimX))
    SumDarkSquared = np.zeros((DimY,DimX))
    SumExposureTime = 0
    
    HotCounterArray = np.zeros((DimY,DimX))
    ZerosArray = np.zeros((DimY,DimX))
    #SumDark = np.zeros((1080,1920))
    
    
    #making master dark frame by finding the mean
    for i in tqdm(range(len(ImagesList))):
    #for all images in list of images
        
        #For Mean and Std Dark Frames
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        SumDark = SumDark + image_data
        SumDarkSquared = SumDarkSquared + image_data**2
        
        SumExposureTime = SumExposureTime + float(image_and_metadata[1])
        
        #For finding Hot Pixels
        FrameMedian = np.median(image_data)
        FrameStd = np.std(image_data)
        
        #hotsig is the number of stdevs above median a pixel must be to be considered hot.
        HotThresh = FrameMedian + hotsig*FrameStd
        HotsToCount = np.where(image_data > HotThresh,1,ZerosArray)
        
        HotCounterArray = HotCounterArray + HotsToCount


    MeanDarkFrame = SumDark/len(ImagesList)
    StdDarkFrame = np.sqrt((SumDarkSquared/len(ImagesList)) - MeanDarkFrame**2)
    HotsArray = np.where((HotCounterArray/len(ImagesList)) > hotfrac,1,ZerosArray)
    
    MeanExposureTime = SumExposureTime/len(ImagesList)
    
    #You could use HotsArray to zero out the hot pixels as follows
    #ImageDataArray = np.where(HotsArray > 0,0,ImageDataArray)
    
    if SaveHotPix == True:
        exposure = MeanExposureTime
        ExposureTime = int(np.rint(exposure))
        np.savetxt(SaveDirectory +'hotpixels_'+str(ExposureTime)+'.csv', HotsArray, delimiter=',')
        #can be read by HotsArray = np.loadtxt('SaveDirectory +'hotpixels.csv', delimiter=',')
        

    if SaveAsFits == True:
        exposure = MeanExposureTime
        gain = image_and_metadata[2]
        ExposureTime = int(np.rint(exposure))
        
        
        #Saving Mean Dark Frame
        hdu = fits.PrimaryHDU(MeanDarkFrame)
        header = hdu.header
        header['EXPOSURE'] = (exposure, "Mean Exposure Time [ms]")
        header['GAIN'] = (gain, "[dB]")
        header['NUMFRAME'] = (NumberOfFrames, "Num Frames [#]")
        hdu.writeto(SaveDirectory + SaveName + str(ExposureTime)+'.fits')
        
        #Saving Std Dark Frame
        hdu = fits.PrimaryHDU(StdDarkFrame)
        header = hdu.header
        header['EXPOSURE'] = (exposure, "Mean Exposure Time [ms]")
        header['GAIN'] = (gain, "[dB]")
        header['NUMFRAME'] = (NumberOfFrames, "Num Frames [#]")
        hdu.writeto(SaveDirectory + 'StdDarkFrame_' + str(ExposureTime)+'.fits')
        
    if SaveAsCSV == True:
        exposure = MeanExposureTime
        ExposureTime = int(np.rint(exposure))
        
        np.savetxt(SaveDirectory + SaveName + str(ExposureTime) +'.csv', MeanDarkFrame, delimiter=',')
        np.savetxt(SaveDirectory + 'StdDarkFrame_' + str(ExposureTime) +'.csv', StdDarkFrame, delimiter=',')


    return MeanDarkFrame

def ReturnMasterDarkAndHotsArray(datadirectory,filepattern,DimX,DimY,hotsig,hotfrac):
    #DimY,DimX the dimensions of the images eg for 1080x1920 DimY=1080, DimX=1920
    
    #This Currently doesn't make a median dark frame because I havent figured out a way of doing that 
    #without making this code take roughly one-million times longer to run.
    
    #FileNameList = MakeFileNamesList(datadirectory,filepattern)
    FileNameList = glob2.glob(datadirectory + filepattern)
    ImagesList = MakeImagesList(FileNameList,1)

    NumberOfFrames = len(ImagesList)
    SumDark = np.zeros((DimY,DimX))
    SumDarkSquared = np.zeros((DimY,DimX))
    SumExposureTime = 0
    
    HotCounterArray = np.zeros((DimY,DimX))
    ZerosArray = np.zeros((DimY,DimX))
    #SumDark = np.zeros((1080,1920))
    
    
    #making master dark frame by finding the mean
    for i in tqdm(range(len(ImagesList))):
    #for all images in list of images
        
        #For Mean and Std Dark Frames
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        SumDark = SumDark + image_data
        SumDarkSquared = SumDarkSquared + image_data**2
        
        SumExposureTime = SumExposureTime + float(image_and_metadata[1])
        
        #For finding Hot Pixels
        FrameMedian = np.median(image_data)
        FrameStd = np.std(image_data)
        
        #hotsig is the number of stdevs above median a pixel must be to be considered hot.
        HotThresh = FrameMedian + hotsig*FrameStd
        HotsToCount = np.where(image_data > HotThresh,1,ZerosArray)
        
        HotCounterArray = HotCounterArray + HotsToCount


    MeanDarkFrame = SumDark/len(ImagesList)
    StdDarkFrame = np.sqrt((SumDarkSquared/len(ImagesList)) - MeanDarkFrame**2)
    HotsArray = np.where((HotCounterArray/len(ImagesList)) > hotfrac,1,ZerosArray)
    
    MeanExposureTime = SumExposureTime/len(ImagesList)
    
    return MeanDarkFrame, HotsArray

def ReturnMasterDarkAndHotsArrayExtra(datadirectory,filepattern,DimX,DimY,hotsig,hotfrac):
    #DimY,DimX the dimensions of the images eg for 1080x1920 DimY=1080, DimX=1920
    
    #This Currently doesn't make a median dark frame because I havent figured out a way of doing that 
    #without making this code take roughly one-million times longer to run.
    
    #FileNameList = MakeFileNamesList(datadirectory,filepattern)
    FileNameList = glob2.glob(datadirectory + filepattern)
    ImagesList = MakeImagesList(FileNameList,1)

    NumberOfFrames = len(ImagesList)
    SumDark = np.zeros((DimY,DimX))
    SumDarkSquared = np.zeros((DimY,DimX))
    SumExposureTime = 0
    
    HotCounterArray = np.zeros((DimY,DimX))
    ZerosArray = np.zeros((DimY,DimX))
    #SumDark = np.zeros((1080,1920))
    
    FrameMedians = []
    FrameStds = []
    
    #making master dark frame by finding the mean
    for i in tqdm(range(len(ImagesList))):
    #for all images in list of images
        
        #For Mean and Std Dark Frames
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        SumDark = SumDark + image_data
        SumDarkSquared = SumDarkSquared + image_data**2
        
        SumExposureTime = SumExposureTime + float(image_and_metadata[1])
        
        #For finding Hot Pixels
        FrameMedian = np.median(image_data)
        FrameStd = np.std(image_data)
        
        FrameMedians.append(FrameMedian)
        FrameStds.append(FrameStd)
        
        #hotsig is the number of stdevs above median a pixel must be to be considered hot.
        HotThresh = FrameMedian + hotsig*FrameStd
        HotsToCount = np.where(image_data > HotThresh,1,ZerosArray)
        
        HotCounterArray = HotCounterArray + HotsToCount


    MeanDarkFrame = SumDark/len(ImagesList)
    StdDarkFrame = np.sqrt((SumDarkSquared/len(ImagesList)) - MeanDarkFrame**2)
    HotsArray = np.where((HotCounterArray/len(ImagesList)) > hotfrac,1,ZerosArray)
    
    MeanExposureTime = SumExposureTime/len(ImagesList)
    
    FrameMedians = np.array(FrameMedians)
    FrameStds = np.array(FrameStds)
    
    return MeanDarkFrame, HotsArray, FrameMedians, FrameStds

def MakeMasterIlluminatedDarkFrame(ImagesList,MeanDarkFrame,StdDarkFrame,Hotpix,DimX,DimY,IntegrationRegionSize,EventThreshStd):
        #IntegrationRegionSize: the side dimension of the area to integrate around (can only be an odd number to be centered)
        #e.g. if IntegrationRegionSize = 5 integrate the signal over the 5x5 pixels centered around the brightest pixel in an event
        
        #EventThreshStd: the number of standard deviations above the noise level a pixel must be inorder to definitely be considered
        #part of an event (Has no impact on which pixels after and event has been located are included in the event sum)
        
        dx = (IntegrationRegionSize-1)/2
        dy = dx
        
        ZerosArray = np.zeros((DimY,DimX))
        SumDarkArray = np.zeros((DimY,DimX))
        SumDarkSquaredArray = np.zeros((DimY,DimX))
        N_Array = np.zeros((DimY,DimX))
        #N_Array an array where N_Array[y,x] is the number of times the pixel at y,x contributed to the total by not being part of an
        #event integration region.
    
        for i in tqdm(range(len(ImagesList))):
        #for all images in list of images
            #print(i)
            #Getting the Image Array from the ImagesList
            image_and_metadata = ImagesList[i]
            image_data = image_and_metadata[0]
            
            #DarkSubtracting
            dark_sub_image_data = image_data - MeanDarkFrame
            
            dark_sub_image_data_no_hots = np.where(Hotpix > 0, 0, dark_sub_image_data)
            #we dont want hotpixels to count as pixels of interest, otherwise we'd never get a full dark frame because hots appear in all
            #images
            
            PixelsOfInterest = np.where(dark_sub_image_data_no_hots > EventThreshStd*StdDarkFrame, 1, ZerosArray)
            #An array with 1's in every pixel position with a count greater than EventThreshStd stdev dark and  0's elsewhere
            OnlyEventPixels = np.where(dark_sub_image_data_no_hots > EventThreshStd*StdDarkFrame, image_data, ZerosArray)
            #The image array but everything but pixels definitely part of an event (PixelsOfInterest) are zeroed
            
            PixelsCountedTowardsEvents = np.zeros((DimY,DimX))
            
            NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
            
            while NumRemainingPixelsOfInterest > 0:
                
                y,x =  np.unravel_index(np.argmax(OnlyEventPixels, axis=None), OnlyEventPixels.shape)
                
                ximin = x-dx
                ximin = ximin.astype(int)
                ximax = x+dx+1
                ximax = ximax.astype(int)
                yimin = y-dy
                yimin = yimin.astype(int)
                yimax = y+dy+1
                yimax = yimax.astype(int)
                
                #Dealing with edge cases so that we dont have to run the lowest if statement for each integration region pixel
                #in all of the much more numerous non edge cases.
                if yimin <= 0 or yimax >= DimY or ximin <= 0 or ximax >= DimX:
                    for xi in range(ximin, ximax):
                        for yi in range(yimin, yimax):
                            if yi > -1 and xi > -1 and yi < DimY and xi < DimX:
                        
                                PixelsCountedTowardsEvents[yi,xi] = 1
                            
                                PixelsOfInterest[yi,xi] = 0
                                NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                                OnlyEventPixels[yi,xi] = 0
                    continue
                
                for xi in range(ximin, ximax):
                    for yi in range(yimin, yimax):
                        
                        PixelsCountedTowardsEvents[yi,xi] = 1
                            
                        PixelsOfInterest[yi,xi] = 0
                        NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                        OnlyEventPixels[yi,xi] = 0
            
            '''
            #<Generating ToHistogram>------------------------------------------
            #setting every instance of a pixel counted towards an event equal to -100000 to later be removed from histogram
            image_data = np.where(PixelsCountedTowardsEvents > 0, -100000, image_data)
            AppendHistogram = image_data
            AppendHistogram = AppendHistogram.flatten()
            AppendHistogram = AppendHistogram[AppendHistogram != -100000]
            
            if i == 0:
                ToHistogram = AppendHistogram
            else:
                ToHistogram = np.append(ToHistogram,AppendHistogram)
            #</Generating ToHistogram>-----------------------------------------
            '''
            #Setting all the pixels counted towards events to 0 so that they do not contribute to the sum
            image_data = np.where(PixelsCountedTowardsEvents > 0, 0, image_data)
            #removing any remaining hotpixels by setting them to -100000 to later be removed from histogram
            #image_data = np.where(Hotpix > 0, -100000, image_data)
            #Zero the hotpixels in the mean illuminated dark frame at the very end.
            
            SumDarkArray = np.where(PixelsCountedTowardsEvents != 1, SumDarkArray + image_data, SumDarkArray)
            SumDarkSquaredArray = np.where(PixelsCountedTowardsEvents != 1, SumDarkSquaredArray + (image_data**2), SumDarkSquaredArray)
            N_Array = np.where(PixelsCountedTowardsEvents != 1, N_Array+1, N_Array)
            
            #for xi in range(DimX):
                #for yi in range(DimY):
                    
                    #if PixelsCountedTowardsEvents[yi,xi] != 1:
                            
                        #SumDarksArray[yi,xi] = SumDarksArray[yi,xi] + image_data[yi,xi]
                        #N_Array[yi,xi] = NumberOfValsToMeanArray[yi,xi]
                        
        #Making MeanDarkFrame by dividing by the number of times a given pixel contributed to the mean.
        DivideByArray = np.where(N_Array == 0, 1, N_Array)
        #So we don't divide by 0  and 0/1 = 0 so nothing is changed
        MeanDarkFrame = SumDarkArray/DivideByArray
        MeanSquaredDark = SumDarkSquaredArray/DivideByArray
        
        #Zeroing out hot pixels
        MeanDarkFrame = np.where(Hotpix > 0, 0, MeanDarkFrame)
        
        #Calculating "Sample Standard Deviation" Frame
        SampleStdFrame = np.sqrt(MeanSquaredDark - (MeanDarkFrame**2))
        
        #Calculating "Standard Error of Mean" Frame
        StandardErrorOfMeanFrame = SampleStdFrame/np.sqrt(DivideByArray)
        
        #Setting MeanDarkFrame to -100000 where no non-event pixels from any of the images could be found
        MeanDarkFrame = np.where(N_Array == 0, -100000, MeanDarkFrame)
        
        #N_Array[y,x] is the number of times the pixel at [y,x] could be used towards a mean and wasn't part of an event area
        
        return MeanDarkFrame, N_Array, SampleStdFrame, StandardErrorOfMeanFrame
    
def MakeMasterIlluminatedDarkFrameStandard(ImagesList,MeanDarkFrame,Hotpix,DimX,DimY,IntegrationRegionSize,thres):
        #IntegrationRegionSize: the side dimension of the area to integrate around (can only be an odd number to be centered)
        #e.g. if IntegrationRegionSize = 5 integrate the signal over the 5x5 pixels centered around the brightest pixel in an event
        
        #thres: the counts above which a pixel must have inorder to definitely be considered
        #part of an event (Has no impact on which pixels after and event has been located are included in the event sum)
        
        dx = (IntegrationRegionSize-1)/2
        dy = dx
        
        ZerosArray = np.zeros((DimY,DimX))
        SumDarkArray = np.zeros((DimY,DimX))
        SumDarkSquaredArray = np.zeros((DimY,DimX))
        N_Array = np.zeros((DimY,DimX))
        #N_Array an array where N_Array[y,x] is the number of times the pixel at y,x contributed to the total by not being part of an
        #event integration region.
    
        for i in tqdm(range(len(ImagesList))):
        #for all images in list of images
            #print(i)
            #Getting the Image Array from the ImagesList
            image_and_metadata = ImagesList[i]
            image_data = image_and_metadata[0]
            
            #DarkSubtracting
            dark_sub_image_data = image_data - MeanDarkFrame
            
            dark_sub_image_data_no_hots = np.where(Hotpix > 0, 0, dark_sub_image_data)
            #we dont want hotpixels to count as pixels of interest, otherwise we'd never get a full dark frame because hots appear in all
            #images
            
            PixelsOfInterest = np.where(dark_sub_image_data_no_hots > thres, 1, ZerosArray)
            #An array with 1's in every pixel position with a count greater than EventThreshStd stdev dark and  0's elsewhere
            OnlyEventPixels = np.where(dark_sub_image_data_no_hots > thres, image_data, ZerosArray)
            #The image array but everything but pixels definitely part of an event (PixelsOfInterest) are zeroed
            
            PixelsCountedTowardsEvents = np.zeros((DimY,DimX))
            
            NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
            
            while NumRemainingPixelsOfInterest > 0:
                
                y,x =  np.unravel_index(np.argmax(OnlyEventPixels, axis=None), OnlyEventPixels.shape)
                
                ximin = x-dx
                ximin = ximin.astype(int)
                ximax = x+dx+1
                ximax = ximax.astype(int)
                yimin = y-dy
                yimin = yimin.astype(int)
                yimax = y+dy+1
                yimax = yimax.astype(int)
                
                #Dealing with edge cases so that we dont have to run the lowest if statement for each integration region pixel
                #in all of the much more numerous non edge cases.
                if yimin <= 0 or yimax >= DimY or ximin <= 0 or ximax >= DimX:
                    for xi in range(ximin, ximax):
                        for yi in range(yimin, yimax):
                            if yi > -1 and xi > -1 and yi < DimY and xi < DimX:
                        
                                PixelsCountedTowardsEvents[yi,xi] = 1
                            
                                PixelsOfInterest[yi,xi] = 0
                                NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                                OnlyEventPixels[yi,xi] = 0
                    continue
                
                for xi in range(ximin, ximax):
                    for yi in range(yimin, yimax):
                        
                        PixelsCountedTowardsEvents[yi,xi] = 1
                            
                        PixelsOfInterest[yi,xi] = 0
                        NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                        OnlyEventPixels[yi,xi] = 0
            
            #Setting all the pixels counted towards events to 0 so that they do not contribute to the sum
            image_data = np.where(PixelsCountedTowardsEvents > 0, 0, image_data)
            
            SumDarkArray = np.where(PixelsCountedTowardsEvents != 1, SumDarkArray + image_data, SumDarkArray)
            SumDarkSquaredArray = np.where(PixelsCountedTowardsEvents != 1, SumDarkSquaredArray + (image_data**2), SumDarkSquaredArray)
            N_Array = np.where(PixelsCountedTowardsEvents != 1, N_Array+1, N_Array)
                        
        #Making MeanDarkFrame by dividing by the number of times a given pixel contributed to the mean.
        DivideByArray = np.where(N_Array == 0, 1, N_Array)
        #So we don't divide by 0  and 0/1 = 0 so nothing is changed
        OS_MeanDarkFrame = SumDarkArray/DivideByArray
        MeanSquaredDark = SumDarkSquaredArray/DivideByArray
        
        #Zeroing out hot pixels
        OS_MeanDarkFrame = np.where(Hotpix > 0, 0, OS_MeanDarkFrame)
        
        #Calculating "Sample Standard Deviation" Frame
        SampleStdFrame = np.sqrt(MeanSquaredDark - (OS_MeanDarkFrame**2))
        
        #Calculating "Standard Error of Mean" Frame
        StandardErrorOfMeanFrame = SampleStdFrame/np.sqrt(DivideByArray)
        
        #Setting MeanDarkFrame to -100000 where no non-event pixels from any of the images could be found
        OS_MeanDarkFrame = np.where(N_Array == 0, -100000, OS_MeanDarkFrame)
        
        #N_Array[y,x] is the number of times the pixel at [y,x] could be used towards a mean and wasn't part of an event area
        
        return OS_MeanDarkFrame, N_Array, SampleStdFrame, StandardErrorOfMeanFrame, SumDarkArray, DivideByArray

        
def EventFinderPhil(ImagesList,ROI,MeanDarkFrame,StdDarkFrame,Hotpix,DimX,DimY,IntegrationRegionSize,EventThreshStd,KeepEdgeEvents = False):
        #IntegrationRegionSize: the side dimension of the area to integrate around (can only be an odd number to be centered)
        #e.g. if IntegrationRegionSize = 5 integrate the signal over the 5x5 pixels centered around the brightest pixel in an event
        
        #EventThreshStd: the number of standard deviations above the noise level a pixel must be inorder to definitely be considered
        #part of an event (Has no impact on which pixels after and event has been located are included in the event sum)
        
        #MinPixVal: The minimum pixel value to be counted in when integrating over an event region 
        #e.g. MinPixVal = 0 means that any pixel with a value above 0 in the event integration with be counted towards the event signal
        #By getting rid of MinPixVal in Utility 1.5 I've gotten rid of thresp equivalent, and having done so hope to prevent the
        #histogram from skewing positive
        
        
        #ROI = [x1,x2,y1,y2]
        
        x1 = ROI[0]
        x2 = ROI[1]
        y1 = ROI[2]
        y2 = ROI[3]
        
        
        dx = (IntegrationRegionSize-1)/2
        dy = dx
        
        ZerosArray = np.zeros((DimY,DimX))
        
        EventSignals = []
        EventX = []
        EventY = []
    
        for i in range(len(ImagesList)):
        #for all images in list of images
            print(i)
            #Getting the Image Array from the ImagesList
            image_and_metadata = ImagesList[i]
            image_data = image_and_metadata[0]
            
            #DarkSubtracting
            image_data = image_data - MeanDarkFrame
            
            #Zeroing Everything that isn't the region of interest ROI
            image_data[0:y1,:] = 0
            image_data[y2:DimY,:] = 0
            image_data[:,0:x1] = 0
            image_data[:,x2:DimX] = 0
            
            PixelsOfInterest = np.where(image_data > EventThreshStd*StdDarkFrame, 1, ZerosArray)
            #An array with 1's in every pixel position with a count greater than EventThreshStd stdev dark and  0's elsewhere
            OnlyEventPixels = np.where(image_data > EventThreshStd*StdDarkFrame, image_data, ZerosArray)
            #The image array but everything but pixels definitely part of an event (PixelsOfInterest) are zeroed
            
            NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
            
            while NumRemainingPixelsOfInterest > 0:
                
                #print(NumRemainingPixelsOfInterest)
                
                y,x =  np.unravel_index(np.argmax(OnlyEventPixels, axis=None), OnlyEventPixels.shape)
                
                EventSignal = 0
                
                ximin = x-dx
                ximin = ximin.astype(int)
                ximax = x+dx+1
                ximax = ximax.astype(int)
                yimin = y-dy
                yimin = yimin.astype(int)
                yimax = y+dy+1
                yimax = yimax.astype(int)
                
                #Won't count event if it's too close to the edge of the frame
                if yimin <= 0 or yimax >= DimY or ximin <= 0 or ximax >= DimX:
                    
                    for xi in range(ximin, ximax):
                        for yi in range(yimin, yimax):
                            
                            if yi > -1 and xi > -1 and yi < DimY and xi < DimX:
                                
                                PixelsOfInterest[yi,xi] = 0
                                NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                                OnlyEventPixels[yi,xi] = 0
                                
                            if KeepEdgeEvents == True:
                                EventSignals.append(False)
                                EventX.append(x)
                                EventY.append(y)
                        
                    continue
                
                for xi in range(ximin, ximax):
                    for yi in range(yimin, yimax):
                        
                        EventSignal = EventSignal + image_data[yi,xi]
                            
                        if Hotpix[yi,xi] == 1:
                            EventSignal = -100000
                            
                        PixelsOfInterest[yi,xi] = 0
                        NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                        OnlyEventPixels[yi,xi] = 0
                        
                if EventSignal > 0:
                    #We only append to the signals and positions lists after we've confirmed that there were no hotpixels in the sum
                    #i.e only if the Event Signal is positive because we set it to negitive if there were hot pixels.
                    EventSignals.append(EventSignal)
                    EventX.append(x)
                    EventY.append(y)
                
        EventSignals = np.array(EventSignals)
        EventX = np.array(EventX)
        EventY = np.array(EventY)
        EventsArray = np.array([EventSignals,EventX,EventY])
        
        return EventSignals, EventX, EventY, EventsArray
    
def EventFinderSteveA(ImageData, MeanDarkFrame, IntegrationRegionSize, hot_pos, DimX, DimY, thres, thresp, verbose=False, diagonals=True):
    
    
    dx = (IntegrationRegionSize-1)/2
    dy = dx
    
    #hot_pos = []
    #for x in range(DimX):
        #for y in range(DimY):
            #if HotsArray[y,x] > 0:
                #hot_pos.append([y,x])
    
    ep, ex, ey, shp = [], [], [], []
    #nx, ny = 1936, 1096
    nx, ny = DimX, DimY
    dx, dy = 2, 2
    
    hotPixels = hot_pos
    
    phot = ImageData - MeanDarkFrame
    phot[phot < 0] = 0
    for ii in range(0, len(hotPixels)):
        phot[int(hotPixels[int(ii)][0])][int(hotPixels[int(ii)][1])] = 0
    f1 = np.reshape(phot, nx*ny)
    q = np.argsort(f1)[::-1]
    j = 0
    above_thres = True
    while above_thres:
        if (j >= 2121856):
            above_thres = False
            break
        i = q[j]
        j += 1
        if (f1[i] >= thres):
            x = (i % DimX) + 1 #x coordinate in image
            y = math.floor((i / DimX) + 1) #y coordinate in image
            xR = int(math.floor(i/DimX)) #x coordinate in array
            #yR = int(i % 1936) #y coordinate in array
            yR = int(i % DimX) #y coordinate in array
            if (xR > dx) and (xR < ny-dx-1) and (yR > dy) and (yR < nx-dy-1):
                area = phot[(xR-dx):(xR+dx+1), (yR-dy):(yR+dy+1)]
                
                p, s, v = cu.getEp(area, thresp, diag=diagonals)
                for xi in range(xR - dx, xR + dx + 1):
                    for yi in range(yR - dy, yR + dy + 1):
                        if (v[xi-xR-dx][yi-yR-dy]):
                            phot[xi, yi] = 0
                            #phot[yi, xi] = 0
                if (p > 0):
                    ep.append(p)
                    ex.append(x)
                    ey.append(y)
                    shp.append(s)
        else:
            above_thres = False
            
    ep = np.array(ep)
    #ex = np.array(ex)
    #ey = np.array(ey)
    #shp = np.array(shp)
    
    #return[ep,ex,ey,shp]
    return ep

def EventFinderSteveB(ImagesList, ROI, MeanDarkFrame, HotsArray,IntegrationRegionSize, thres, thresp, verbose=False, diagonals=True):
    
    
    CroppedHotsArray = HotsArray[ROI[2]:ROI[3],ROI[0]:ROI[1]]
    CroppedDarkFrame = MeanDarkFrame[ROI[2]:ROI[3],ROI[0]:ROI[1]]
    
    DimX = np.shape(CroppedDarkFrame)[1]
    DimY = np.shape(CroppedDarkFrame)[0]
    
    hot_pos = []
    for x in range(DimX):
        for y in range(DimY):
            if CroppedHotsArray[y,x] > 0:
                hot_pos.append([y,x])
                
    ep = []
    
    FrameMeanEventSignals = []
    AssociatedFileNames = []
    
    for index in tqdm(range(len(ImagesList))):
        
        image_and_metadata = ImagesList[index]
        
        ImageData = image_and_metadata[0]
        ImageData = ImageData[ROI[2]:ROI[3],ROI[0]:ROI[1]]
        FileName = image_and_metadata[5]
        ThisImagesEventSignals = []
        
        AssociatedFileNames.append(FileName)
    
        EventStuff = EventFinderSteveA(ImageData, CroppedDarkFrame, IntegrationRegionSize, hot_pos, DimX, DimY, thres, thresp, verbose=False, diagonals=True)
        
        for i in range(np.size(EventStuff)):
            
            ep.append(EventStuff[i])
            #print(EventStuff[i])
            
            ThisImagesEventSignals.append(EventStuff[i])
            
        ThisImagesEventSignals = np.array(ThisImagesEventSignals)
        FrameMeanEventSignal = np.mean(ThisImagesEventSignals)
        FrameMeanEventSignals.append(FrameMeanEventSignal)
        
    FrameMeanEventSignals = np.array(FrameMeanEventSignals)
    AssociatedFileNames = np.array(AssociatedFileNames)
    AllEventSignals = np.array(ep)
    
    return AllEventSignals, FrameMeanEventSignals, AssociatedFileNames
    
        
        
        

def EventFinderStandard(ImagesList,ROI,MeanDarkFrame,Hotpix,DimX,DimY,IntegrationRegionSize,thres,thresp,KeepEdgeEvents = False):
        #IntegrationRegionSize: the side dimension of the area to integrate around (can only be an odd number to be centered)
        #e.g. if IntegrationRegionSize = 5 integrate the signal over the 5x5 pixels centered around the brightest pixel in an event
        
        #thres: the counts above which a pixel must have inorder to definitely be considered
        #part of an event (Has no impact on which pixels after and event has been located are included in the event sum)
        
        #thresp: (previously MinPixVal) The minimum pixel value to be counted in when integrating over an event region 
        #e.g. thresp = 0 means that any pixel with a value above 0 in the event integration with be counted towards the event signal
        
        #ROI = [x1,x2,y1,y2]
        
        x1 = ROI[0]
        x2 = ROI[1]
        y1 = ROI[2]
        y2 = ROI[3]
        
        
        dx = (IntegrationRegionSize-1)/2
        dy = dx
        
        ZerosArray = np.zeros((DimY,DimX))
        
        EventSignals = []
        EventX = []
        EventY = []
    
        for i in tqdm(range(len(ImagesList))):
        #for all images in list of images
            #print(i)
            #Getting the Image Array from the ImagesList
            image_and_metadata = ImagesList[i]
            image_data = image_and_metadata[0]
            
            #DarkSubtracting
            image_data = image_data - MeanDarkFrame
            
            #Zeroing Everything that isn't the region of interest ROI
            image_data[0:y1,:] = 0
            image_data[y2:DimY,:] = 0
            image_data[:,0:x1] = 0
            image_data[:,x2:DimX] = 0
            
            #PixelsOfInterest = np.where(image_data > EventThreshStd*StdDarkFrame, 1, ZerosArray)
            PixelsOfInterest = np.where(image_data > thres, 1, ZerosArray)
            #An array with 1's in every pixel position with a count greater than thres and  0's elsewhere
            OnlyEventPixels = np.where(image_data > thres, image_data, ZerosArray)
            #The image array but everything but pixels definitely part of an event (PixelsOfInterest) are zeroed
            
            NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
            
            while NumRemainingPixelsOfInterest > 0:
                
                #print(NumRemainingPixelsOfInterest)
                
                y,x =  np.unravel_index(np.argmax(OnlyEventPixels, axis=None), OnlyEventPixels.shape)
                
                EventSignal = 0
                
                ximin = x-dx
                ximin = ximin.astype(int)
                ximax = x+dx+1
                ximax = ximax.astype(int)
                yimin = y-dy
                yimin = yimin.astype(int)
                yimax = y+dy+1
                yimax = yimax.astype(int)
                
                #Won't count event if it's too close to the edge of the frame
                if yimin <= 0 or yimax >= DimY or ximin <= 0 or ximax >= DimX:
                    
                    for xi in range(ximin, ximax):
                        for yi in range(yimin, yimax):
                            
                            if yi > -1 and xi > -1 and yi < DimY and xi < DimX:
                                
                                PixelsOfInterest[yi,xi] = 0
                                NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                                OnlyEventPixels[yi,xi] = 0
                                
                            if KeepEdgeEvents == True:
                                EventSignals.append(False)
                                EventX.append(x)
                                EventY.append(y)
                        
                    continue
                
                for xi in range(ximin, ximax):
                    for yi in range(yimin, yimax):
                        
                        #EventSignal = EventSignal + image_data[yi,xi]
                        
                        if image_data[yi,xi] > thresp:
                            EventSignal = EventSignal + image_data[yi,xi]
                            
                            #Added this to try to fix problem
                            image_data[yi,xi] = 0
                            
                        if Hotpix[yi,xi] == 1:
                            EventSignal = -100000
                            
                        PixelsOfInterest[yi,xi] = 0
                        NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                        OnlyEventPixels[yi,xi] = 0
                        
                if EventSignal > 0:
                    #We only append to the signals and positions lists after we've confirmed that there were no hotpixels in the sum
                    #i.e only if the Event Signal is positive because we set it to negitive if there were hot pixels.
                    EventSignals.append(EventSignal)
                    EventX.append(x)
                    EventY.append(y)
                
        EventSignals = np.array(EventSignals)
        EventX = np.array(EventX)
        EventY = np.array(EventY)
        EventsArray = np.array([EventSignals,EventX,EventY])
        
        return EventSignals, EventX, EventY, EventsArray
                
                
                
def NoEventPixelHistogram(ImagesList,DarkImagesList,ROI,MeanDarkFrame,StdDarkFrame,Hotpix,DimX,DimY,IntegrationRegionSize,EventThreshStd):
        #IntegrationRegionSize: the side dimension of the area to integrate around (can only be an odd number to be centered)
        #e.g. if IntegrationRegionSize = 5 integrate the signal over the 5x5 pixels centered around the brightest pixel in an event
        
        #EventThreshStd: the number of standard deviations above the noise level a pixel must be inorder to definitely be considered
        #part of an event (Has no impact on which pixels after and event has been located are included in the event sum)
        
        #MinPixVal: The minimum pixel value to be counted in when integrating over an event region 
        #e.g. MinPixVal = 0 means that any pixel with a value above 0 in the event integration with be counted towards the event signal
        #By getting rid of MinPixVal in Utility 1.5 I've gotten rid of thresp equivalent, and having done so hope to prevent the
        #histogram from skewing positive
        
        
        #ROI = [x1,x2,y1,y2]
        
        x1 = ROI[0]
        x2 = ROI[1]
        y1 = ROI[2]
        y2 = ROI[3]
        
        
        dx = (IntegrationRegionSize-1)/2
        dy = dx
        
        ZerosArray = np.zeros((DimY,DimX))
    
        for i in tqdm(range(len(ImagesList))):
        #for all images in list of images
            #print(i)
            #Getting the Image Array from the ImagesList
            image_and_metadata = ImagesList[i]
            image_data = image_and_metadata[0]
            
            dark_image_and_metadata = DarkImagesList[i]
            dark_image_data = dark_image_and_metadata[0]
            
            #DarkSubtracting
            image_data = image_data - MeanDarkFrame
            dark_image_data = dark_image_data - MeanDarkFrame
            #Zeroing Everything that isn't the region of interest ROI
            image_data[0:y1,:] = 0
            image_data[y2:DimY,:] = 0
            image_data[:,0:x1] = 0
            image_data[:,x2:DimX] = 0
            
            dark_image_data[0:y1,:] = 0
            dark_image_data[y2:DimY,:] = 0
            dark_image_data[:,0:x1] = 0
            dark_image_data[:,x2:DimX] = 0
            
            PixelsOfInterest = np.where(image_data > EventThreshStd*StdDarkFrame, 1, ZerosArray)
            #An array with 1's in every pixel position with a count greater than EventThreshStd stdev dark and  0's elsewhere
            OnlyEventPixels = np.where(image_data > EventThreshStd*StdDarkFrame, image_data, ZerosArray)
            #The image array but everything but pixels definitely part of an event (PixelsOfInterest) are zeroed
            
            PixelsCountedTowardsEvents = np.zeros((DimY,DimX))
            
            NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
            
            while NumRemainingPixelsOfInterest > 0:
                
                y,x =  np.unravel_index(np.argmax(OnlyEventPixels, axis=None), OnlyEventPixels.shape)
                
                ximin = x-dx
                ximin = ximin.astype(int)
                ximax = x+dx+1
                ximax = ximax.astype(int)
                yimin = y-dy
                yimin = yimin.astype(int)
                yimax = y+dy+1
                yimax = yimax.astype(int)
                
                #Won't count event if it's too close to the edge of the frame
                if yimin <= 0 or yimax >= DimY or ximin <= 0 or ximax >= DimX:
                    continue
                
                for xi in range(ximin, ximax):
                    for yi in range(yimin, yimax):
                        
                        PixelsCountedTowardsEvents[yi,xi] = 1
                            
                        PixelsOfInterest[yi,xi] = 0
                        NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                        OnlyEventPixels[yi,xi] = 0
            
            #setting every instance of a pixel counted towards an event equal to -100000 to later be removed from histogram
            image_data = np.where(PixelsCountedTowardsEvents > 0, -100000, image_data)
            dark_image_data = np.where(PixelsCountedTowardsEvents > 0, -100000, dark_image_data)
            
            #removing any remaining hotpixels by setting them to -100000 to later be removed from histogram
            image_data = np.where(Hotpix > 0, -100000, image_data)
            dark_image_data = np.where(Hotpix > 0, -100000, dark_image_data)
            
            #Cropping images so that only the ROI gets counted towards histogram
            cropped_image_data = image_data[y1:y2,x1:x2]
            cropped_dark_image_data = dark_image_data[y1:y2,x1:x2]
            
            AppendHistogram = cropped_image_data
            AppendHistogram = AppendHistogram.flatten()
            AppendHistogram = AppendHistogram[AppendHistogram != -100000]
            #removing all the counts of -100000 that were set when that pixel counted towards an event or was hot.
            
            AppendDarkHistogram = cropped_dark_image_data
            AppendDarkHistogram = AppendDarkHistogram.flatten()
            AppendDarkHistogram = AppendDarkHistogram[AppendDarkHistogram != -100000]
            #removing all the counts of -100000 that were set when that pixel counted towards an event or was hot.
            
            if i == 0:
               
                Histogram = AppendHistogram
                DarkHistogram = AppendDarkHistogram

            else:

                Histogram = np.append(Histogram,AppendHistogram)
                DarkHistogram = np.append(DarkHistogram,AppendDarkHistogram)
                
        
        return Histogram, DarkHistogram
    
    
def NoEventPixelHistogramStandard(ImagesList,ROI,MeanDarkFrame,Hotpix,DimX,DimY,IntegrationRegionSize,thres):
        #IntegrationRegionSize: the side dimension of the area to integrate around (can only be an odd number to be centered)
        #e.g. if IntegrationRegionSize = 5 integrate the signal over the 5x5 pixels centered around the brightest pixel in an event
        
        #thres: the counts above which a pixel must have inorder to definitely be considered
        #part of an event (Has no impact on which pixels after and event has been located are included in the event sum)
        
        
        #ROI = [x1,x2,y1,y2]
        
        x1 = ROI[0]
        x2 = ROI[1]
        y1 = ROI[2]
        y2 = ROI[3]
        
        
        dx = (IntegrationRegionSize-1)/2
        dy = dx
        
        ZerosArray = np.zeros((DimY,DimX))
    
        for i in tqdm(range(len(ImagesList))):
        #for all images in list of images
            #Getting the Image Array from the ImagesList
            image_and_metadata = ImagesList[i]
            image_data = image_and_metadata[0]
            
            
            #DarkSubtracting
            image_data = image_data - MeanDarkFrame
            #Zeroing Everything that isn't the region of interest ROI
            image_data[0:y1,:] = 0
            image_data[y2:DimY,:] = 0
            image_data[:,0:x1] = 0
            image_data[:,x2:DimX] = 0
            
            PixelsOfInterest = np.where(image_data > thres, 1, ZerosArray)
            #An array with 1's in every pixel position with a count greater than thres and  0's elsewhere
            OnlyEventPixels = np.where(image_data > thres, image_data, ZerosArray)
            #The image array but everything but pixels definitely part of an event (PixelsOfInterest) are zeroed
            
            PixelsCountedTowardsEvents = np.zeros((DimY,DimX))
            NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
            
            while NumRemainingPixelsOfInterest > 0:
                
                y,x =  np.unravel_index(np.argmax(OnlyEventPixels, axis=None), OnlyEventPixels.shape)
                
                ximin = x-dx
                ximin = ximin.astype(int)
                ximax = x+dx+1
                ximax = ximax.astype(int)
                yimin = y-dy
                yimin = yimin.astype(int)
                yimax = y+dy+1
                yimax = yimax.astype(int)
                
                #Won't count event if it's too close to the edge of the frame
                if yimin <= 0 or yimax >= DimY or ximin <= 0 or ximax >= DimX:
                    
                    for xi in range(ximin, ximax):
                        for yi in range(yimin, yimax):
                            
                            if yi > -1 and xi > -1 and yi < DimY and xi < DimX:
                                
                                PixelsOfInterest[yi,xi] = 0
                                NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                                OnlyEventPixels[yi,xi] = 0
                    
                    continue
                
                for xi in range(ximin, ximax):
                    for yi in range(yimin, yimax):
                        
                        PixelsCountedTowardsEvents[yi,xi] = 1
                            
                        PixelsOfInterest[yi,xi] = 0
                        NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                        OnlyEventPixels[yi,xi] = 0
            
            #setting every instance of a pixel counted towards an event equal to -100000 to later be removed from histogram
            image_data = np.where(PixelsCountedTowardsEvents > 0, -100000, image_data)
            
            #removing any remaining hotpixels by setting them to -100000 to later be removed from histogram
            image_data = np.where(Hotpix > 0, -100000, image_data)
            
            #Cropping images so that only the ROI gets counted towards histogram
            cropped_image_data = image_data[y1:y2,x1:x2]
            
            AppendHistogram = cropped_image_data
            AppendHistogram = AppendHistogram.flatten()
            AppendHistogram = AppendHistogram[AppendHistogram != -100000]
            #removing all the counts of -100000 that were set when that pixel counted towards an event or was hot.
            
            if i == 0:
                Histogram = AppendHistogram
            else:
                Histogram = np.append(Histogram,AppendHistogram)
                
        return Histogram

def NoEventPixelHistogramStandardElectrons(ImagesList,ROI,MeanDarkFrame,Hotpix,DimX,DimY,IntegrationRegionSize,thres,Gain,Offset):
        #IntegrationRegionSize: the side dimension of the area to integrate around (can only be an odd number to be centered)
        #e.g. if IntegrationRegionSize = 5 integrate the signal over the 5x5 pixels centered around the brightest pixel in an event
        
        #thres: the counts (in ADU) above which a pixel must have inorder to definitely be considered
        #part of an event (Has no impact on which pixels after and event has been located are included in the event sum)
        
        
        #ROI = [x1,x2,y1,y2]
        
        x1 = ROI[0]
        x2 = ROI[1]
        y1 = ROI[2]
        y2 = ROI[3]
        
        
        dx = (IntegrationRegionSize-1)/2
        dy = dx
        
        ZerosArray = np.zeros((DimY,DimX))
    
        for i in tqdm(range(len(ImagesList))):
        #for all images in list of images
            #Getting the Image Array from the ImagesList
            image_and_metadata = ImagesList[i]
            image_dataADU = image_and_metadata[0]
            
            
            #DarkSubtracting
            image_dataADU = image_dataADU - MeanDarkFrame
            #Zeroing Everything that isn't the region of interest ROI
            image_dataADU[0:y1,:] = 0
            image_dataADU[y2:DimY,:] = 0
            image_dataADU[:,0:x1] = 0
            image_dataADU[:,x2:DimX] = 0
            
            #Now making image_data, but in Electrons
            image_data = image_dataADU*Gain
            image_data[0:y1,:] = 0
            image_data[y2:DimY,:] = 0
            image_data[:,0:x1] = 0
            image_data[:,x2:DimX] = 0
            '''
            #MeanDarkFrame_e = (MeanDarkFrame-Offset)*Gain
            #image_data = ((image_dataADU-Offset)*Gain) - MeanDarkFrame_e
            image_data = ((image_dataADU-MeanDarkFrame)*Gain)
            image_data[0:y1,:] = 0
            image_data[y2:DimY,:] = 0
            image_data[:,0:x1] = 0
            image_data[:,x2:DimX] = 0
            '''
            PixelsOfInterest = np.where(image_dataADU > thres, 1, ZerosArray)
            #An array with 1's in every pixel position with a count greater than thres and  0's elsewhere
            OnlyEventPixels = np.where(image_dataADU > thres, image_data, ZerosArray)
            #The image array but everything but pixels definitely part of an event (PixelsOfInterest) are zeroed
            
            PixelsCountedTowardsEvents = np.zeros((DimY,DimX))
            NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
            
            while NumRemainingPixelsOfInterest > 0:
                
                y,x =  np.unravel_index(np.argmax(OnlyEventPixels, axis=None), OnlyEventPixels.shape)
                
                ximin = x-dx
                ximin = ximin.astype(int)
                ximax = x+dx+1
                ximax = ximax.astype(int)
                yimin = y-dy
                yimin = yimin.astype(int)
                yimax = y+dy+1
                yimax = yimax.astype(int)
                
                #Won't count event if it's too close to the edge of the frame
                if yimin <= 0 or yimax >= DimY or ximin <= 0 or ximax >= DimX:
                    
                    for xi in range(ximin, ximax):
                        for yi in range(yimin, yimax):
                            
                            if yi > -1 and xi > -1 and yi < DimY and xi < DimX:
                                
                                PixelsOfInterest[yi,xi] = 0
                                NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                                OnlyEventPixels[yi,xi] = 0
                    
                    continue
                
                for xi in range(ximin, ximax):
                    for yi in range(yimin, yimax):
                        
                        PixelsCountedTowardsEvents[yi,xi] = 1
                            
                        PixelsOfInterest[yi,xi] = 0
                        NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                        OnlyEventPixels[yi,xi] = 0
            
            #setting every instance of a pixel counted towards an event equal to -100000 to later be removed from histogram
            image_data = np.where(PixelsCountedTowardsEvents > 0, -100000, image_data)
            
            #removing any remaining hotpixels by setting them to -100000 to later be removed from histogram
            image_data = np.where(Hotpix > 0, -100000, image_data)
            
            #Cropping images so that only the ROI gets counted towards histogram
            cropped_image_data = image_data[y1:y2,x1:x2]
            
            AppendHistogram = cropped_image_data
            AppendHistogram = AppendHistogram.flatten()
            AppendHistogram = AppendHistogram[AppendHistogram != -100000]
            #removing all the counts of -100000 that were set when that pixel counted towards an event or was hot.
            
            if i == 0:
                Histogram = AppendHistogram
            else:
                Histogram = np.append(Histogram,AppendHistogram)
                
        return Histogram
                
def FindCentroids(ImageData,nsigma,verbose = False):

    #threshold = detect_threshold(ImageData, nsigma=2.)
    #determines the threshold level for 2 sigma above background
    #threshold = detect_threshold(ImageData, nsigma=11.)
    threshold = detect_threshold(ImageData, nsigma=nsigma)

    #producing segmentation image with sources with 10 more connected pixels above threshold
    #the kernel from the commented out above was optional
    segm = detect_sources(ImageData, threshold, npixels=20)

    cat = SourceCatalog(ImageData, segm)
    #not available in this version of python
    #cat = source_properties(ImageData, segm)
    sources = cat.to_table()
    sources_table = sources
    #named for convenience to return from this function
    if verbose == True:
        print(sources)
    '''
    for col in sources.colnames:
        sources[col].info.format = '%g'
    #I don't remember why I'm doing this, but its from the point source version
    '''

    Brightnesses = np.array(sources_table["segment_flux"])
    source_index = np.argmax(Brightnesses)

    xcentroid = np.array(sources['xcentroid'])
    xcentroid = xcentroid[source_index]
    ycentroid = np.array(sources['ycentroid'])
    ycentroid = ycentroid[source_index]
    #centroid = {'xcentroid':xcentroid,'ycentroid':ycentroid}
    #positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    
    #centroid, source_index

    return xcentroid, ycentroid, sources_table

#PhilEventFinderWithThresp

def EventFinderPhilThresp(ImagesList,ROI,MeanDarkFrame,StdDarkFrame,Hotpix,DimX,DimY,IntegrationRegionSize,EventThreshStd,MinPixVal):
        #IntegrationRegionSize: the side dimension of the area to integrate around (can only be an odd number to be centered)
        #e.g. if IntegrationRegionSize = 5 integrate the signal over the 5x5 pixels centered around the brightest pixel in an event
        
        #EventThreshStd: the number of standard deviations above the noise level a pixel must be inorder to definitely be considered
        #part of an event (Has not impact on which pixels after and event has been located are included in the event sum)
        
        #MinPixVal=thresp: The minimum pixel value to be counted in when integrating over an event region 
        #e.g. MinPixVal = 0 means that any pixel with a value above 0 in the event integration with be counted towards the event signal
        
        #ROI = [x1,x2,y1,y2]
        
        x1 = ROI[0]
        x2 = ROI[1]
        y1 = ROI[2]
        y2 = ROI[3]
        
        
        dx = (IntegrationRegionSize-1)/2
        dy = dx
        
        ZerosArray = np.zeros((DimY,DimX))
        
        EventSignals = []
        EventX = []
        EventY = []
    
        for i in range(len(ImagesList)):
        #for all images in list of images
            print(i)
            #Getting the Image Array from the ImagesList
            image_and_metadata = ImagesList[i]
            image_data = image_and_metadata[0]
            
            #DarkSubtracting
            image_data = image_data - MeanDarkFrame
            
            #Zeroing Everything that isn't the region of interest ROI
            image_data[0:y1,:] = 0
            image_data[y2:DimY,:] = 0
            image_data[:,0:x1] = 0
            image_data[:,x2:DimX] = 0
            
            PixelsOfInterest = np.where(image_data > EventThreshStd*StdDarkFrame, 1, ZerosArray)
            #print(PixelsOfInterest)
            #An array with 1's in every pixel position with a count greater than EventThreshStd stdev dark and  0's elsewhere
            OnlyEventPixels = np.where(image_data > EventThreshStd*StdDarkFrame, image_data, ZerosArray)
            #The image array but everything but pixels definitely part of an event (PixelsOfInterest) are zeroed
            
            NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
            
            while NumRemainingPixelsOfInterest > 0:
                
                #print(NumRemainingPixelsOfInterest)
                
                #xyMax =  np.argmax(OnlyEventPixels)
                #x = xyMax[1]
                #y = xyMax[0]
                
                y,x =  np.unravel_index(np.argmax(OnlyEventPixels, axis=None), OnlyEventPixels.shape)
                
                EventSignal = 0
                
                #x = x.astype(int)
                #y = y.astype(int)
                
                ximin = x-dx
                ximin = ximin.astype(int)
                ximax = x+dx+1
                ximax = ximax.astype(int)
                yimin = y-dy
                yimin = yimin.astype(int)
                yimax = y+dy+1
                yimax = yimax.astype(int)
                #print(ximin.dtype)
                #print(ximax1.dtype)
                
                #Won't count event if it's too close to the edge of the frame
                if yimin <= 0 or yimax >= DimY or ximin <= 0 or ximax >= DimX:
                    continue
                
                for xi in range(ximin, ximax):
                    for yi in range(yimin, yimax):
                        
                        if image_data[yi,xi] > MinPixVal:
                            EventSignal = EventSignal + image_data[yi,xi]
                            
                        if Hotpix[yi,xi] == 1:
                            EventSignal = -100000
                            
                        PixelsOfInterest[yi,xi] = 0
                        NumRemainingPixelsOfInterest = np.sum(PixelsOfInterest)
                        OnlyEventPixels[yi,xi] = 0
                        
                if EventSignal > 0:
                    #We only append to the signals and positions lists after we've confirmed that there were no hotpixels in the sum
                    #i.e only if the Event Signal is positive because we set it to negitive if there were hot pixels.
                    EventSignals.append(EventSignal)
                    EventX.append(x)
                    EventY.append(y)
                
        EventSignals = np.array(EventSignals)
        EventX = np.array(EventX)
        EventY = np.array(EventY)
        EventsArray = np.array([EventSignals,EventX,EventY])
        
        return EventSignals, EventX, EventY, EventsArray


def MakePolarArray(image_data,Hotpix,MaskArray,xcentroid,ycentroid,DimX,DimY,deg = True):
    
    PixelSignals = []
    EventX = []
    EventY = []
    rList = []
    phiList = []

    for xi in range(DimX):
        for yi in range(DimY):
            
            if MaskArray[yi,xi] == False:
                continue
            
            if Hotpix[yi,xi] == 0:
                PixelSignal = image_data[yi,xi]
                newx = xi - xcentroid
                newy = yi - ycentroid
                r = math.sqrt((newx**2) + (newy**2))
                phi = math.atan2(newy,newx)
                
                if deg == True:
                    phi = math.degrees(phi)
                
                EventX.append(xi)
                EventY.append(yi)
                PixelSignals.append(PixelSignal)
                rList.append(r)
                phiList.append(phi)
                
    PixelSignals = np.array(PixelSignals)
    EventX = np.array(EventX)
    EventY = np.array(EventY)
    rArray = np.array(rList)
    phiArray = np.array(phiList)
    PolarArray = np.array([PixelSignals,EventX,EventY,rArray,phiArray])
    
    return PolarArray

#<Functions From Jupyter>------------------------------------------------------

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def MakeHistogramSet(ToHistogramArray,MinX,MaxX,BinsPerUnitX,BinSize=False):
    HistBins = [MinX]
    PlotBins = [MinX]
    
    if BinSize != False:
        BinsPerUnitX = 1/BinSize
        BinsPerUnitX = int(BinsPerUnitX)
        ActualBinSize = 1/BinsPerUnitX
        print("Actual Bin Size: " + str(ActualBinSize))
        
    
    for i in range(int((MaxX-MinX)*BinsPerUnitX) + 1):
        HistBins.append(HistBins[i]+1/BinsPerUnitX)
        PlotBins.append(PlotBins[i]+1/BinsPerUnitX)
        
    HistBins = np.array(HistBins)
    HistBins = HistBins - 0.5/BinsPerUnitX
    
    PlotBins = np.array(PlotBins)
    PlotBins = PlotBins[:-1]
    
    theHistogram = np.histogram(ToHistogramArray, bins = HistBins)
    theHistogram = theHistogram[0]
    
    print("Only Valid if Integer: " + str((MaxX-MinX)*BinsPerUnitX))
    print("Bin Size: " + str(1/BinsPerUnitX))
    
    
    return theHistogram, PlotBins

def NoHotsToHistogram(Frame,ROI,HotsArray):
    x1 = ROI[0]
    x2 = ROI[1]
    y1 = ROI[2]
    y2 = ROI[3]
    PixelSignals = []
    for xi in range(x1,x2):
        for yi in range(y1,y2):
            if HotsArray[yi,xi] < 1:
                PixelSignals.append(Frame[yi,xi])
    PixelSignals = np.array(PixelSignals)
    return PixelSignals

def ChiSquaredOfFit1D(DataArray,ErrorArray,FitArray,NumberOfFitParameters):
    
    ChiSquared = 0
    NumberDataPoints = np.size(DataArray)
    
    for i in tqdm(range(NumberDataPoints)):
        ChiSquared += ((FitArray[i] - DataArray[i])**2)/((ErrorArray[i])**2)
        
    dof = NumberDataPoints - NumberOfFitParameters
    ReducedChiSquared = ChiSquared/dof
    
    return ChiSquared, ReducedChiSquared, dof

def ChiSquaredOfFit(DataFrame,ErrorFrame,FitFrame,HotsArray,bArray,min_b,DimX,DimY,NumberOfFitParameters):
    
    ChiSquared = 0
    NumberDataPoints = 0
    
    for xi in tqdm(range(DimX)):
        for yi in range(DimY):
            if bArray[yi,xi] > min_b and HotsArray[yi,xi] < 1:
                
                ThisTerm = ((FitFrame[yi,xi] - DataFrame[yi,xi])**2)/((ErrorFrame[yi,xi])**2)
                ChiSquared += ThisTerm
                NumberDataPoints += 1
    
    dof = NumberDataPoints - NumberOfFitParameters
    ReducedChiSquared = ChiSquared/dof
    
    return ChiSquared, ReducedChiSquared

#L1: Distance from crystal to photodiode [mm]
#sigmaL1: Uncertainty in distance from crystal to photodiode [mm]
#L2: Distance from crystal to sensor [mm]
#sigmaL1: Uncertainty in distance from crystal to photodiode [mm]
#dimX_PD: X-axis dimension length of the chamber photodiode [mm]
#dimY_PD: Y-axis dimension length of the chamber photodiode [mm]
#DimX_Sensor: X-axis dimension of the sensor [pixels]
#DimY_Sensor: Y-axis dimension of the sensor [pixels]
#PixelScale: Length of a sensor pixel [mm]

def HypotheticalSensorDimensions(L1,L2,sigmaL1,sigmaL2,dimX_PD,dimY_PD,DimX_Sensor,DimY_Sensor,PixelScale):
    
    dimX_Sensor = DimX_Sensor*PixelScale
    dimY_Sensor = DimY_Sensor*PixelScale
    #The x and y dimensions of the sensor in units of mm converted from pixels
    
    '''
    ThetaPD_X = math.atan((dimX_PD/2)/L1)
    ThetaPD_Y = math.atan((dimY_PD/2)/L1)
    #Half the angular extent of the photodiode from the crystal along either the x or y axis
    
    ThetaSensor_X = math.atan((dimX_Sensor/2)/L2)
    ThetaSensor_Y= math.atan((dimY_Sensor/2)/L2)
    #Half the angular extent of the sensor from the crystal along either the x or y axis
    
    dimX_Sensor_H = 2*L2*math.tan(ThetaPD_X)
    dimY_Sensor_H = 2*L2*math.tan(ThetaPD_Y)
    #or after having looked closer at the math
    '''
    dimX_Sensor_H = dimX_PD*(L2/L1)
    dimY_Sensor_H = dimY_PD*(L2/L1)
    
    DimX_Sensor_H = dimX_Sensor_H/PixelScale
    DimY_Sensor_H = dimY_Sensor_H/PixelScale
    
    #maximums
    L1Max = L1 + sigmaL1
    L2Max = L2 + sigmaL2
    
    #minimums
    L1Min = L1 - sigmaL1
    L2Min = L2 - sigmaL2
    
    #Upper bound sensor dimensions occur when L1 is at minimum and L2 is at maximum
    dimX_Sensor_H_Max = dimX_PD*(L2Max/L1Min)
    dimY_Sensor_H_Max = dimY_PD*(L2Max/L1Min)
    
    DimX_Sensor_H_Max = dimX_Sensor_H_Max/PixelScale
    DimY_Sensor_H_Max = dimY_Sensor_H_Max/PixelScale
    
    #Lower bound sensor dimensions occur when L1 is at maximum and L2 is at minimum
    dimX_Sensor_H_Min = dimX_PD*(L2Min/L1Max)
    dimY_Sensor_H_Min = dimY_PD*(L2Min/L1Max)
    
    DimX_Sensor_H_Min = dimX_Sensor_H_Min/PixelScale
    DimY_Sensor_H_Min = dimY_Sensor_H_Min/PixelScale
    
    
    return DimX_Sensor_H, DimY_Sensor_H, DimX_Sensor_H_Max, DimY_Sensor_H_Max, DimX_Sensor_H_Min, DimY_Sensor_H_Min
    
def MakeSumFrameAndPoissonError(ImagesList,MeanDarkFrame,HotsArray,DimX,DimY,Gain,Offset):
    #HotsArray to zero hotpixels
    #note that in the error arrays the hotpixels are set to False not 0.
    SumSignalFrame = np.zeros((DimY,DimX))
    SumDarkSubSignalFrame = np.zeros((DimY,DimX))
    SumExposureTime = 0
    ZerosArray = np.zeros((DimY,DimX))
    #Converting from ADU to e-
    MeanDarkFrame = (MeanDarkFrame-Offset)*Gain
    
    for i in range(len(ImagesList)):
    #for all images in list of images
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        #Convert from ADU to e-
        image_data = (image_data - Offset)*Gain
        #dark subtraction
        dark_sub_image_data = image_data - MeanDarkFrame
        SumSignalFrame = SumSignalFrame + image_data
        SumDarkSubSignalFrame = SumDarkSubSignalFrame + dark_sub_image_data

        SumExposureTime = SumExposureTime + float(image_and_metadata[1])
        
    MeanExposureTime = SumExposureTime/len(ImagesList)
    
    #Zeroing the signal of hotpixels
    SumSignalFrame = np.where(HotsArray > 0, 0, SumSignalFrame)
    SumDarkSubSignalFrame = np.where(HotsArray > 0, 0, SumDarkSubSignalFrame)
    
    ErrorFrame = np.sqrt(SumSignalFrame)
    DarkSubErrorFrame = np.sqrt((len(ImagesList)*MeanDarkFrame)+SumDarkSubSignalFrame)
    #Setting error to false on hot pixels which should not be counted towards error or anything else
    ErrorFrame = np.where(HotsArray > 0, False, ErrorFrame)
    DarkSubErrorFrame = np.where(HotsArray > 0, False, DarkSubErrorFrame)
    
    return SumSignalFrame, SumDarkSubSignalFrame, ErrorFrame, DarkSubErrorFrame, MeanExposureTime

def MakeSumFrameAndGehrelsError(ImagesList,MeanDarkFrame,HotsArray,DimX,DimY,Gain,Offset):
    #HotsArray to zero hotpixels
    #note that in the error arrays the hotpixels are set to False not 0.
    SumSignalFrame = np.zeros((DimY,DimX))
    SumDarkSubSignalFrame = np.zeros((DimY,DimX))
    SumExposureTime = 0
    ZerosArray = np.zeros((DimY,DimX))
    #Converting from ADU to e-
    MeanDarkFrame = (MeanDarkFrame-Offset)*Gain
    
    for i in range(len(ImagesList)):
    #for all images in list of images
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        #Convert from ADU to e- and accouning for ADU offset
        image_data = (image_data - Offset)*Gain
        #dark subtraction
        dark_sub_image_data = image_data - MeanDarkFrame
        SumSignalFrame = SumSignalFrame + image_data
        SumDarkSubSignalFrame = SumDarkSubSignalFrame + dark_sub_image_data

        SumExposureTime = SumExposureTime + float(image_and_metadata[1])
        
    MeanExposureTime = SumExposureTime/len(ImagesList)
    
    #Zeroing the signal of hotpixels
    SumSignalFrame = np.where(HotsArray > 0, 0, SumSignalFrame)
    SumDarkSubSignalFrame = np.where(HotsArray > 0, 0, SumDarkSubSignalFrame)
    
    #Determining Gehrels Error Frames
    SumSignalWithNoNegitives = np.where(SumSignalFrame<0,0,SumSignalFrame)
    ErrorFrame = np.sqrt(SumSignalWithNoNegitives + 0.75) + 1
    #I got negative numbers in the square root
    #DarkSubErrorFrame = np.sqrt((len(ImagesList)*MeanDarkFrame)+SumDarkSubSignalFrame)
    #Setting error to false on hot pixels which should not be counted towards error or anything else
    ErrorFrame = np.where(HotsArray > 0, False, ErrorFrame)
    #DarkSubErrorFrame = np.where(HotsArray > 0, False, DarkSubErrorFrame)
    
    return SumSignalFrame, SumDarkSubSignalFrame, ErrorFrame,  MeanExposureTime

def MakeSumAndMeanFrameAndEmpiricalError(ImagesList,MeanDarkFrame,HotsArray,DimX,DimY,Gain,Sigma_Gain,FGV,Offset):
    #HotsArray to zero hotpixels
    #note that in the error arrays the hotpixels are set to False not 0.
    #Sigma_Gain: Uncertainty on the mean gain value
    #Check on all this later to be sure I didn't mess up the math or make false assumptions.
    SumSignalFrameADU = np.zeros((DimY,DimX))
    SumSquaredSignalFrameADU = np.zeros((DimY,DimX))
    SumDarkSubSignalFrameADU = np.zeros((DimY,DimX))
    SumExposureTime = 0
    ZerosArray = np.zeros((DimY,DimX))
    
    MeanDarkFrameADU_no_offset = (MeanDarkFrame-Offset)
    MeanDarkFrameADU = MeanDarkFrame
    
    N_Images = len(ImagesList)
    
    for i in range(len(ImagesList)):
    #for all images in list of images
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        image_dataADU = image_data
        #Accouning for ADU offset
        image_data_no_offsetADU = (image_data - Offset)
        #dark subtraction
        dark_sub_image_data = image_dataADU - MeanDarkFrameADU
        SumSignalFrameADU = SumSignalFrameADU + image_data
        SumSquaredSignalFrameADU = SumSquaredSignalFrameADU + (image_data_no_offsetADU**2)
        SumDarkSubSignalFrameADU = SumDarkSubSignalFrameADU + dark_sub_image_data

        SumExposureTime = SumExposureTime + float(image_and_metadata[1])
        
    MeanExposureTime = SumExposureTime/N_Images
    
    #Correcting for offset
    SumSignalFrameADU = SumSignalFrameADU - Offset*N_Images
    
    
    #Zeroing the signal of hotpixels
    SumSignalFrameADU = np.where(HotsArray > 0, 0, SumSignalFrameADU)
    SumSquaredSignalFrameADU = np.where(HotsArray > 0, 0, SumSquaredSignalFrameADU)
    SumDarkSubSignalFrameADU = np.where(HotsArray > 0, 0, SumDarkSubSignalFrameADU)
    
    MeanSignalFrameADU = SumSignalFrameADU/N_Images
    MeanSquaredSignalFrameADU = SumSquaredSignalFrameADU/N_Images
    MeanDarkSubSignalFrameADU = SumDarkSubSignalFrameADU/N_Images
    
    #Determining Empirical Error Frame for ADU Signal
    
    #Calculating "Sample Standard Deviation" Frame
    EmpiricalSampleStdFrameADU = np.sqrt(MeanSquaredSignalFrameADU - (MeanSignalFrameADU**2))
    EmpiricalStandardErrorOfSumFrameADU = EmpiricalSampleStdFrameADU
    
    #Calculating "Standard Error of Mean" Frame
    EmpiricalStandardErrorOfMeanFrameADU = EmpiricalSampleStdFrameADU/math.sqrt(N_Images)
    
    #Converting Signals to Photoelectrons e-
    SumSignalFrame = SumSignalFrameADU*Gain
    SumDarkSubSignalFrame = SumDarkSubSignalFrameADU*Gain
    
    MeanSignalFrame = MeanSignalFrameADU*Gain
    MeanDarkSubSignalFrame = MeanDarkSubSignalFrameADU*Gain
    
    #Propogating Error to get Error Frames in units of Photoelectrons
    
    #With only Fractional Gain Variation (FGV) (no overall gain variation accounted for)
    Sigma_Gain_FGV = Gain*FGV
    
    StandardErrorOfMeanFrameFGV = np.sqrt(((Gain*EmpiricalStandardErrorOfMeanFrameADU)**2) + ((MeanDarkSubSignalFrameADU*Sigma_Gain_FGV)**2))
    StandardErrorOfSumFrameFGV = np.sqrt(((Gain*EmpiricalStandardErrorOfSumFrameADU)**2) + ((SumDarkSubSignalFrameADU*Sigma_Gain_FGV)**2))
    
    #Now with Fractional Gain Variation (FGV) AND Overall Gain Uncertainty (Sigma_Gain) Accounted for)
    #Via uncertainty propogation and algebra on g = g_0 + g_1(x,y) where g_1 = FGV*(MeanGain=Gain) 
    
    Sigma_Gain_All = Sigma_Gain*np.sqrt(1+(FGV**2))
    
    StandardErrorOfMeanFrameAll = np.sqrt(((Gain*EmpiricalStandardErrorOfMeanFrameADU)**2) + ((MeanDarkSubSignalFrameADU*Sigma_Gain_All)**2))
    StandardErrorOfSumFrameAll = np.sqrt(((Gain*EmpiricalStandardErrorOfSumFrameADU)**2) + ((SumDarkSubSignalFrameADU*Sigma_Gain_All)**2))
    
    #Setting error to false on hot pixels which should not be counted towards error or anything else
    StandardErrorOfMeanFrameFGV = np.where(HotsArray > 0, False, StandardErrorOfMeanFrameFGV)
    StandardErrorOfSumFrameFGV = np.where(HotsArray > 0, False, StandardErrorOfSumFrameFGV)
    StandardErrorOfMeanFrameAll = np.where(HotsArray > 0, False, StandardErrorOfMeanFrameAll)
    StandardErrorOfSumFrameAll = np.where(HotsArray > 0, False, StandardErrorOfSumFrameAll)
    
    return SumSignalFrame, SumDarkSubSignalFrame, StandardErrorOfSumFrameFGV, StandardErrorOfSumFrameAll, MeanSignalFrame, MeanDarkSubSignalFrame, StandardErrorOfMeanFrameFGV, StandardErrorOfMeanFrameAll,  MeanExposureTime

def NoHotsFrameROIMean(Frame,ROI,Hotpixels):
    return np.mean(NoHotsToHistogram(Frame,ROI,Hotpixels))

def Find_b(x,y,x_centroid,y_centroid,eccentricity,orientation):
    theta = orientation
    ecc = eccentricity
    
    newx = x-x_centroid
    newy = y-y_centroid
    
    phi = theta - math.atan2(newy,newx)
    
    r = math.sqrt((newx**2)+(newy**2))
    
    b = r*math.sqrt(1-(ecc**2)*(math.cos(phi)**2))
    
    return b

def Make_b_Array(DimX,DimY,x_centroid,y_centroid,eccentricity,orientation):
    b_array = np.zeros((DimY,DimX))
    
    for xi in tqdm(range(DimX)):
        for yi in range(DimY):
            b_array[yi,xi] = Find_b(xi,yi,x_centroid,y_centroid,eccentricity,orientation)
            
    return b_array

def Make_b_Array16(DimX,DimY,x_centroid,y_centroid,eccentricity,orientation):
    b_array = np.zeros((DimY,DimX), dtype = np.float16)
    
    for xi in tqdm(range(DimX)):
        for yi in range(DimY):
            b_array[yi,xi] = Find_b(xi,yi,x_centroid,y_centroid,eccentricity,orientation)
            
    return b_array

def Make_b_Array16_2(DimX,DimY,x_centroid,y_centroid,eccentricity,orientation):
    #b_array = np.zeros((DimY,DimX), dtype = np.float16)
    
    theta = orientation
    ecc = eccentricity
    
    #Original x
    #x = np.zeros((DimY,DimX), dtype = np.float32)
    #y = np.zeros((DimY,DimX), dtype = np.float32)
    
    newx = np.zeros((DimY,DimX), dtype = np.float32)
    newy = np.zeros((DimY,DimX), dtype = np.float32)
    
    for xi in tqdm(range(DimX)):
        newx[:, xi] =  xi - x_centroid
    for yi in tqdm(range(DimY)):
        newx[yi,:] =  yi - y_centroid
    
    
    #newx = x-x_centroid
    #newy = y-y_centroid
    
    phi = (-1)*np.arctan2(newy,newx) + theta
    
    r = np.sqrt((newx**2)+(newy**2))
    
    b_array = r*np.sqrt(1-(ecc**2)*(np.cos(phi)**2))
    
    return b_array

def Make_b_Array32memmap(DimX,DimY,x_centroid,y_centroid,eccentricity,orientation,memmapName,memmapDirectory):
    
    theta = orientation
    ecc = eccentricity
    
    #ZerosArray = np.zeros((DimY,DimX), dtype = np.float32)
    
    #Initializing Zeros Array memmap
    #ZerosMemmap = np.memmap(memmapDirectory+'zeros.mymemmap', dtype='float32', mode='w+', shape=(DimY,DimX))
    #Writing ZerosArray to ZerosMemmap (It might do this by default)
    #ZerosMemmap[:] = ZerosArray
    
    #initializing newx and newy memmaps
    newx = np.memmap(memmapDirectory+'newx.mymemmap', dtype='float32', mode='w+', shape=(DimY,DimX))
    newy = np.memmap(memmapDirectory+'newy.mymemmap', dtype='float32', mode='w+', shape=(DimY,DimX))
    
    for xi in tqdm(range(DimX)):
        newx[:, xi] =  xi - x_centroid
    for yi in tqdm(range(DimY)):
        newy[yi,:] =  yi - y_centroid
        
    #I put the entire operation into one confusing line to save ram by only having one numpy array 
    b_array = (np.sqrt((newx**2)+(newy**2)))*np.sqrt(1-(ecc**2)*(np.cos((-1)*np.arctan2(newy,newx) + theta)**2))
    #initalizing output 
    b_memmap = np.memmap(memmapDirectory+memmapName+'.mymemmap', dtype='float32', mode='w+', shape=(DimY,DimX))
    b_memmap[:] = b_array[:]
    
    del b_array
    del newx
    del newy
    
    gc.collect()
    
    return b_memmap


def FindFitPixelValuesOG(func,fitParameters,DimX,DimY,min_b,x_centroid,y_centroid,eccentricity,orientation):
    
    FitValuesArray = np.zeros((DimY,DimX))
    
    for xi in range(DimX):
        for yi in range(DimY):
            bi = Find_b(xi,yi,x_centroid,y_centroid,eccentricity,orientation)
            if bi < min_b:
                continue
            
            FitValuesArray[yi,xi] = func(bi, *fitParameters)
            
    return FitValuesArray


def FindFitPixelValues(func,fitParameters,b_array,min_b):
    
    FitValuesArray = func(b_array, *fitParameters)
    FitValuesArray = np.where(b_array < min_b, 0, FitValuesArray)
    
    return FitValuesArray

def FindFitErrorPixelValues(ErrFunc,fitParameters,ErrFitParameters,b_array,min_b):
    
    FitErrorValuesArray = ErrFunc(b_array, fitParameters,ErrFitParameters)
    FitErrorValuesArray = np.where(b_array < min_b, 0, FitErrorValuesArray)
    
    return FitErrorValuesArray

def GenerateEllipseParameters(b_out,eccentricity,width):
    
    b_in = b_out - width
    a_out = b_out/(math.sqrt(1-(eccentricity**2)))
    a_in = (b_in/b_out)*a_out
    #from the same way photutils finds b_in in EllipticalAnnulus when ungiven
    #could try a_in = a_out - width if I get bad fits fromthe way I'm doing this.
    #or a_in = b_in/(math.sqrt(1-(eccentricity**2)))
    return a_in, a_out, b_out, b_in

def GetApertureSum(image_data,aperture):
    phot_table = aperture_photometry(image_data, aperture)
    aperture_sum = phot_table["aperture_sum"][0]
    return aperture_sum

def GetApertureInfo(image_data,aperture):
    
    AperturePixels = aperture.area_overlap(image_data,method='center')
    #do_photometry(data, error=None, mask=None, method='exact', subpixels=5)
    '''error: array_like or Quantity, optional
    The pixel-wise Gaussian 1-sigma errors of the input data. error is assumed to include all sources of error, 
    including the Poisson error of the sources (see calc_total_error) . 
    error must have the same shape as the input data.'''
    AperturePhotometry = aperture.do_photometry(image_data,method='center')
    ApertureSum = AperturePhotometry[0][0]
    ApertureSumError = AperturePhotometry[1]
    
    return AperturePixels, ApertureSum, ApertureSumError

def GetEllipticalAnnulusSumMeanAndErrors1(image_data,b_out,b_in,b_array,ErrorFrame,HotsArray,DimX,DimY):
    
    ZerosArray = np.zeros((DimY,DimX))
    
    #Generating an array of 1s in only the pixels within the annulus and zeros elsewhere
    AnnulusPixels = np.where(b_array < b_out, 1, ZerosArray)
    AnnulusPixels = np.where(b_array < b_in, 0, AnnulusPixels)
    #Making Sure Hot Pixels aren't counted
    AnnulusPixels = np.where(HotsArray > 0, 0, AnnulusPixels)
    
    #Determining the number of pixels in the annulus
    NumberPixels = np.sum(AnnulusPixels)
    
    #Summing all the pixel values reamaining in the annulus
    AnnulusSum = np.where(AnnulusPixels > 0, image_data, ZerosArray)
    AnnulusSum = np.sum(AnnulusSum)
    
    #Finding the Variance of the sum by summing all the squared errors of the pixels used
    Errors = np.where(AnnulusPixels > 0, ErrorFrame, ZerosArray)
    SquaredErrors = Errors**2
    AnnulusSumVariance = np.sum(SquaredErrors)
    
    #Finding the Error of the sum as the square root of the variance of the sum
    AnnulusSumError = math.sqrt(AnnulusSumVariance)
    
    #Determining the Annulus Mean and Error of mean by dividing the sum values by the number of pixels used
    AnnulusMean = AnnulusSum/NumberPixels
    AnnulusErrorOfMean = AnnulusSumError/NumberPixels
    #AnnulusErrorOfMean = AnnulusSumError/math.sqrt(NumberPixels)
    
    return AnnulusSum, AnnulusSumError, AnnulusMean, AnnulusErrorOfMean

def GetEllipticalAnnulusSumMeanAndErrorsFGV(image_data,SumSignalFrame,b_out,b_in,b_array,ErrorFrame,HotsArray,DimX,DimY,Gain,FGV):
    #FGV: pixel-by-pixel fractional gain variation
    #ErrorFrame: Frame of counting statistics errors (ususaly Gehrels error)
    #SumSignalFrame: Offset corrected (not dark subtracted) sum signal frame in units of e-
    
    G = FGV
    Sigma_g_FGV = G*Gain
    
    ZerosArray = np.zeros((DimY,DimX))
    
    #Generating an array of 1s in only the pixels within the annulus and zeros elsewhere
    AnnulusPixels = np.where(b_array < b_out, 1, ZerosArray)
    AnnulusPixels = np.where(b_array < b_in, 0, AnnulusPixels)
    #Making Sure Hot Pixels aren't counted
    AnnulusPixels = np.where(HotsArray > 0, 0, AnnulusPixels)
    
    #Determining the number of pixels in the annulus
    NumberPixels = np.sum(AnnulusPixels)
    
    #Summing all the pixel values reamaining in the annulus
    AnnulusSum = np.where(AnnulusPixels > 0, image_data, ZerosArray)
    AnnulusSum = np.sum(AnnulusSum)
    
    #Finding the Variance of the sum by summing all the squared errors of the pixels used
    GehrelsErrors = np.where(AnnulusPixels > 0, ErrorFrame, ZerosArray)
    SquaredGehrelsErrors = GehrelsErrors**2
    
    #Need the sum signal in units of ADU to find the variance from FGV
    SumSignalFrameADU = SumSignalFrame/Gain
    
    VarianceFromFGV = np.where(AnnulusPixels > 0, (SumSignalFrameADU*Sigma_g_FGV)**2, ZerosArray)
    
    #sigma^2(S_e-) = (geh(g*S_adu))^2 + (S_adu*sigma(g))^2
    #geh is Gehrels counting statistics error, and sigma(g) is gain error due to pixel by pixel variations
    
    AnnulusSumVariance = np.sum(SquaredGehrelsErrors) + np.sum(VarianceFromFGV)
    
    #Finding the Error of the sum as the square root of the variance of the sum
    AnnulusSumError = math.sqrt(AnnulusSumVariance)
    
    #Determining the Annulus Mean and Error of mean by dividing the sum values by the number of pixels used
    #That's what I did originaly, but I think it's actually sqrt(N) instead of N for the error
    AnnulusMean = AnnulusSum/NumberPixels
    AnnulusErrorOfMean = AnnulusSumError/NumberPixels
    #AnnulusErrorOfMean = AnnulusSumError/math.sqrt(NumberPixels)
    
    return AnnulusSum, AnnulusSumError, AnnulusMean, AnnulusErrorOfMean
    
def GetEllipticalAnnulusSumMeanAndErrorsFGV_Frac(image_data,SumSignalFrame,b_out,b_in,b_array,ErrorFrame,HotsArray,DimX,DimY,Gain,FGV,Frac):
    #FGV: pixel-by-pixel fractional gain variation
    #ErrorFrame: Frame of counting statistics errors (ususaly Gehrels error)
    #SumSignalFrame: Offset corrected (not dark subtracted) sum signal frame in units of e-
    
    G = FGV
    Sigma_g_FGV = G*Gain
    
    ZerosArray = np.zeros((DimY,DimX))
    
    #Generating an array of 1s in only the pixels within the annulus and zeros elsewhere
    AnnulusPixels = np.where(b_array < b_out, 1, ZerosArray)
    AnnulusPixels = np.where(b_array < b_in, 0, AnnulusPixels)
    #Making Sure Hot Pixels aren't counted
    AnnulusPixels = np.where(HotsArray > 0, 0, AnnulusPixels)
    
    #Determining the number of pixels in the annulus
    NumberPixels = np.sum(AnnulusPixels)
    
    #Summing all the pixel values reamaining in the annulus
    AnnulusSum = np.where(AnnulusPixels > 0, image_data, ZerosArray)
    AnnulusSum = np.sum(AnnulusSum)
    
    #Finding the Variance of the sum by summing all the squared errors of the pixels used
    GehrelsErrors = np.where(AnnulusPixels > 0, ErrorFrame, ZerosArray)
    SquaredGehrelsErrors = GehrelsErrors**2
    
    #Need the sum signal in units of ADU to find the variance from FGV
    SumSignalFrameADU = SumSignalFrame/Gain
    
    VarianceFromFGV = np.where(AnnulusPixels > 0, (SumSignalFrameADU*Sigma_g_FGV)**2, ZerosArray)
    
    #sigma^2(S_e-) = (geh(g*S_adu))^2 + (S_adu*sigma(g))^2
    #geh is Gehrels counting statistics error, and sigma(g) is gain error due to pixel by pixel variations
    
    AnnulusMean = AnnulusSum/NumberPixels
    
    AnnulusSumVariance = np.sum(SquaredGehrelsErrors) + np.sum(VarianceFromFGV) + (NumberPixels**2)*(Frac**2)*(AnnulusMean**2)
    
    #Finding the Error of the sum as the square root of the variance of the sum
    AnnulusSumError = math.sqrt(AnnulusSumVariance)
    
    #Determining the Annulus Mean and Error of mean by dividing the sum values by the number of pixels used
    #That's what I did originaly, but I think it's actually sqrt(N) instead of N for the error
    #AnnulusMean = AnnulusSum/NumberPixels
    AnnulusErrorOfMean = AnnulusSumError/NumberPixels
    #AnnulusErrorOfMean = AnnulusSumError/math.sqrt(NumberPixels)
    
    return AnnulusSum, AnnulusSumError, AnnulusMean, AnnulusErrorOfMean

def MakeAllROIPixelToHistogram(ImagesList,ROI,HotsArray,MeanDarkFrame):
    
    for i in range(len(ImagesList)):
        #for all images in list of images
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        #DarkSubtracting
        image_data = image_data - MeanDarkFrame

        ToHistogramThisFrame = np.array(NoHotsToHistogram(image_data,ROI,HotsArray))

        if i == 0:
            ToHistogram = ToHistogramThisFrame
        else:
            ToHistogram = np.append(ToHistogram,ToHistogramThisFrame)

    return ToHistogram
    
def MakeAllROIPixelToHistogramElectrons(ImagesList,ROI,HotsArray,MeanDarkFrame,Gain,Offset):
    
    darkframe_e = (MeanDarkFrame-Offset)*Gain
    
    for i in range(len(ImagesList)):
        #for all images in list of images
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        #DarkSubtracting
        #image_data = image_data - MeanDarkFrame
        
        image_data_e = (image_data-Offset)*Gain
        DarkSubImageDataElectrons = image_data_e - darkframe_e
        
        ToHistogramThisFrame = np.array(NoHotsToHistogram(DarkSubImageDataElectrons,ROI,HotsArray))

        if i == 0:
            ToHistogram = ToHistogramThisFrame
        else:
            ToHistogram = np.append(ToHistogram,ToHistogramThisFrame)

    return ToHistogram
    
def ROItoPoints(ROI):
    
    x1 = ROI[0]
    y1 = ROI[2]

    x2 = ROI[1]
    y2 = ROI[2]
    
    x3 = ROI[1]
    y3 = ROI[3]
    
    x4 = ROI[0]
    y4 = ROI[3]
    
    x = [x1,x2,x3,x4]
    y = [y1,y2,y3,y4]
    
    return x, y

def MakeTileFrames(ImagesList,NumFrames,SumSignalFrame,DimROI,HotsArray,MeanDarkFrame,Gain,Offset,DimX,DimY,MaxROIsignal,IntegrationRegionSize,thres):
    #SignalFrame: The frame from where the ROI mean signal is checked against MaxROIsignal to determine if the ROI is Usable.
    
    DimXTile = math.floor(DimX/DimROI)
    DimYTile = math.floor(DimY/DimROI)
    
    TiledMeanSignalFrame = np.zeros((DimY,DimX))
    TiledMeanBackgroundSignalFrame = np.zeros((DimY,DimX))
    TiledMeanBackgroundSignalFrameADU = np.zeros((DimY,DimX))
    TiledSumSignalFrame = np.zeros((DimY,DimX))
    TiledSumBackgroundSignalFrame = np.zeros((DimY,DimX))
    TiledSumSignalAboveBackgroundFrame = np.zeros((DimY,DimX))
    
    for TileX in tqdm(range(DimXTile)):
        for TileY in range(DimYTile):
            
            TileROI = [TileX*DimROI,(TileX + 1)*DimROI,TileY*DimROI,(TileY + 1)*DimROI]
            x1 = TileROI[0]
            x2 = TileROI[1]
            y1 = TileROI[2]
            y2 = TileROI[3]
            
            #Check if Tile is Useable
            ROIMeanSignal = NoHotsFrameROIMean(SumSignalFrame,TileROI,HotsArray)
            if ROIMeanSignal > MaxROIsignal:
                continue
            
            #Making the ToHistograms
            AllPixelROIToHistogramElectrons = MakeAllROIPixelToHistogramElectrons(ImagesList,TileROI,HotsArray,MeanDarkFrame,Gain,Offset)
            NoEventPixelROIToHistogramElectrons = NoEventPixelHistogramStandardElectrons(ImagesList,TileROI,MeanDarkFrame,HotsArray,DimX,DimY,IntegrationRegionSize,thres,Gain,Offset)
            
            ROIMeanPixelSignal = np.mean(AllPixelROIToHistogramElectrons)
            ROINoEventMeanPixelSignal = np.mean(NoEventPixelROIToHistogramElectrons)

            ROIExpectedDarkSubSumFramePixelSignal = ROIMeanPixelSignal*NumFrames
            ROIExpectedNoEventDarkSubSumFramePixelSignal = ROINoEventMeanPixelSignal*NumFrames
            
            ROIExpectedSignalAboveBackground = ROIExpectedDarkSubSumFramePixelSignal-ROIExpectedNoEventDarkSubSumFramePixelSignal
            
            #make frame to add to FinalOutputFrame
            AddToFrame = np.zeros((DimY,DimX)) + 1.0
            #Zeroing Everything that isn't the TileROI
            AddToFrame[0:y1,:] = 0
            AddToFrame[y2:DimY,:] = 0
            AddToFrame[:,0:x1] = 0
            AddToFrame[:,x2:DimX] = 0
            
            TiledMeanSignalFrame = TiledSumSignalFrame + AddToFrame*ROIMeanPixelSignal
            TiledMeanBackgroundSignalFrame = TiledMeanBackgroundSignalFrame + AddToFrame*ROINoEventMeanPixelSignal
            TiledMeanBackgroundSignalFrameADU = TiledMeanBackgroundSignalFrame + AddToFrame*(ROINoEventMeanPixelSignal/Gain)
            
            TiledSumSignalFrame = TiledSumSignalFrame + AddToFrame*ROIExpectedDarkSubSumFramePixelSignal
            TiledSumBackgroundSignalFrame = TiledSumBackgroundSignalFrame + AddToFrame*ROIExpectedNoEventDarkSubSumFramePixelSignal
            
            TiledSumSignalAboveBackgroundFrame = TiledSumSignalAboveBackgroundFrame + AddToFrame*ROIExpectedSignalAboveBackground
            
            
    return TiledMeanSignalFrame, TiledMeanBackgroundSignalFrame, TiledMeanBackgroundSignalFrameADU, TiledSumSignalFrame, TiledSumBackgroundSignalFrame, TiledSumSignalAboveBackgroundFrame


#<QE Calc>--------------------------------------------------------------------

def FindElectronFluxSensor(MeasuredSensorElectronSignal, ErrMeasuredSensorElectronSignal, ExpTime, NumFrames):
    
    ElectronFluxSensor = MeasuredSensorElectronSignal/(NumFrames*ExpTime)
    ErrElectronFluxSensor = ErrMeasuredSensorElectronSignal/(NumFrames*ExpTime)
    
    return ElectronFluxSensor, ErrElectronFluxSensor

def FindFractionPDSignal(MeasuredSensorElectronSignal, ErrMeasuredSensorElectronSignal,HypotheticalSensorSignal,ErrHypotheticalSensorSignal):
    
    FractionPDSignal = MeasuredSensorElectronSignal/HypotheticalSensorSignal
    ErrFractionPDSignal = np.sqrt(((ErrMeasuredSensorElectronSignal/HypotheticalSensorSignal)**2)+(((-1*MeasuredSensorElectronSignal*ErrHypotheticalSensorSignal/(HypotheticalSensorSignal**2)))**2))
    
    return FractionPDSignal, ErrFractionPDSignal


def FindSensorQE(E_eV,W,ExpTime,NumFrames,MeasuredSensorElectronSignal,ErrMeasuredSensorElectronSignal,PhotonFluxPD,ErrPhotonFluxPD,HypotheticalSensorSignal,ErrHypotheticalSensorSignal):
    
    ElectronFluxSensor, ErrElectronFluxSensor = FindElectronFluxSensor(MeasuredSensorElectronSignal, ErrMeasuredSensorElectronSignal, ExpTime, NumFrames)
    
    FractionPDSignal, ErrFractionPDSignal = FindFractionPDSignal(MeasuredSensorElectronSignal, ErrMeasuredSensorElectronSignal,HypotheticalSensorSignal,ErrHypotheticalSensorSignal)
    
    PhotonFluxSensor = FractionPDSignal*PhotonFluxPD
    ErrPhotonFluxSensor = np.sqrt(((PhotonFluxPD*ErrFractionPDSignal)**2)+((FractionPDSignal*ErrPhotonFluxPD)**2))
    
    EeV_W = E_eV/W
    
    QE = ElectronFluxSensor/(EeV_W*PhotonFluxSensor)
    
    QEvar1 = ((1/(EeV_W*PhotonFluxSensor))*ErrElectronFluxSensor)**2
    QEvar2 = (((-1*ElectronFluxSensor)/(EeV_W*(PhotonFluxSensor**2)))*ErrPhotonFluxSensor)**2
    
    ErrQE = np.sqrt(QEvar1+ QEvar2)
    
    return QE, ErrQE


#<new>



#</new>



#<Incomplete>------------------------------------------------------------------

def EventFinderSteveOld(ImageData, MeanDarkFrame, hot_pos, thres, thresp, verbose=False, diagonals=True):
    
    ep, ex, ey, shp = [], [], [], []
    nx, ny = 1936, 1096
    dx, dy = 2, 2
    
    hotPixels = hot_pos
    
    phot = ImageData - MeanDarkFrame
    phot[phot < 0] = 0
    for ii in range(0, len(hotPixels)):
        phot[int(hotPixels[int(ii)][0])][int(hotPixels[int(ii)][1])] = 0
    f1 = np.reshape(phot, nx*ny)
    q = np.argsort(f1)[::-1]
    j = 0
    above_thres = True
    while above_thres:
        if (j >= 2121856):
            above_thres = False
            break
        i = q[j]
        j += 1
        if (f1[i] >= thres):
            x = (i % 1936) + 1 #x coordinate in image
            y = math.floor((i / 1936) + 1) #y coordinate in image
            xR = int(math.floor(i/1936)) #x coordinate in array
            yR = int(i % 1936) #y coordinate in array
            if (xR > dx) and (xR < ny-dx-1) and (yR > dy) and (yR < nx-dy-1):
                area = phot[(xR-dx):(xR+dx+1), (yR-dy):(yR+dy+1)]
                
                p, s, v = cu.getEp(area, thresp, diag=diagonals)
                for xi in range(xR - dx, xR + dx + 1):
                    for yi in range(yR - dy, yR + dy + 1):
                        if (v[xi-xR-dx][yi-yR-dy]):
                            phot[xi, yi] = 0
                            #phot[yi, xi] = 0
                if (p > 0):
                    ep.append(p)
                    ex.append(x)
                    ey.append(y)
                    shp.append(s)
        else:
            above_thres = False
            
    ep = np.array(ep)
    ex = np.array(ex)
    ey = np.array(ey)
    shp = np.array(shp)
    
    return[ep,ex,ey,shp]
        
    


def EventFinder(ImagesList,MeanDarkFrame,StdDarkFrame,Hotpix,DimX,DimY):
    
    ZerosArray = np.zeros((DimY,DimX))
    
    for i in range(len(ImagesList)):
    #for all images in list of images
        
        image_and_metadata = ImagesList[i]
        image_data = image_and_metadata[0]
        #DarkSubtracting
        image_data = image_data - MeanDarkFrame
        
        PixelsOfInterest = np.where(image_data > 3*StdDarkFrame, 1, ZerosArray)
        #An array with 1's in every pixel position with a count greater than 3 stdev dark and  0's elsewhere
        OnlyEventPixels = np.where(image_data > 3*StdDarkFrame, image_data, ZerosArray)
        #The image array but everything but pixels definitely part of an event are zeroed
        
        
        
        
        #make the hot pixels negitve 100 million so that the sum of any group with a hotpixel is allways less than 0 (negitive)

'''
datadirectory = '2022-02-15 Experiment Run\Beamline\Darks/'
filepattern = 'Dark_300*.FTS'
SaveDirectory = '2022-02-15 Experiment Run\MasterDarkFrames/'
SaveName = 'MeanDarkFrame_'
DimX = 1096
DimY = 1936
hotsig = 5.0
hotfrac = 0.1
MeanDarkFrame = MakeMasterDarkFrames(datadirectory,filepattern,DimX,DimY,hotsig,hotfrac,SaveDirectory,SaveName,SaveHotPix = True, SaveAsFits = True,SaveAsCSV = True)
'''