"""
Modified By: Tyler Chase
Date: 04/08/2018
Parsing code for DICOMS and contour files
"""

import dicom
from dicom.errors import InvalidDicomError

import numpy as np
from PIL import Image, ImageDraw

import pandas
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import random

# Permute both X and y keeping same relative ordering
def permute(X, y, label):
    """Permute X, y, and their labels in a consistent manor

    :param X: input X to model
    :param y: output y of model
    :param label: origin label of X and y
    :return: permuted X, permuted y, permuted label
    """
    SEED = 455
    random.seed(SEED)
    N = len(y)
    indices = np.arange(N)
    random.shuffle(indices)
    X_shuf = []
    y_shuf = []
    label_shuf = []
    for ind in indices:
        X_shuf.append(X[ind])
        y_shuf.append(y[ind])
        label_shuf.append(label[ind])
    return X_shuf, y_shuf, label_shuf

# Function for permuting and splitting data into training, developement, and test
def batchify(X, y, labels, batch_size = 8, epochs = 2): 
    """Load batches of data for 2-D deep learning model
    here we print information about the batch for unit testing instead of feeding into a model
    
    :param X: input X to model
    :param y: output y of model
    :param label: origin label of X and y
    :param batch_size: number of data points in each batch
    :param epochs: number of times we pass over the dataset when training
    """
    # Permute data points
    X, y, labels  = permute(X, y, labels)
    
    totalNum = len(X)
    for epoch in range(epochs):
        for i, j in enumerate(np.arange(0, totalNum, batch_size)):
            X_batch = np.array(X[j:j+batch_size])
            y_batch = np.array(y[j:j+batch_size])
            labels_batch = np.array(labels[j:j+batch_size])
            # Unit tests of batch
            print('epoch:', epoch+1)
            print('batch:', i+1)
            print('X_batch shape:', np.shape(X_batch))
            print('y_batch shape:', np.shape(y_batch))
            for i in range(len(labels_batch)):
                plotName = 'DCM: ' + labels_batch[i]['DCM_id'] + \
                               ',     Contours: ' + labels_batch[i]['contour_id'] + \
                               ',     FileNumber: ' + labels_batch[i]['file_num']
                print(plotName)
            print()

class Pipe:
    def __init__(self, dataAddress):
        self.dcmArray, self.contourArray, self.labels = self.pipeToNumpy(dataAddress) 

    def parse_contour_file(self, filename):
        """Parse the given contour filename

        :param filename: filepath to the contourfile to parse
        :return: list of tuples holding x, y coordinates of the contour
        """

        coords_lst = []

        with open(filename, 'r') as infile:
            for line in infile:
                coords = line.strip().split()

                x_coord = float(coords[0])
                y_coord = float(coords[1])
                coords_lst.append((x_coord, y_coord))

        return coords_lst


    def parse_dicom_file(self, filename):
        """Parse the given DICOM filename

        :param filename: filepath to the DICOM file to parse
        :return: dictionary with DICOM image data
        """

        try:
            dcm = dicom.read_file(filename)
            dcm_image = dcm.pixel_array

            try:
                intercept = dcm.RescaleIntercept
            except AttributeError:
                intercept = 0.0
            try:
                slope = dcm.RescaleSlope
            except AttributeError:
                slope = 0.0

            if intercept != 0.0 and slope != 0.0:
                dcm_image = dcm_image*slope + intercept
            dcm_dict = {'pixel_data' : dcm_image}
            return dcm_dict
        except InvalidDicomError:
            return None


    def poly_to_mask(self, polygon, width, height):
        """Convert polygon to mask

        :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
         in units of pixels
        :param width: scalar image width
        :param height: scalar image height
        :return: Boolean mask of shape (height, width)
        """

        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(width, height), color=0)
        ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
        mask = np.array(img).astype(bool)
        return mask

    def processDcmContour(self, fileDcm, fileContour):
        """Process dcm and contour file, returning a DICOM image and binary mask of contour
    
        :param fileDcm: address of dcm file
        :param fileContour: address of contour file
        :return: DICOM image and binary mask of contour
        """
        # Process DCM
        dcmDict = self.parse_dicom_file(fileDcm)    
        dicomImage = dcmDict['pixel_data']
    
        # Process Contour
        coordsList = self.parse_contour_file(fileContour)
        height, width = np.shape(dicomImage)
        contourMask = self.poly_to_mask(coordsList, height, width)
    
        return dicomImage, contourMask
    
    def plotDcmContour(self, title, dcmImage, contour):
        """Visualize a scan image cross-section and it's i-contour

        :param dcm: dictionary with DICOM image data
        :param contour: boolean mask of i-contour
        """
        #dcmImage = dcm['pixel_data']       
        f, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (30, 14))
        f.suptitle(title, fontsize = 40)
        ax1.imshow(dcmImage, cmap = plt.cm.gray)
        ax1.imshow(contour, cmap = "bwr", alpha = 0.2, vmin = -1, vmax = 1)
        ax1.set_title('Contour Highlighted', fontsize = 30)
        ax2.imshow(dcmImage, cmap = plt.cm.gray)
        ax2.set_title('Image', fontsize = 30)
        plt.show()
        
    def plotPipedData(self):
        """Visualize all scan image cross-sections with their respective i-contour
        """       
        for i in range(np.shape(self.dcmArray)[0]):
            plotName = 'DCM: ' + self.labels[i]['DCM_id'] + ',     Contours: ' + self.labels[i]['contour_id'] + \
                       ',     FileNumber: ' + self.labels[i]['file_num']
            self.plotDcmContour(plotName, self.dcmArray[i], self.contourArray[i])        
    
    def determineContourNumbers(self, folder):
        """Return array of contour file numbers from a folder of contours
    
        :param folder: address for a folder containing contours
        :return: array of contour file numbers
        """
        files = glob.glob(folder + '/IM-0001-*-icontour-manual.txt')
        files_2 = []
        for i in files:
            temp_1 = re.sub(folder + '/IM-0001-0*', '', i)
            temp_2 = re.sub('-icontour-manual.txt', '', temp_1)
            files_2.append(int(temp_2))
            files_2.sort()                     
        return files_2

    def pipeToNumpy(self, dataAddress):
        """Process all DCMs and Contours to a numpy arrays (ordered by Index ID, and Numerically ordered)
    
        :param dataAddress: address of the data folder downloaded from email
        :return: numpy array of all DICOM images, numpy array of all contour binary masks, 
                 and a dictionary containing file label data
        """
        # Load CSV linking dcm files and contours
        link = pandas.read_csv(dataAddress + '/link.csv')
    
        # Process Contours and DCM
        contours = []
        DCMs = []
        labels = []
        for _, row in link.iterrows():
            fileNumbers = self.determineContourNumbers(dataAddress + '/contourfiles/' + row['original_id'] + '/i-contours')
            for j in fileNumbers:
                numStr = "%04d" % j 
                fileNameContour = dataAddress + '/contourfiles/' + row['original_id'] + '/i-contours' +  \
                                  '/IM-0001-' + numStr + '-icontour-manual.txt'
                fileNameDcm = dataAddress + '/dicoms/' + row['patient_id'] + '/' + str(j) + '.dcm'
                dicomImage, contourMask = self.processDcmContour(fileNameDcm, fileNameContour)
                DCMs.append(dicomImage)
                contours.append(contourMask)
                labels.append({'DCM_id':row['patient_id'], 'contour_id':row['original_id'], 'file_num':str(j)})
        return np.array(DCMs), np.array(contours), labels
