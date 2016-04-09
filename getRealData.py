from numpy import zeros, array, reshape
import matplotlib.pylab as plt
from scipy import ndimage as ndi
from scipy import misc
import os
from skimage import feature

def loadRealData(folder, img_size = [28,28]):
    print ">>> Loading Real World Data<<<"    
    
    data = []
    l = []

    for i in range(10):
        path = 'D:\workspace\Data\ELM\REAL\\'+folder+'\\'+str(i)+'\\'
        dirs = os.listdir(path)

        for item in dirs: 
            # data read and processing
            img = ndi.imread(path+item, True, 'L')
            img = misc.imresize(img, img_size, 'nearest')

            # denoising            
            img = ndi.median_filter(img, 2.5)
            
            #filters: blurring / smoothing
            img = ndi.filters.gaussian_filter(img, 0.25)
            #img = ndi.filters.uniform_filter(img, size = 2.5)
            
            #sharppen            
            #img_blurred = ndi.filters.gaussian_filter(img, sigma = 1)
            #img_blurred_f = ndi.filters.gaussian_filter(img_blurred, 0.5)
            #alpha = 2
            #img = img_blurred + alpha*(img_blurred - img_blurred_f)

            #edge detection
            img = feature.canny(img, sigma=0.5)
            
            data.append(reshape(img, img_size[0]*img_size[1]))
            l.append(i)

    data = array(data)
    N = data.shape[0]
    label = zeros((N, 10))
    l = array(l)
    label[range(N),l] = 1

    print ">>>Load data succeed<<<"
    print "Data shape: ", data.shape
    
    print "Example data point: "
    plt.figure(1)
    plt.imshow( reshape(data[0,:], img_size))
    plt.figure(2)
    plt.imshow( reshape(data[100,:], img_size))
    plt.figure(3)
    plt.imshow( reshape(data[250,:], img_size))
    print "Label: ", label[250,:].argmax()
    
    return data, label

loadRealData('greyscale')