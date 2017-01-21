# Import libraries
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot
import numpy as np
import matplotlib.pyplot as plt

# Load VGG model, trained on ImageNet
base_model = VGG16(weights='imagenet')
#base_model = VGG16(weights=None)
print "VGG model loaded"

# Get weights layer conv1 of block1
weights = base_model.get_layer('block1_conv1').get_weights()
print 'Weights shape: '+str(weights[0].shape)
print 'Weights at [0][:, :, :, 1]: '+str(weights[0][:, :, :, 1])

# Define control variables
x_grid = 8
y_grid = 8
step = 3
num_filter = 0

# Initialize output and result matrix
output = np.empty((weights[0].shape[0],weights[0].shape[1],weights[0].shape[2]))
result = np.empty((weights[0].shape[0]*x_grid,weights[0].shape[1]*y_grid,weights[0].shape[2]))

print 'Output shape: '+str(output.shape)
print 'Results shape: '+str(result.shape)

# Loop to generate results matrix with filter weights
for x_dr in range(8):
	for y_dr in range(8):
		output = weights[0][:,:,:,num_filter]
		num_filter = num_filter + 1 
		result[x_dr*step:x_dr*step+step,y_dr*step:y_dr*step+step,:]=output

print 'Result matrix shape: '+str(result.shape)

# Interpolation flag defines can blur image, be careful
plt.imshow(result, interpolation='none')
plt.show()	
