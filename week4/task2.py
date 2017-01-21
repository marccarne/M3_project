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
print "VGG model loaded"

# Visalize topology in an image
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)

# Read and process one image
img_path = '../data/MIT/test/coast/art1130.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
plt.imshow(img)
plt.show()

# Define layers of block 3
block_layer = ["block3_conv1","block3_conv2","block3_conv3","block3_pool"]

# Loop to iterate layers of block 3
for layers in block_layer:
	# Crop the model up to a certain layer
	model = Model(input=base_model.input, output = base_model.get_layer(layers).output)
	print "Model loaded"

	# Get the features from images
	features = model.predict(x)
	print 'Features shape: '+features.shape

	# Remove first dimension
	output = features.reshape(features.shape[1:])
	print 'Output shape after first dimension removed: '+output.shape

 	# Initialize feature map
	feature_map = np.empty((output.shape[0],output.shape[1]))
	
	# Mean operation
	feature_map = np.mean(output, axis=2)	
	plt.imshow(feature_map)
	plt.show()

	# Max operation
	feature_map = np.max(output, axis=2)
	plt.imshow(feature_map)
	plt.show()

	# Min operation
	feature_map = np.min(output, axis=2)
	plt.imshow(feature_map)
	plt.show()

	# Median opereation
	feature_map = np.median(output, axis=2)
	plt.imshow(feature_map)
	plt.show()
