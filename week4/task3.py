# Set path of site-packages to find OpenCV libraries
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

# Import libraries
import cv2
import numpy as np
import cPickle
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot


"""
Function: load_model
Description: load model
Input: model, layer_name
Output: model
"""
def load_model(model, layer_name):
	
  	# Load VGG model
	base_model = VGG16(weights=model)
	
	# Viusalize topology in an image
	# Crop the model up to a certain layer
	model = Model(input=base_model.input, output=base_model.get_layer(layer_name).output)
	return model


"""
Function: extract_train_features
Description: extract train features
Input: images_filenames, labels, textractor, n_images, is_pca
Output: d, l
"""
def extract_train_features(images_filenames, labels, textractor, n_images, is_pca):

    train_descriptors = []
    train_label_per_descriptor = []

    for i in range(len(images_filenames)):
        filename = images_filenames[i]
        if train_label_per_descriptor.count(labels[i]) < n_images:
            print 'Reading image ' + filename
            img = image.load_img(filename, target_size=(224, 224))
	    x = image.img_to_array(img)
	    x = np.expand_dims(x, axis=0)
	    x = preprocess_input(x)
	    features = textractor.predict(x)

	    train_descriptors.append(features)
            train_label_per_descriptor.append(labels[i])

    d = train_descriptors[0]
    l = np.array([train_label_per_descriptor[0]] * train_descriptors[0].shape[0])

    for i in range(1, len(train_descriptors)):
        d = np.vstack((d, train_descriptors[i]))
        l = np.hstack((l, np.array([train_label_per_descriptor[i]] * train_descriptors[i].shape[0])))

    if is_pca:
        print "Apply PCA algorithm to reduce dimensionality"
        pca.fit(d)
        dtrfm = pca.transform(d)
	d = dtrfm
    return (d, l)


"""
Function: train_svm
Description: train svm
Input: d_scaled, l, kernel_type
Output: clf_train
"""
def train_svm(d_scaled, l, kernel_type):
    
    print 'Training the SVM classifier...'
    clf_train = svm.SVC(kernel=kernel_type, C=100).fit(d_scaled, l)
    print 'Done!'
    return clf_train


"""
Function: test_classifier
Description: test classifier
Input: images_filenames, labels, cextractor, cclf, cstdslr, is_pca, cpca
Output: numcorrect
"""
def classifier(images_filenames, labels, cextractor, cclf, cstdslr, is_pca, cpca):
    
    numtestimages = np.arange(10,105,5)
    cnumcorrect = 0
    
    for i in range(len(images_filenames)):
        filename = images_filenames[i]
        img = image.load_img(filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        des = cextractor.predict(x)


	if is_pca:
            dtrfm = cpca.transform(des)

        predictions = cclf.predict(cstdslr.transform(des))
        values, counts = np.unique(predictions, return_counts=True)
        predictedclass = values[np.argmax(counts)]
        print 'image ' + filename + ' was from class ' + labels[i] + ' and was predicted ' + predictedclass
        numtestimages += 1
        
	if predictedclass == labels[i]:
            cnumcorrect += 1
    
    return cnumcorrect


"""
Function: core
Description: system core
Input: num_images
Output: 
"""
def core(num_images):
	
	# Start timer to analyse performance
	start = time.time()

	# Read the train and test files
	train_images_filenames = cPickle.load(open('dat_files/train_images_filenames.dat', 'r'))
	test_images_filenames = cPickle.load(open('dat_files/test_images_filenames.dat', 'r'))
	train_labels = cPickle.load(open('dat_files/train_labels.dat', 'r'))
	test_labels = cPickle.load(open('dat_files/test_labels.dat', 'r'))

	print 'Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels)
	print 'Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels)

	# PCA components
	pca = PCA(n_components=20)

	# Crop the model up to a last FC layer
	extractor = load_model('imagenet','fc2')

	# Extract train features
	apply_pca = False
	D, L = extract_train_features(train_images_filenames, train_labels, extractor, num_images, apply_pca)

	# Train a linear SVM classifier
	stdSlr = StandardScaler().fit(D)
	D_scaled = stdSlr.transform(D)

	kernel = 'linear'
	clf = train_svm(D_scaled, L, kernel)

	print 'Classifier trained'

	# Get all the test data and predict their labels
	numcorrect = classifier(test_images_filenames, test_labels, extractor, clf, stdSlr, apply_pca, pca)

	# Calculate accuracy
	accuracy = str(numcorrect * 100.0 / len(test_images_filenames))
	out = 'Accuracy: '+str(accuracy)+', num images used to train: '+str(num_images)+'\n'
    	fo = open('accuracies_3.txt' ,'a')
    	fo.write(out)
    	fo.close()

	# End timer to print time spent
	end = time.time()
	print 'Done in ' + str(end - start) + ' secs.'

	return

#-----------------------------------------------------#
#------------------- Main function -------------------#
#-----------------------------------------------------#
if __name__ == "__main__":
   
    # Define array of number of images
    num_images = np.arange(50,1881,50)

    # Compute core 
    for i in range(len(num_images)):
	print '\nComputing core: number of images = '+str(num_images)+', iteration = '+str(i)  
	core(num_images[i]) 
        
    print 'Overall finished'
