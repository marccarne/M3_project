# Import libraries
import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from yael import ynumpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


"""
Function: read_data
Description: read the train and test files
Input: train_files, train_labels, test_files, test_labels
Output: train_images_filenames, train_labels, test_images_filenames, test_labels
"""
def read_data(train_files, train_labels, test_files, test_labels):
    
    train_images_filenames = cPickle.load(open(train_files,'r'))
    test_images_filenames = cPickle.load(open(test_files,'r'))
    train_labels = cPickle.load(open(train_labels,'r'))
    test_labels = cPickle.load(open(test_labels,'r'))

    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)
    
    return (train_images_filenames, train_labels, test_images_filenames, test_labels)


"""
Function: extract_features
Description: extract SIFT keypoints and descriptors
Input: train_images_filenames, train_labels, detector
Output: D, Train_descriptors, Train_label_per_descriptor
"""
def extract_train_features(train_images_filenames, train_labels, detector):

	Train_descriptors = []
	Train_label_per_descriptor = []

	# Extract SIFT keypoints and descriptors
	for i in range(len(train_images_filenames)):
		# Compute DENSE SIFT
		kpt,des= compute_dense(train_images_filenames[i], detector) 		
		Train_descriptors.append(des)
		Train_label_per_descriptor.append(train_labels[i])
		print str(len(kpt))+' extracted keypoints and descriptors'

	size_descriptors=Train_descriptors[0].shape[1]
	D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
	startingpoint=0
	for i in range(len(Train_descriptors)):
		D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
		startingpoint+=len(Train_descriptors[i])

	return (D, Train_descriptors, Train_label_per_descriptor)


"""
Function: compute_dense
Description: compute dense SIFT
Input: timage_filename, detector
Output: kp,des
"""
def compute_dense(image_filename, detector):

    print 'Reading image '+ str(image_filename)	
    ima=cv2.imread(image_filename)
    gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
			
    dense=cv2.FeatureDetector_create("Dense")
    kp=dense.detect(gray)
    kp,des=detector.compute(gray,kp)
    
    return kp,des


"""
Function: compute_fisher_vectors
Description: compute fisher vectors
We will use INRIA's yael library
Input: k, D, Train_descriptors
Output: fisher, gmm
"""
def compute_fisher_vectors(k, D, Train_descriptors):

	print 'Computing gmm with '+str(k)+' centroids'
	init=time.time()
	gmm = ynumpy.gmm_learn(np.float32(D), k)
	end=time.time()
	print 'Done in '+str(end-init)+' secs.'

	init=time.time()
	fisher=np.zeros((len(Train_descriptors),k*128*2),dtype=np.float32)
	for i in xrange(len(Train_descriptors)):
		fisher[i,:]= ynumpy.fisher(gmm, Train_descriptors[i], include = ['mu','sigma'])

	end=time.time()
	print 'Done in '+str(end-init)+' secs.'

	return (fisher, gmm)


"""
Function: train_SVM
Description: train SVM classifier
Input: visual_words, train_labels
Output: clf, stdSlr
"""
def train_SVM(visual_words, train_labels):
	# Train a linear SVM classifier
	stdSlr = StandardScaler().fit(visual_words)
	D_scaled = stdSlr.transform(visual_words)
	print 'Training the SVM classifier...'
	clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, train_labels)
	print 'Done!'

	return (clf, stdSlr)


"""
Function: evaluate_test
Description: get all the test data and predict their labels
Input: clf, stdSl, test_images_filenames, pca
Output: accuracy
"""
def evaluate_test(clf, stdSl, test_images_filenames, k, detector, gmm):

	fisher_test=np.zeros((len(test_images_filenames),k*128*2),dtype=np.float32)
	for i in range(len(test_images_filenames)):
		filename=test_images_filenames[i]
		print 'Reading image '+filename
		kpt,des= compute_dense(test_images_filenames[i], detector) 
		fisher_test[i,:]=ynumpy.fisher(gmm, des, include = ['mu','sigma'])

	accuracy = 100*clf.score(stdSl.transform(fisher_test), test_labels)

	return accuracy


#------------------ Main function ------------------ #
# NOTE: minimum accuracy accepted 61.71% (in 251 secs)
if __name__ == "__main__":

	start = time.time()

	# Paths of dat files
	train_im 	= 'dat_files/train_images_filenames.dat'
	train_lbls 	= 'dat_files/train_labels.dat'
	test_im	 	= 'dat_files/test_images_filenames.dat'
	test_lbls  	= 'dat_files/test_labels.dat'

	# Read the train and test files
	train_images_filenames, train_labels, test_images_filenames, test_labels = read_data(train_im, train_lbls, test_im, test_lbls)

	# Create the SIFT detector object
	detector = cv2.SIFT(nfeatures=100)

	# Apply PCA
	# n_components = 32
	# pca = PCA(n_components)

	# Extract train feactures
	D, Train_descriptors, Train_label_per_descriptor = extract_train_features(train_images_filenames, train_labels, detector)
	
	# pca.fit(D)
	# D_pca = pca.transform(D)

	# Define number of clusters
	k = 32

	# Compute fisher vectors
	fisher, gmm = compute_fisher_vectors(k, D, Train_descriptors)

	# L2 normalization over the fisher vectors
	# fisher_normL2 = normalize(x[:,np.newaxis], axis=0).ravel()

	# Train a linear SVM classifier
	clf, stdSl = train_SVM(fisher, Train_label_per_descriptor)

	# Evaluate test	
	accuracy = evaluate_test(clf, stdSl, test_images_filenames, k, detector, gmm)

	print 'Accuracy: '+str(accuracy)+'% k: '+str(k)

	end=time.time()
	print 'Done in '+str(end-start)+' secs.'


