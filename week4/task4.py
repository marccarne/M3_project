# -*- coding: utf-8 -*-

#Import libraries
import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt

def lod_model(model, layer_name):
        #load VGG model
        base_model = VGG16(weights=model)
        #visalize topology in an image
        #crop the model up to a certain layer
        model = Model(input=base_model.input, output=base_model.get_layer(layer_name).output)
        return model

def read_data(train_files, train_labels, test_files, test_labels):
    
    train_images_filenames = cPickle.load(open(train_files,'r'))
    test_images_filenames = cPickle.load(open(test_files,'r'))
    train_labels = cPickle.load(open(train_labels,'r'))
    test_labels = cPickle.load(open(test_labels,'r'))

    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)
    
    return (train_images_filenames, train_labels, test_images_filenames, test_labels)

def extract_train_features(train_images_filenames, train_labels, detector):
        
    Train_descriptors = []
    Train_label_per_descriptor = []

    for i in range(len(train_images_filenames)):
        
        kpt,des= compute_feature(train_images_filenames[i], detector)
        
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
        print str(len(kpt))+' extracted keypoints and descriptors'


    return (Train_descriptors, Train_label_per_descriptor)

def compute_codebook(codebook_name, Train_descriptors, k=512):

    # Transform everything to numpy arrays
    size_descriptors=len(Train_descriptors[0][1])
    D=np.zeros(((len(Train_descriptors[0])*len(Train_descriptors)),size_descriptors),dtype=np.uint8)
    startingpoint=0
    for i in range(len(Train_descriptors)):
	for j in range(len(Train_descriptors[i])):
        	D[startingpoint:startingpoint+len(Train_descriptors[i][j])]=Train_descriptors[i][j]
        	startingpoint+=len(Train_descriptors[i][j])
        
    print 'Computing kmeans with '+str(k)+' centroids'
    init=time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4)
    codebook.fit(D)
    cPickle.dump(codebook, open(codebook_name, "wb"))
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return codebook

def BoW(codebook, descriptors):
    #cluster_centers_ : array, [n_clusters, n_features]
    init=time.time()
    #visual_words=np.zeros((len(descriptors),k),dtype=np.float32)
    k = codebook.cluster_centers_.shape[0]
    print k
    visual_words=np.zeros((len(descriptors),k),dtype=np.float32)
    for i in xrange(len(descriptors)):
        words=codebook.predict(descriptors[i])
        visual_words[i,:]=np.bincount(words,minlength=k)

    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    
    return visual_words

def train_SVM(visual_words, train_labels, toStore= False, filename = ""):

    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, train_labels)
    print 'Done!'
    
    if toStore:
        cPickle.dump(clf, open(filename, "wb"))
    return clf, stdSlr
    
def compute_feature(image_filename, detector):

    print 'Reading image '+ image_filename
    img = image.load_img(image_filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = detector.predict(x)
    
    features = features.reshape(features.shape[1:])
    to_grid = features[:,:,3]
    print to_grid.shape
    print features.shape
    #dense=cv2.FeatureDetector_create("Dense")
    
    step_size = 1
    kpt = [cv2.KeyPoint(x, y, step_size) for y in range(0, to_grid.shape[0], step_size) for x in range(0, to_grid.shape[1], step_size)]

    #kpt=dense.detect(to_grid)
    des = []
    for kp in kpt:
	kp=kp.pt
	des.append(features[kp[1],kp[0],:])
    return kpt,des
    
def evaluate_test(test_images_filenames, test_labels, codebook, stdSlr, detector, clf):
    k = codebook.cluster_centers_.shape[0]
    # get all the test data and predict their labels
    visual_words_test=np.zeros((len(test_images_filenames), k),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        #extract features for a single image
        kpt,des= compute_feature(test_images_filenames[i], detector)
        #extract VW for a single image
        words=codebook.predict(des)
        visual_words_test[i,:]=np.bincount(words,minlength=k)
    
          
    return 100*clf.score(stdSlr.transform(visual_words_test), test_labels)
    
def core(k):

    start = time.time()
    
    # read the train and test files
    
    train_images_filenames, train_labels, test_images_filenames, test_labels = read_data('dat_files/train_images_filenames.dat', 'dat_files/train_labels.dat', 'dat_files/test_images_filenames.dat', 'dat_files/test_labels.dat')
    
    # create the SIFT detector object
    
    detector = lod_model('imagenet', 'block3_conv3')
    
    # read the just 30 train images per class
    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    Train_descriptors, Train_label_per_descriptor = extract_train_features(train_images_filenames, train_labels, detector)
    
    #as default k=512
    codebook = compute_codebook("codebook.dat", Train_descriptors, k= 50)
    
    
    visual_words = BoW(codebook, Train_descriptors)
    
    # Train a linear SVM classifier
    
    clf, stdSlr = train_SVM(visual_words, train_labels)
    
    accuracy = evaluate_test(test_images_filenames, test_labels, codebook, stdSlr, detector, clf)
    print 'Final accuracy: ' + str(accuracy)
    
    end=time.time()
    print 'Done in '+str(end-start)+' secs.'
    
    out = 'Accuracy:'+str(accuracy)+' k: '+str(k)+'. \n'
    
    fo = open('accuracies.txt' ,'a')
    fo.write(out)
    fo.close()
    
    return
    ## 49.56% in 285 secs.
    
#------------------ Main function ------------------ #
if __name__ == "__main__":
	cbook_size_k = 64
        
	print cbook_size_k
	core(cbook_size_k)    
        
	print 'Overall finished'

