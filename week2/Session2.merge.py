# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 18:31:05 2016

@author: Gonza
"""
#This is a version to run straight on the cluster by using the job101.sh bash script.
#I don't know much about bash scripts so if you check the .sh file you will find
#it to be really short, just a call for this .py.
#For execution, the session2.merge.py, job101.sh and all .dat files must be within
#a same folder (i named it 'program'), and at the same time, 'program' must be
#within another folder in the /home directory (I used the name 'Session2').
#The Database must be in the cluster so you pass it with scp and put it in the 
# '/home' directory.
#This version will launch the BoVW for as many different 'k' as you set in the
#variable 'cbook_size_k', down in the main function. For each k, it will compute
# accuracy and store it in a .txt file called 'accuracies' in the directory where
#'Session2.merge.py' is.
#Could not yet make it do the ROC and AUC curve.  

#Import libraries
import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster


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
    size_descriptors=Train_descriptors[0].shape[1]
    D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
    startingpoint=0
    for i in range(len(Train_descriptors)):
        D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
        startingpoint+=len(Train_descriptors[i])
        
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
    ima=cv2.imread(image_filename)
    gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    kpt,des=detector.detectAndCompute(gray,None)
    
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
    
    train_images_filenames, train_labels, test_images_filenames, test_labels = read_data('train_images_filenames.dat', 'train_labels.dat', 'test_images_filenames.dat', 'test_labels.dat')
    
    # create the SIFT detector object
    
    detector = cv2.SIFT(nfeatures=100)
    
    # read the just 30 train images per class
    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    Train_descriptors, Train_label_per_descriptor = extract_train_features(train_images_filenames, train_labels, detector)
    
    #as default k=512
    codebook = compute_codebook("codebook.dat", Train_descriptors, k= 50)
    
    
    visual_words = BoW(codebook, Train_descriptors)
    
    # Train a linear SVM classifier
    
    clf, stdSlr = train_SVM(visual_words, train_labels)
    
    accuracy, roc_value = evaluate_test(test_images_filenames, test_labels, codebook, stdSlr, detector, clf)
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
    cbook_size_k = np.arange(64,2048,64)
        
    for i in range(len(cbook_size_k)):
        
        print cbook_size_k[i]
        core(cbook_size_k[i])
        print 'i='+str(i)+'.'    
        
    print 'Overall finished'

