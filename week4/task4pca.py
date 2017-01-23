# Import libraries

# Set path of site-packages to find OpenCV libraries
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

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
from sklearn.decomposition import PCA

"""
Function: load_model
Description: load model that provides learning of system
Input: model, layer_name
Output: model
"""
def lod_model(model, layer_name):
    
    # Load VGG model
    base_model = VGG16(weights=model)
    
    # Visalize topology in an image
    # Crop the model up to a certain layer
    model = Model(input=base_model.input, output=base_model.get_layer(layer_name).output)
    return model


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
Function: extract_train_features
Description: read the train and test files
Input: train_images_filenames, train_labels, detector
Output: Train_descriptors, Train_label_per_descriptor
"""
def extract_train_features(train_images_filenames, train_labels, detector, apply_pca, pca):
        
    Train_descriptors = []
    Train_label_per_descriptor = []

    for i in range(len(train_images_filenames)):
        kpt,des= compute_feature(train_images_filenames[i], detector) 
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
    
    if apply_pca:
        print 'Applying pca to descriptors'
        Train_descriptors,Train_label_per_descriptor = pca_app(Train_descriptors,Train_label_per_descriptor)
    
    return (Train_descriptors,Train_label_per_descriptor)

"""
Function: apply pca
Description: use pca to lower dimensionality of train descriptors
Input: Train_descriptors,Train_label_per_descriptor
Output: d,l
"""

def pca_app(Train_descriptors,Train_label_per_descriptor):

    d = Train_descriptors[0]
    l = np.array([Train_label_per_descriptor[0]] * len(Train_descriptors[0]))

    for i in range(1, len(Train_descriptors)):
        d = np.vstack((d, Train_descriptors[i]))
        l = np.hstack((l, np.array([Train_label_per_descriptor[i]] * len(Train_descriptors[i]))))
        #print 'Level '+str(i)+' completed'
    print 'fitting pca, d:'+str(d.shape)
    pca.fit(d)
    print 'd fitted'
    d = pca.transform(d)

    return (d,l)


"""
Function: compute_codebook
Description: compute codebook
Input: codebook_name, Train_descriptors, k=512
Output: codebook
"""
def compute_codebook(codebook_name, Train_descriptors, k=512):

    # Transform everything to numpy arrays
    #size_descriptors=len(Train_descriptors[0][1])
    #D=np.zeros(((len(Train_descriptors[0])*len(Train_descriptors)),size_descriptors),dtype=np.uint8)
    #startingpoint=0
    #for i in range(len(Train_descriptors)):
    #    for j in range(len(Train_descriptors[i])):
    #        D[startingpoint:startingpoint+len(Train_descriptors[i][j])]=Train_descriptors[i][j]
    #        startingpoint+=len(Train_descriptors[i][j])
        
    print 'Computing kmeans with '+str(k)+' centroids'
    init=time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4)
    codebook.fit(Train_descriptors)
    cPickle.dump(codebook, open(codebook_name, "wb"))
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return codebook


"""
Function: BoW
Description: bag of words
Input: codebook, descriptors
Output: visual_words
"""
def BoW(codebook, descriptors, train_image_filenames):
    #cluster_centers_ : array, [n_clusters, n_features]
    init=time.time()
    #visual_words=np.zeros((len(descriptors),k),dtype=np.float32)
    k = codebook.cluster_centers_.shape[0]
    print k
    visual_words=np.zeros((len(train_image_filenames),k),dtype=np.float32)
    for i in xrange(len(train_image_filenames)):
        words=codebook.predict(descriptors[i])
        visual_words[i,:]=np.bincount(words,minlength=k)

    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    
    return visual_words


"""
Function: train_SVM
Description: train SVM
Input: visual_words, train_labels, toStore= False, filename = ""
Output: clf, stdSlr
"""
def train_SVM(visual_words, train_labels, toStore= False, filename = ""):

    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, train_labels)
    print 'Done!'
    
    if toStore:
        cPickle.dump(clf, open(filename, "wb"))
    return (clf, stdSlr)
   
 
"""
Function: compute_feature
Description: compute features
Input: image_filename, detector
Output: kpt,des
"""
def compute_feature(image_filename, detector):

    print 'Reading image '+ image_filename
    img = image.load_img(image_filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = detector.predict(x)
    
    features = features.reshape(features.shape[1:])
    to_grid = features[:,:,3]
      
    step_size = 2
    kpt = [cv2.KeyPoint(x, y, step_size) for y in range(0, to_grid.shape[0], step_size) for x in range(0, to_grid.shape[1], step_size)]

    des = []
    for kp in kpt:
        kp=kp.pt
        des.append(features[kp[1],kp[0],:])
    return (kpt, des)
 

"""
Function: evaluate_test
Description: evaluate test
Input: test_images_filenames, test_labels, codebook, stdSlr, detector, clf
Output: accuracy
"""   
def evaluate_test(test_images_filenames, test_labels, codebook, stdSlr, detector, clf, apply_pca, cpca):
    k = codebook.cluster_centers_.shape[0]
    # get all the test data and predict their labels
    visual_words_test=np.zeros((len(test_images_filenames), k),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        #extract features for a single image
        kpt,des= compute_feature(test_images_filenames[i], detector)
        if apply_pca:
            print 'Applying pca to test descriptors'
            des = cpca.transform(des)
        #extract VW for a single image
        words=codebook.predict(des)
        visual_words_test[i,:]=np.bincount(words,minlength=k)
    
          
    return 100*clf.score(stdSlr.transform(visual_words_test), test_labels)

"""
Function: core
Description: system core
Input: k
Output: 
""" 
def core(k,pca):

    start = time.time()
    
    # Paths of dat files
    train_im 	= 'dat_files/train_images_filenames.dat'
    train_lbls 	= 'dat_files/train_labels.dat'
    test_im	= 'dat_files/test_images_filenames.dat'
    test_lbls  	= 'dat_files/test_labels.dat'

    # Read the train and test files
    train_images_filenames, train_labels, test_images_filenames, test_labels = read_data(train_im, train_lbls, test_im, test_lbls)
    
    # Create detector loading model imagenet at one layer 
    detector = lod_model('imagenet', 'block3_conv3')
    
    apply_pca = True

    # Extract train features
    Train_descriptors, Train_label_per_descriptor = extract_train_features(train_images_filenames, train_labels, detector, apply_pca, pca)
    print 'Size of train_descriptors: '+str(Train_descriptors.shape)    
    # Compute coodebook
    codebook = compute_codebook("codebook_4pca.dat", Train_descriptors, k=k)
    
    # Compute bag of words
    visual_words = BoW(codebook, Train_descriptors,train_images_filenames)
    print 'size of train labels'+str(len(train_labels))
    print 'size of Train_labels'+str(Train_label_per_descriptor.shape)
    print 'size of visual words'+str(visual_words.shape)        
    # Train a linear SVM classifier 
    clf, stdSlr = train_SVM(visual_words,train_labels)
    
    # Calculate accuracy of system
    accuracy = evaluate_test(test_images_filenames, test_labels, codebook, stdSlr, detector, clf, apply_pca, pca)
    print 'Final accuracy: ' + str(accuracy)
    
    end=time.time()
    print 'Done in '+str(end-start)+' secs.'
    out = 'Accuracy:'+str(accuracy)+' k: '+str(k)+'. \n' 
    fo = open('accuracies_4pca.txt' ,'a')
    fo.write(out)
    fo.close()
    
    return

#----------------------------------------------------#    
#------------------ Main function ------------------ #
#----------------------------------------------------#
if __name__ == "__main__":

    # Define array of k sizes
    cbook_size_k = np.array([512,1024])
    # Create  pca object
    pca = PCA(n_components =100)
    # Compute core 
    for i in range(len(cbook_size_k)):
        print '\nComputing core: cbook_size = '+str(cbook_size_k[i])+', iteration = '+str(i)  
        core(cbook_size_k[i],pca) 
        
    print 'Overall finished'
