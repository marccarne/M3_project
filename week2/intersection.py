
#Evaluation using the intersection Kernel
def evaluate_test_Kernel(test_images_filenames, test_labels, codebook, stdSlr, D_scaled):
    k = codebook.cluster_centers_.shape[0]
    # get all the test data and predict their labels
    visual_words_test=np.zeros((len(test_images_filenames), k),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        #extract features for a single image
        kpt,des= compute_feature(test_images_filenames[i], detector)
        #extract VW for a single image
        words=codebook.predict(des)
        visual_words_test[i,:]=np.bincount(words,minlength=k)
    
    predictMatrix = intersection_Kernel(stdSlr.transform(visual_words_test), D_scaled)

    return 100*clf.score(predictMatrix, test_labels)

#Training using the intersection Kernel
def train_SVM_intersection(visual_words, train_labels, toStore= False, filename = ""):

    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    kernelMatrix = intersection_Kernel(D_scaled, D_scaled)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='precomputed', C=1).fit(kernelMatrix, train_labels)
    print 'Done!'
    
    if toStore:
        cPickle.dump(clf, open(filename, "wb"))
    return clf, stdSlr, D_scaled

#Intersection Kernel
def intersection_Kernel(mat1, mat2):
    mat = []
    for v1 in mat1:
        row = []
        for v2 in mat2:
            row.append(intersection(v1,v2))
        mat.append(row)
    return np.array(mat)
            
def intersection(hist1, hist2):

    return sum([min(a,b) for a,b in zip(hist1,hist2)])

#Remaining functions did not change, are the same from the basic code, only these need to be added in order to perform the evaluation with the intersectoin kernel











