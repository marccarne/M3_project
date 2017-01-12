# coding: utf-8

import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster

# ## Pyramid 

# ### Classe imatge

# In[17]:

#Classe Imagen para estructurar mejor el código
class image:
    def __init__(self, descriptors, keypoints, size):
        self.des = descriptors
        self.kpt = keypoints
        self.shape = size


# ### Lectura de dades

# In[18]:

def read_data(train_files, train_labels, test_files, test_labels):
    
    train_images_filenames = cPickle.load(open(train_files,'r'))
    test_images_filenames = cPickle.load(open(test_files,'r'))
    train_labels = cPickle.load(open(train_labels,'r'))
    test_labels = cPickle.load(open(test_labels,'r'))

    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)
    
    return (train_images_filenames, train_labels, test_images_filenames, test_labels)


# ### Extracció de característiques

# In[19]:

def extract_image_features(train_images_filenames, train_labels, detector):
    return [(compute_feature(img, detector), label) for img,label in zip(train_images_filenames, train_labels)]


# In[20]:

def compute_feature(image_filename, detector):

    ima=cv2.imread(image_filename)
    gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    kpt,des=detector.detectAndCompute(gray,None)
    new_image = image(des, kpt, gray.shape )
    
    return new_image


# ### Pyramid kernel

# In[21]:

def intersection(hist1, hist2):
    return sum([min(a,b) for a,b in zip(hist1,hist2)])


# In[22]:

#Obtenemos el número de elementos que forma cada partición de la imagen
def compute_pyramid_levels(v, i, lvl):
    if i == 0:
        return v
    return compute_pyramid_levels(pow(v,lvl),i-1,lvl)


# In[23]:

def Compute_value(v1, v2, part, n_words, lvl):
    
    results = []
    #lista conteniendo el número de elementos que forma cada partición de la imagen (en este cacso, [16,4,1])
    valors = [compute_pyramid_levels(lvl, i, lvl) for i in range(lvl,0, -1)] + [1]
    
    for i in range(lvl+1):
        
        #obtenemos el indice a partir del qual están los elementos del nivel i de la pirámide
        tall = part - sum(valors[:i+1])
        #obtenemos el resultado de la función intersección del nivel requerido
        results.append(intersection(v1[n_words*tall:n_words*tall+valors[i]*n_words],v2[n_words*tall:n_words*tall+valors[i]*n_words]))

    #Tenim el vector de resultats de les interseccions: results = [h_0, h_1, h_2]
    #obtenemos el valor mediante la fórmula presente en la slide 60
    valor = results[0]/2 + results[1]/4 + results[2]/4
    return valor


# In[24]:

def Pyramid_Kernel(mat1, mat2, lvl = 2):
    
    mat = []
    part = 1
    for i in range(1, lvl+1):
        part += compute_pyramid_levels(lvl,i,lvl)
        
    n_words = mat1[0].shape[0]/part

    for v1 in mat1:
        row = []
        for v2 in mat2:
            row.append(Compute_value(v1, v2, part, n_words, lvl))
        mat.append(row)
    return np.array(mat)


# ### Codeboook

# In[25]:

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


# ### Bow

# In[26]:

def Pyramid_BoW_fisher(gmm, Image_info, x_part, y_part):
    
    k = codebook.cluster_centers_.shape[0]
    # Dimensió de cada vector = k* nº cel·les, en aquest cas 21 (16 peques, 4 qadrants i la sencera)
    visual_words=[]
    i = 0
    for img,label in Image_info:

        total_rows = x_part ** 2
        total_columns = y_part ** 2

        x_step = img.shape[0]/total_rows
        y_step = img.shape[1]/total_columns

        Q = [total_rows][total_columns]
        Q_int = [x_part][y_part]

        #classifiquem els descriptors segons les coordenades del kp al qual pertanyen
        for kpt, desc in zip(img.kpt, img.des):

            #nota: shape(num_files, num_columnes) \ coordenada del punt = (x,y)
            kpt = kpt.pt

            for row in xrange(total_rows):
                for column in xrange(total_columns):
                    if kpt[0] < y_step*(column+1) and kpt[0] > y_step*column and kpt[1] < x_step*(row+1) and kpt[1] > x_step*row:
                        Q[row][column].append(desc.tolist())

        
        #Componer nivel intermedio
        for row in xrange(x_part):
            for column in xrange(y_part):
                for sub_r in xrange(x_part):
                    for sub_c in xrange(y_range):
                        Q_int[row][column] = np.array(Q[row][column] + Q[row*x_part+sub_r][column*y_part+sub_c])
                #Q_int[row][column] = np.array(Q[row*x_part:row*x_part+(x_part),column*y_part:column*y_part+(y_part)])

        #Per comoditat, formem una llista amb tots els descriptors classificats
        #Q = [img.des, Q1, Q2, Q3, Q4, np.array(Q11), np.array(Q12), np.array(Q13), np.array(Q14), np.array(Q21), np.array(Q22), np.array(Q23), np.array(Q24), np.array(Q31), np.array(Q32), np.array(Q33), np.array(Q34), np.array(Q41), np.array(Q42), np.array(Q43), np.array(Q44)]

        des_array = []
        des_array.append(img.des)

        for arr_r in xrange(x_part):
            for arr_c in xrange(y_part):
                des_array.append(Q_int[arr_r][arr_c])

        for arr_r in xrange(total_rows):
            for arr_c in xrange(total_columns):
                des_array.append(Q[arr_r][arr_c])


        #Iniciem el descriptor piramidal
        Pdesc = []
        for q in des_array:
            #Generate fisher vectors with each grid partition
            if len(q):
                #Fisher prediction
                Pdesc += ynumpy.fisher(gmm, q, include=['mu', 'sigma'])
                #Pdesc += np.bincount(codebook.predict(np.array(q)),minlength=k).tolist() just for BOW
            else:
                Pdesc += np.zeros(k, dtype=np.int64).tolist()

        visual_words.append(Pdesc)
    
    return visual_words


# ### SVM

# In[27]:

def train_SVM_pyramid(visual_words, train_labels, toStore= False, filename = ""):

    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    kernelMatrix = Pyramid_Kernel(D_scaled, D_scaled)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='precomputed', C=1).fit(kernelMatrix, train_labels)
    print 'Done!'
    
    if toStore:
        cPickle.dump(clf, open(filename, "wb"))
    return clf, stdSlr, D_scaled


# ### Eval

# In[28]:

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


# In[35]:

def evaluate_pyramid(test_info, codebook, stdSlr, D_scaled):
    
    visual_words_test = np.array(Pyramid_BoW(codebook, test_info))
    
    predictMatrix = Pyramid_Kernel(stdSlr.transform(visual_words_test), D_scaled)

    return 100*clf.score(predictMatrix, test_labels)

    
 #------------------ Main function ------------------ #
if __name__ == "__main__":
	start = time.time()

	# read the train and test files

	train_images_filenames, train_labels, test_images_filenames, test_labels = read_data('train_images_filenames.dat', 'train_labels.dat', 'test_images_filenames.dat', 'test_labels.dat')


	# create the SIFT detector object

	detector = cv2.SIFT(nfeatures=50)

	# read the just 30 train images per class
	# extract SIFT keypoints and descriptors
	# store descriptors in a python list of numpy arrays

	Train_info = extract_image_features(train_images_filenames, train_labels, detector)
	Test_info = extract_image_features(test_images_filenames, test_labels, detector)

	#as default k=512
	print "computing codebook"
	codebook = compute_codebook("codebook_pyramid_2.dat", [img[0].des for img in Train_info], k = 20)

	print "computing train VW"
	visual_words = Pyramid_BoW(codebook, Train_info)

	# Train a linear SVM classifier
	print "computing SVM"
	clf, stdSlr, D_scaled = train_SVM_pyramid(visual_words, [label for img, label in Train_info])

	#accuracy = evaluate_test_Kernel(test_images_filenames, test_labels, codebook, stdSlr,D_scaled)
	print "testing!"
	accuracy = evaluate_pyramid(Test_info, codebook, stdSlr,D_scaled)
	print 'Final accuracy: ' + str(accuracy)
    
	k = 20
	out = 'Accuracy:'+str(accuracy)+' k: '+str(k)+'. \n'
	fo = open('accuracies.txt' ,'a')
	fo.write(out)
	fo.close()

	end=time.time()
	print 'Done in '+str(end-start)+' secs.'

	## 49.56% in 285 secs.


