#!/usr/bin/python
# -*- coding: utf-8 -*-


#Funció per a generar les n particions de train i fer la cross validation. Retorna la tupla (particions_train, particions_labels)
def N_fold(train_images_filenames, train_labels, n):
    
    labels, counts = np.unique(train_labels, return_counts=True)

    diccionari_labels = {}

    for x in labels:
        diccionari_labels[x] = [y for y in range(len(train_labels)) if train_labels[y] == x]
        shuffle(diccionari_labels[x])
    diccionari_labels
    elements = [(x,y/n) for x,y in zip(labels, counts)]
    
    particions = []

    for i in range(n):
        particio = []
        for label, amount in elements:
            if i != n-1:
                particio += diccionari_labels[label][i*amount:(i+1)*amount]
            else:
                particio += diccionari_labels[label][i*amount:]
        particions.append(particio)

    shuffle(particions[-1])

    remaining = len(particions[-1])/n

    i = 0    
    while len(particions[-1]) > min([len(x) for x in particions]):
        i = i%n
        particions[i].append(particions[-1].pop())
        i+=1

    folds = []
    for indexos in particions:
        filenames_particion = []
        labels_particion=[]
        for index in indexos:
            filenames_particion.append(train_images_filenames[index])
            labels_particion.append(train_labels[index])
        folds.append((filenames_particion, labels_particion))
        
    return folds


"""Rutina per crear els conjunts de train i validation"""

folds = N_fold(train_images_filenames, train_labels, 100)

for i in range(len(folds)):
    
    validation_files = folds[i][0]
    validation_labels = folds[i][1]
    
    train_files_part = [f for x,y in folds[0:i] for f in x]
    train_files_part = [f for x,y in folds[0:i] for f in x ]
    train_files_part += [f for x,y in folds[i+1:] for f in x ]
    train_labels_part = [l for x,y in folds[0:i] for l in y ]
    train_labels_part += [l for x,y in folds[i+1:] for l in y ]
    
#Ara train_files_part es el conjunt de fitxers de train,train_labels_part el conjunt de labels de train 
#validation_files el conjunt de fitxers de validació (jugaria el paper de test_files)
#validation_labels el conjunt de labels de validacio (jugaria el paper de test_labels)


