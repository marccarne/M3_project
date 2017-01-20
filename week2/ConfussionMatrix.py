#!/usr/bin/python
# -*- coding: utf-8 -*-


#Aquesta funció es cridaria dins de evaluate_test, passant-li com a paràmetres stdSlr.transform(visual_words_test), test_labels
#Funció per calcular la matriu de confussió, rep com a paràmetres el Ground Truth i les labels predites pel classificador
def ConfMatrix(GT, Predicted):
    
    
    # Iniciem la matriu de confussió com un sol 0
    Mat_confu = np.zeros((1,1))

    #Iniciem el diccionari de labels, on assignem la coordenada de la label en la matriu
    dic_labels = {}
    num_labels = 0

    #Iterem sobre la parella (original_labels, predicted_labels)
    for label, predict in zip(GT, Predicted): 

        #Comprobem si la label de training ja està en la matriu
        try:
            dic_labels[label]

        #En cas negatiu, l'afegim al diccionari i li assignem la seva posició en la matriu    
        except KeyError:

            num_labels += 1
            dic_labels[label] = num_labels        

            #Afegim una fila a la matriu amb tantes columnes com etiquetes detectades fins al moment 
            Mat_confu = np.c_[Mat_confu, np.zeros((num_labels,1))]
            Mat_confu = np.vstack([Mat_confu, np.zeros((1, Mat_confu.shape[1]))])

            #Assignem el l'índex de la label a la coordenada corresponent de l'eix de la matriu
            Mat_confu[0][num_labels] = num_labels
            Mat_confu[num_labels][0] = num_labels


        #Comprobem si la predicted label ja està en la matriu
        try:
            dic_labels[predict]

        #En cas negatiu, l'afegim al diccionari i li assignem la seva posició en la matriu   
        except KeyError:
            num_labels +=1
            dic_labels[predict] = num_labels


            #Afegim una fila a la matriu amb tantes columnes com etiquetes detectades fins al moment
            Mat_confu = np.c_[Mat_confu, np.zeros((num_labels,1))]
            Mat_confu = np.vstack([Mat_confu, np.zeros((1, Mat_confu.shape[1]))])
            Mat_confu[num_labels][0] = num_labels
            Mat_confu[0][num_labels] = num_labels


        Mat_confu[dic_labels[label]][dic_labels[predict]] +=1
        
    return Mat_confu