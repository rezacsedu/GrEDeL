from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import copy
# prepare sequence
#from sklearn.cross_validation import StratifiedKFold
import keras
import time
from keras.layers import Dropout
keras.backend.clear_session()
def load_data(embedding_dir,training_cases_dir,step_length,GraphEmbedding_length,OntologyEmbedding_length):
    # load your data using this function
    relation2vec_G={}
    file=open(embedding_dir+str(GraphEmbedding_length)+"/relation2vec.bern","r")
    line=file.readline()
    count=0
    while line:
        vector_temp=[]
        sline=line.strip("\t\n").split("\t")
        for i in sline:
            vector_temp.append(float(i))
    
        relation2vec_G[count]=copy.deepcopy(vector_temp)
        line=file.readline()
        count += 1
    file.close()
    print("----------------------------1----------------------------")
    entity2vec_G={}
    file=open(embedding_dir+str(GraphEmbedding_length)+"/entity2vec.bern","r")
    line=file.readline()
    
    count=0
    while line:
        vector_temp=[]
        sline=line.strip("\n\t").split("\t")
        for i in sline:
            vector_temp.append(float(i))
    
        entity2vec_G[count]=copy.deepcopy(vector_temp)
        line=file.readline()
        count += 1
    file.close()
    print("----------------------------2----------------------------")

    relation2vec_Ontology={}
    file=open(embedding_dir+str(OntologyEmbedding_length)+"_ontology/relation2vec.bern","r")
    line=file.readline()
    count=0
    while line:
        vector_temp=[]
        sline=line.strip("\t\n").split("\t")
        for i in sline:
            vector_temp.append(float(i))

        relation2vec_Ontology[count]=copy.deepcopy(vector_temp)
        line=file.readline()
        count += 1
    file.close()
    print("----------------------------3----------------------------")
    entity2vec_Ontology={}
    file=open(embedding_dir+str(OntologyEmbedding_length)+"_ontology/entity2vec.bern","r")
    line=file.readline()

    count=0
    while line:
        vector_temp=[]
        sline=line.strip("\n\t").split("\t")
        for i in sline:
            vector_temp.append(float(i))

        entity2vec_Ontology[count]=copy.deepcopy(vector_temp)
        line=file.readline()
        count += 1
    file.close()
    print("----------------------------4----------------------------")

#    training_data_number=0
    # 6273_36_7_7_274713_107_14_14_473268_107_10_10_354045_118_33_33_106647_126
    file=open(training_cases_dir+"training_cases_positive.txt","r")
    line=file.readline()
    X=array([0 for i in range(GraphEmbedding_length+OntologyEmbedding_length)])
    Y=array([0])
    positive_data_count=0
    while line:
        sline=line.strip("\n").split("_")
        if len(sline) != 18:
            line=file.readline()
            continue
        for i in range(0,len(sline),2):
            if (i/2%2) == 0:
                X=np.append(X,copy.deepcopy(entity2vec_G[int(sline[i+1])]),axis=0)
                X=np.append(X,copy.deepcopy(entity2vec_Ontology[int(sline[i+1])]),axis=0)

            elif (i/2%2) == 1:
                X=np.append(X,copy.deepcopy(relation2vec_G[int(sline[i+1])]),axis=0)
                X=np.append(X,copy.deepcopy(relation2vec_Ontology[int(sline[i+1])]),axis=0)

            else:
                print("wrong data")
                line=file.readline()
                continue
        line=file.readline()
        if positive_data_count % 1000 == 0:
            print("正例数量：\t"+str(positive_data_count))
        positive_data_count += 1
        Y=np.append(Y,[1],axis=0)
    file.close()
    print("----------------------------3----------------------------")
    file=open(training_cases_dir+"training_cases_negative.txt","r")
    line=file.readline()
    negative_data_count=0
    while line:
        sline=line.strip("\n").split("_")
        if len(sline) != 18:
            line=file.readline()
            continue
        for i in range(0,len(sline),2):
            if (i/2%2) == 0:
                X=np.append(X,copy.deepcopy(entity2vec_G[int(sline[i+1])]),axis=0)
                X=np.append(X,copy.deepcopy(entity2vec_Ontology[int(sline[i+1])]),axis=0)
            elif (i/2%2) == 1:
                X=np.append(X,copy.deepcopy(relation2vec_G[int(sline[i+1])]),axis=0)
                X=np.append(X,copy.deepcopy(relation2vec_Ontology[int(sline[i+1])]),axis=0)
            else:
                print("wrong data")
                line=file.readline()
                continue
        Y=np.append(Y,[0],axis=0)
        if negative_data_count % 1000 == 0:
            print("负例的数量:\t"+str(negative_data_count))
        negative_data_count += 1
        line=file.readline()
    file.close()
    print("----------------------------4----------------------------")

    
    index = [i for i in range(GraphEmbedding_length+OntologyEmbedding_length)]
    X=np.delete(X,index)
    Y=np.delete(Y,0)    
    datalen=len(X)/step_length/(GraphEmbedding_length+OntologyEmbedding_length)
    X=X.reshape(int(datalen), step_length, (GraphEmbedding_length+OntologyEmbedding_length))
    Y=Y.reshape(int(datalen), 1)
    print("----------------------------5----------------------------")
    
    X,Y=unison_shuffled_copies(X,Y)
    return X,Y

def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def create_model(step_length,embedding_length,ontology_length,neurons):
    # create your model using this function
    ## drug-p-e-p-target-p-e-p-disease
    # define LSTM configuration
    # create LSTM
    model = Sequential()
    Dropout(0.2, input_shape=(step_length,(embedding_length+ontology_length)))
    model.add(Dropout(0.5,input_shape=(step_length, (embedding_length+ontology_length))))
    model.add(LSTM(neurons))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    return model
    # train LSTM

def create_BLSTM_model(step_length,embedding_length,ontology_length,neurons):
    # create your model using this function
    ## drug-p-e-p-target-p-e-p-disease
    # define LSTM configuration
    # create LSTM
    model = Sequential()
    Dropout(0.2, input_shape=(step_length,(embedding_length+ontology_length)))
    model.add(Dropout(0.5,input_shape=(step_length, (embedding_length+ontology_length))))
    model.add(LSTM(neurons))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    return model
    # train LSTM
  

def train_and_evaluate_model(model,data_train,labels_train,data_test,labels_test):
    # fit and evaluate here.
    n_epoch = 200
    n_batch = 10
    model.fit(data_train,labels_train, epochs=n_epoch, batch_size=n_batch,verbose=2)
    # evaluate
    #result = model.predict(data_test, batch_size=n_batch, verbose=0)
    predicted = model.predict(data_test)
    predicted = np.reshape(predicted, (predicted.size,))
    #print(data_test)
    return predicted
    #print(labels_test)
#    same_count=0
#    total_count=0
#    for i in range(len(labels_test)):
#        if labels_test[i] < 0 and predicted[i] < 0:
#            same_count += 1
#        elif labels_test[i] > 0 and predicted[i] > 0:
#            same_count += 1
#        total_count += 1
#    print("precision:\t"+str(same_count/total_count))
        
if __name__ == "__main__":
    print("Graph Embedding length is: 25 50 75 100?")
    embedding_length=input()
    embedding_length=int(embedding_length)
    print("Ontology Embedding length is: 10 20 30 40 50?")
    ontology_length=input()
    ontology_length=int(ontology_length)
    print("neurons are:")
    neurons=input()
    neurons=int(neurons)
    X, y = load_data("/home/a914/zhaodi/Embedding/TransE/","/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TrainingCases/",9,embedding_length,ontology_length)
    output_result=open("/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/GraphAndOntologyEmbedding/"+str(embedding_length)+"_"+str(ontology_length)+"_"+str(neurons)+".txt","w+")
    training_model_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/Training_Models/LSTM_GraphAndOntology_Embedding/"
    ten_percent=int(len(y)/10)
    n_folds = 10
    step_length=9
    for cross_validation_number in range(10):
        output_result.write("cross validation\t"+str(cross_validation_number)+"\n")
        start=time.clock()
        print("Cross validation number: "+str(cross_validation_number))
        if cross_validation_number == 0:
            train_data=X[cross_validation_number*ten_percent:]
            train_label=y[cross_validation_number*ten_percent:]
        elif cross_validation_number == 9:
            train_data=X[:cross_validation_number*ten_percent]
            train_label=y[:cross_validation_number*ten_percent]
        else:
            train_data_1=X[:cross_validation_number*ten_percent]
            train_data_2=X[(cross_validation_number+1)*ten_percent:]
            train_label_1=y[:cross_validation_number*ten_percent]
            train_label_2=y[(cross_validation_number+1)*ten_percent:]
            train_data=np.concatenate((train_data_1,train_data_2),axis=0)
            train_label=np.concatenate((train_label_1,train_label_2),axis=0)
        test_data=X[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
        test_lable=y[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
    
        model = create_model(step_length,embedding_length,ontology_length,neurons)
        predicated=train_and_evaluate_model(model,train_data, train_label, test_data, test_lable)
        if len(test_lable) != len(predicated):
            output_result.write("===========================Wrong Results==================================")
        for i in range(len(predicated)):
            output_result.write(str(test_lable[i])+":"+str(predicated[i])+"\t")
        output_result.write("\n")
        model.save(training_model_file+str(embedding_length)+"_"+str(ontology_length)+"_"+str(neurons)+"_"+str(cross_validation_number)+".h5")
    output_result.close()

    # -*- coding: utf-8 -*-


