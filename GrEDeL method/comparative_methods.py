import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from numpy import array
import time
import random
import pickle
import copy

class comparative_methods:
    def load_data(self,embedding_dir,training_cases_dir,step_length,embedding_length):
        # load your data using this function
        relation2vec={}
        file=open(embedding_dir+"relation2vec.bern","r")
        line=file.readline()
        count=0
        while line:
            vector_temp=[]
            sline=line.strip("\t\n").split("\t")
            for i in sline:
                vector_temp.append(float(i))

            relation2vec[count]=copy.deepcopy(vector_temp)
            line=file.readline()
            count += 1
        file.close()
        print("----------------------------1----------------------------")
        entity2vec={}
        file=open(embedding_dir+"entity2vec.bern","r")
        line=file.readline()

        count=0
        while line:
            vector_temp=[]
            sline=line.strip("\n\t").split("\t")
            for i in sline:
                vector_temp.append(float(i))

            entity2vec[count]=copy.deepcopy(vector_temp)
            line=file.readline()
            count += 1
        file.close()
        print("----------------------------2----------------------------")
        #    training_data_number=0
        # 6273_36_7_7_274713_107_14_14_473268_107_10_10_354045_118_33_33_106647_126
        file=open(training_cases_dir+"training_cases_positive.txt","r")
        line=file.readline()
        X=array([0 for i in range(embedding_length)])
        Y=array([0])
        positive_data_count=0
        while line:
            sline=line.strip("\n").split("_")
            if len(sline) != 18:
                line=file.readline()
                continue
            for i in range(0,len(sline),2):
                if (i/2%2) == 0:
                    X=np.append(X,copy.deepcopy(entity2vec[int(sline[i])]),axis=0)

                elif (i/2%2) == 1:
                    X=np.append(X,copy.deepcopy(relation2vec[int(sline[i])]),axis=0)

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
                    X=np.append(X,copy.deepcopy(entity2vec[int(sline[i])]),axis=0)
                elif (i/2%2) == 1:
                    X=np.append(X,copy.deepcopy(relation2vec[int(sline[i])]),axis=0)
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


        index = [i for i in range(embedding_length)]
        X=np.delete(X,index)
        Y=np.delete(Y,0)
        datalen=len(X)/step_length/embedding_length
        X=X.reshape(int(datalen), step_length*embedding_length)
        Y=Y.reshape(int(datalen), 1)
        #print(Y.shape)
        #print(type(Y))
        print("----------------------------5----------------------------")

        X,Y=self.unison_shuffled_copies(X,Y)
        return X,Y

    def unison_shuffled_copies(self,a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]


    def Logistic_Regression_training_model(self,embedding_dir,training_cases_dir,step_length,embedding_length,output_dir):
        X,y=self.load_data(embedding_dir,training_cases_dir,step_length,embedding_length)
        #for lamb_2 in [0.0001,0.001,0.01,0.1,1,10,100]:
        for lamb_2 in [1]:
            lr = LogisticRegression(penalty='l2',C=1/lamb_2, random_state=0)
            file_result=open(output_dir+"/lr_results_"+str(lamb_2)+".txt","w+")
            ten_percent=int(len(y)/10)
            Precision_value=0
            Recall_value=0
            F_value=0
            ##开始十倍交叉验证
            Precision_value=0
            Recall_value=0
            F_value=0
            for cross_validation_number in range(10):
                start=time.clock()
                #print("Cross validation number: "+str(cross_validation_number))
                if cross_validation_number == 0:
                    train_sample=X[cross_validation_number*ten_percent:]
                    train_label=y[cross_validation_number*ten_percent:]
                elif cross_validation_number == 9:
                    train_sample=X[:cross_validation_number*ten_percent]
                    train_label=y[:cross_validation_number*ten_percent]
                else:
                    train_sample_1=X[:cross_validation_number*ten_percent]
                    train_sample_2=X[(cross_validation_number+1)*ten_percent:]
                    train_label_1=y[:cross_validation_number*ten_percent]
                    train_label_2=y[(cross_validation_number+1)*ten_percent:]
                    train_sample=np.concatenate((train_sample_1,train_sample_2),axis=0)
                    train_label=np.concatenate((train_label_1,train_label_2),axis=0)
                test_sample=X[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
                test_lable=y[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
                output=open(output_dir+"/LR_model_"+str(cross_validation_number)+".sav","wb+")
                lr_model=lr.fit(train_sample,train_label.ravel())
                pickle.dump(lr_model,output)
                output.close()
                predicates_prob=lr_model.predict_proba(test_sample)
                TP=TN=FP=FN=0
                for i in range(len(predicates_prob)):
                    p1=predicates_prob[i][0]
                    p2=predicates_prob[i][1]
                    label=int(test_lable[i])
                    if p1 > p2:
                        p=0
                    else:
                        p=1
                    if label == 1:
                        if label == p:
                            TP += 1
                        else:
                            FN += 1
                    # label是0
                    else:
                        if label == p:
                            TN += 1
                        else:
                            FP += 1
                tem_Precision_value=(TP)/(TP+FP)
                tem_Recall_value=(TP)/(TP+FN)
                tem_F_value=(2*tem_Precision_value*tem_Recall_value)/(tem_Precision_value+tem_Recall_value)
                file_result.write((str(cross_validation_number)+":\t"+str(tem_Precision_value)+"\t"+str(tem_Recall_value)+"\t"+str(tem_F_value))+"\n")
                Precision_value += tem_Precision_value
                Recall_value += tem_Recall_value
                F_value += tem_F_value

            print(str(lamb_2)+":\t"+'%.3f' % (Precision_value/10)+"\t"+ '%.3f' % (Recall_value/10) +"\t"+'%.3f' %(F_value/10))

    def Support_Vector_Machine_training_model(self,embedding_dir,training_cases_dir,step_length,embedding_length,output_dir):
        X,y=self.load_data(embedding_dir,training_cases_dir,step_length,embedding_length)
        #for lamb_2 in [0.0001,0.001,0.01,0.1,1,10,100]:
        for lamb_2 in [1]:
            lr = LogisticRegression(penalty='l2',C=1/lamb_2, random_state=0)
            file_result=open(output_dir+"/lr_results_"+str(lamb_2)+".txt","w+")
            ten_percent=int(len(y)/10)
            Precision_value=0
            Recall_value=0
            F_value=0
            ##开始十倍交叉验证
            for cross_validation_number in range(10):
                start=time.clock()
                #print("Cross validation number: "+str(cross_validation_number))
                if cross_validation_number == 0:
                    train_sample=X[cross_validation_number*ten_percent:]
                    train_label=y[cross_validation_number*ten_percent:]
                elif cross_validation_number == 9:
                    train_sample=X[:cross_validation_number*ten_percent]
                    train_label=y[:cross_validation_number*ten_percent]
                else:
                    train_sample_1=X[:cross_validation_number*ten_percent]
                    train_sample_2=X[(cross_validation_number+1)*ten_percent:]
                    train_label_1=y[:cross_validation_number*ten_percent]
                    train_label_2=y[(cross_validation_number+1)*ten_percent:]
                    train_sample=np.concatenate((train_sample_1,train_sample_2),axis=0)
                    train_label=np.concatenate((train_label_1,train_label_2),axis=0)
                test_sample=X[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
                test_lable=y[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
                clf = svm.SVC()
                output=open(output_dir+"/svm_model_"+str(cross_validation_number)+".sav","wb+")
                svm_model=clf.fit(train_sample,train_label.ravel())
                pickle.dump(svm_model,output)
                output.close()
                predicate_results=svm_model.predict(test_sample)
                TP=TN=FP=FN=0
                for i in range(len(predicate_results)):
                    p=predicate_results[i]
                    label=test_lable[i][0]
                    #print("p:"+str(p)+"\tlabel:"+str(label))
                    if label == 1:
                        if label == p:
                            TP += 1
                        else:
                            FN += 1
                    # label是0
                    else:
                        if label == p:
                            TN += 1
                        else:
                            FP += 1
                tem_Precision_value=(TP)/(TP+FP)
                tem_Recall_value=(TP)/(TP+FN)
                tem_F_value=(2*tem_Precision_value*tem_Recall_value)/(tem_Precision_value+tem_Recall_value)
                file_result.write((str(cross_validation_number)+":\t"+str(tem_Precision_value)+"\t"+str(tem_Recall_value)+"\t"+str(tem_F_value))+"\n")
                Precision_value += tem_Precision_value
                Recall_value += tem_Recall_value
                F_value += tem_F_value

            print(str(lamb_2)+":\t"+'%.3f' % Precision_value+"\t"+ '%.3f' % Recall_value +"\t"+'%.3f' %F_value)

    def Decision_Tree_training_model(self,embedding_dir,training_cases_dir,step_length,embedding_length,output_dir):
        X,y=self.load_data(embedding_dir,training_cases_dir,step_length,embedding_length)
        #for lamb_2 in [0.0001,0.001,0.01,0.1,1,10,100]:
        for lamb_2 in [1]:
            lr = LogisticRegression(penalty='l2',C=1/lamb_2, random_state=0)
            file_result=open(output_dir+"/lr_results_"+str(lamb_2)+".txt","w+")
            ten_percent=int(len(y)/10)
            Precision_value=0
            Recall_value=0
            F_value=0
            ##开始十倍交叉验证
            for cross_validation_number in range(10):
                start=time.clock()
                #print("Cross validation number: "+str(cross_validation_number))
                if cross_validation_number == 0:
                    train_sample=X[cross_validation_number*ten_percent:]
                    train_label=y[cross_validation_number*ten_percent:]
                elif cross_validation_number == 9:
                    train_sample=X[:cross_validation_number*ten_percent]
                    train_label=y[:cross_validation_number*ten_percent]
                else:
                    train_sample_1=X[:cross_validation_number*ten_percent]
                    train_sample_2=X[(cross_validation_number+1)*ten_percent:]
                    train_label_1=y[:cross_validation_number*ten_percent]
                    train_label_2=y[(cross_validation_number+1)*ten_percent:]
                    train_sample=np.concatenate((train_sample_1,train_sample_2),axis=0)
                    train_label=np.concatenate((train_label_1,train_label_2),axis=0)
                test_sample=X[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
                test_lable=y[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
                clf = tree.DecisionTreeClassifier()
                output=open(output_dir+"/dt_model_"+str(cross_validation_number)+".sav","wb+")
                dt_model=clf.fit(train_sample,train_label.ravel())
                pickle.dump(dt_model,output)
                output.close()
                predicate_results=dt_model.predict(test_sample)
                TP=TN=FP=FN=0
                for i in range(len(predicate_results)):
                    p=predicate_results[i]
                    label=test_lable[i][0]
                    #print("p:"+str(p)+"\tlabel:"+str(label))
                    if label == 1:
                        if label == p:
                            TP += 1
                        else:
                            FN += 1
                    # label是0
                    else:
                        if label == p:
                            TN += 1
                        else:
                            FP += 1
                tem_Precision_value=(TP)/(TP+FP)
                tem_Recall_value=(TP)/(TP+FN)
                tem_F_value=(2*tem_Precision_value*tem_Recall_value)/(tem_Precision_value+tem_Recall_value)
                file_result.write((str(cross_validation_number)+":\t"+str(tem_Precision_value)+"\t"+str(tem_Recall_value)+"\t"+str(tem_F_value))+"\n")
                Precision_value += tem_Precision_value
                Recall_value += tem_Recall_value
                F_value += tem_F_value

            print(str(lamb_2)+":\t"+'%.3f' % Precision_value+"\t"+ '%.3f' % Recall_value +"\t"+'%.3f' %F_value)


    def Multi_Layer_Perceptron_training_model(self,embedding_dir,training_cases_dir,step_length,embedding_length,output_dir):
        X,y=self.load_data(embedding_dir,training_cases_dir,step_length,embedding_length)
        #for lamb_2 in [0.0001,0.001,0.01,0.1,1,10,100]:
        for lamb_2 in [1]:
            lr = LogisticRegression(penalty='l2',C=1/lamb_2, random_state=0)
            file_result=open(output_dir+"/lr_results_"+str(lamb_2)+".txt","w+")
            ten_percent=int(len(y)/10)
            Precision_value=0
            Recall_value=0
            F_value=0
            ##开始十倍交叉验证
            for cross_validation_number in range(10):
                start=time.clock()
                #print("Cross validation number: "+str(cross_validation_number))
                if cross_validation_number == 0:
                    train_sample=X[cross_validation_number*ten_percent:]
                    train_label=y[cross_validation_number*ten_percent:]
                elif cross_validation_number == 9:
                    train_sample=X[:cross_validation_number*ten_percent]
                    train_label=y[:cross_validation_number*ten_percent]
                else:
                    train_sample_1=X[:cross_validation_number*ten_percent]
                    train_sample_2=X[(cross_validation_number+1)*ten_percent:]
                    train_label_1=y[:cross_validation_number*ten_percent]
                    train_label_2=y[(cross_validation_number+1)*ten_percent:]
                    train_sample=np.concatenate((train_sample_1,train_sample_2),axis=0)
                    train_label=np.concatenate((train_label_1,train_label_2),axis=0)
                test_sample=X[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
                test_lable=y[cross_validation_number*ten_percent:(cross_validation_number+1)*ten_percent]
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 2), random_state=1)
                output=open(output_dir+"/mlp_model_"+str(cross_validation_number)+".sav","wb+")
                mlp_model=clf.fit(train_sample,train_label.ravel())
                pickle.dump(mlp_model,output)
                output.close()
                predicate_results=mlp_model.predict(test_sample)
                TP=TN=FP=FN=0
                for i in range(len(predicate_results)):
                    p=predicate_results[i]
                    label=test_lable[i][0]
                    print("p:"+str(p)+"\tlabel:"+str(label))
                    if label == 1:
                        if label == p:
                            TP += 1
                        else:
                            FN += 1
                    # label是0
                    else:
                        if label == p:
                            TN += 1
                        else:
                            FP += 1
                tem_Precision_value=(TP)/(TP+FP)
                tem_Recall_value=(TP)/(TP+FN)
                tem_F_value=(2*tem_Precision_value*tem_Recall_value)/(tem_Precision_value+tem_Recall_value)
                file_result.write((str(cross_validation_number)+":\t"+str(tem_Precision_value)+"\t"+str(tem_Recall_value)+"\t"+str(tem_F_value))+"\n")
                Precision_value += tem_Precision_value
                Recall_value += tem_Recall_value
                F_value += tem_F_value

            print(str(lamb_2)+":\t"+'%.3f' % Precision_value+"\t"+ '%.3f' % Recall_value +"\t"+'%.3f' %F_value)


## HOME COMPUTER
embedding_length=25
embedding_dir="/home/a914/zhaodi/Embedding/TransE/"+str(embedding_length)+"/"
training_cases_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TrainingCases/"
output_dir_LR="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/ComparativeMethods/LogisticRegression/"
output_dir_SVM="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/ComparativeMethods/SupportVectorMachine/"
output_dir_DT="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/ComparativeMethods/DecisionTree/"
s=comparative_methods()
s.Logistic_Regression_training_model(embedding_dir,training_cases_dir,9,25,output_dir_LR)
s.Support_Vector_Machine_training_model(embedding_dir,training_cases_dir,9,25,output_dir_SVM)
s.Decision_Tree_training_model(embedding_dir,training_cases_dir,9,25,output_dir_DT)
#s.Multi_Layer_Perceptron_training_model(embedding_dir,training_cases_dir,step_length=9,embedding_length=25,outputdir=output_dir)