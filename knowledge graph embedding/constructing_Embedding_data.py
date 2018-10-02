import time
import pickle
import random
class obtain_experimental_data:
    def construct_Embedding_training_data(self,file_name,output_dir):
        Train_entities={}
        Train_predicates={}
        Train_entities_umls={}
        Train_predicates_umls={}
        output_entities=open(output_dir+"TrainingData/"+"entitiy2id.txt","w+")
        output_predicates=open(output_dir+"TrainingData/"+"relation2id.txt","w+")
        output_train=open(output_dir+"TrainingData/"+"train.txt","w+")

        output_umls_entities=open(output_dir+"TrainingDataUmls/"+"entity2id.txt","w+")
        output_umls_predicates=open(output_dir+"TrainingDataUmls/"+"relation2id.txt","w+")
        output_umls_train=open(output_dir+"TrainingDataUmls/"+"train.txt","w+")

        file=open(file_name,"r")
        line=file.readline()
        start=time.clock()
        line_count=0
        while line:
            line_count += 1
            if (line_count%100000 == 0):
                print(line_count)
            #focal hand dystonia	patients	with	PROCESS_OF	dsyn	podg
            #hand	focal hand dystonia	hand dystonia	PROCESS_OF	dsyn	podg
            sline=line.split("\t")
            if sline[0] == "" or sline[1] == "":
                line=file.readline()
                continue
            subject=self.process_en(sline[0])
            object=self.process_en(sline[1])
            predicate=sline[3]
            subject_type=sline[4]
            object_type=sline[5].strip("\n")
            if subject not in Train_entities and subject != "":
                Train_entities[subject]=1
            if object not in Train_entities and object != "":
                Train_entities[object]=1
            if predicate not in Train_predicates:
                Train_predicates[predicate]=1

            if subject_type not in Train_entities_umls:
                Train_entities_umls[subject_type]=1
            if object_type not in Train_predicates_umls:
                Train_entities_umls[object_type]=1
            if predicate not in Train_predicates_umls:
                Train_predicates_umls[predicate]=1

            if subject != "" and object != "":
                output_train.write(subject+"\t"+object+"\t"+predicate+"\n")
            if subject_type != "" and object_type != "":
                output_umls_train.write(subject_type+"\t"+object_type+"\t"+predicate+"\n")
            line=file.readline()
        count=0
        for entity in Train_entities:
            output_entities.write(entity+"\t"+str(count)+"\n")
            count += 1

        count=0
        for predicate in Train_predicates:
            output_predicates.write(predicate+"\t"+str(count)+"\n")
            count += 1

        count=0
        for entity in Train_entities_umls:
            output_umls_entities.write(entity+"\t"+str(count)+"\n")
            count += 1

        count=0
        for predicate in Train_predicates_umls:
            output_umls_predicates.write(predicate+"\t"+str(count)+"\n")
            count += 1

        output_entities.close()
        output_predicates.close()
        output_train.close()
        output_umls_entities.close()
        output_umls_predicates.close()
        output_umls_train.close()
        print "finished..."

    def process_en(self,sentence):
        sentence=sentence.lower().strip("\n. ")
        sentence_split=sentence.split(" ")
        new_sentence=sentence_split[0].strip(".")
        for split in sentence_split[1:]:
            split=split.strip(". ")
            new_sentence += "_"
            new_sentence += split
        if new_sentence[-1] == 's':
            new_sentence=new_sentence[:-1]
        return new_sentence

##########################################Graph Embedding + LSTM ####################################

file_name="/home/a914/zhaodi/Embedding/data/predications_of_Knowledge_Graph_1/1990-2013/predications.txt"
output_dir="/home/a914/zhaodi/Embedding/data/EmbeddingTrainingData 2/1990 to 2013/"
s=obtain_experimental_data()
s.construct_Embedding_training_data(file_name,output_dir)

