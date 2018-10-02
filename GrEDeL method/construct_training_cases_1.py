import time
import pickle
import random
class construct_training_cases:

    def construct_KnowledgeGraph(self,predication_file,entity_relation_to_id_dir,output_dir):
        entity_to_id,relation_to_id,entity_type_to_id,relation_type_to_id=self.mapping(entity_relation_to_id_dir)

        KG={}
        file=open(predication_file,"r")
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
            if len(sline) != 6:
                line=file.readline()
                continue
            if sline[0] == "" or sline[1] == "":
                line=file.readline()
                continue
            subject=self.process_en(sline[0])
            object=self.process_en(sline[1])
            predicate=sline[3]
            s_type=sline[4]
            o_type=sline[5].strip("\n")
            if subject == "" or object == "" or predicate == "" or s_type == "" or o_type == "":
                line=file.readline()
                continue
            subject_type=entity_to_id[subject]+"_"+entity_type_to_id[s_type]
            object_type=entity_to_id[object]+"_"+entity_type_to_id[o_type]
            predicate_type=relation_to_id[predicate]+"_"+relation_type_to_id[predicate]
            #print "-------------------------------------------"
            #print subject_type
            #print predicate_type
            #print object_type
            if subject_type not in KG:
                KG[subject_type]={}
                KG[subject_type][object_type]={}
                KG[subject_type][object_type][predicate_type]=1
            elif object_type not in KG[subject_type]:
                KG[subject_type][object_type]={}
                KG[subject_type][object_type][predicate_type]=1
            else:
                KG[subject_type][object_type][predicate_type]=1
            line=file.readline()
        file.close()
        elapsed=time.clock()-start
        print(str(elapsed))
        output=open(output_dir+"KG","wb+")
        pickle.dump(KG,output)
        output.close()
        print(len(KG))

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



    def mapping(self,entity_relation_to_id_dir):
        print("start mapping ...")
        entity_to_id={}
        relation_to_id={}
        entity_type_to_id={}
        relation_type_to_id={}
        file=open(entity_relation_to_id_dir+"TrainingData/entity2id.txt","r")
        line=file.readline()
        while line:
            sline=line.strip("\n").split("\t")
            entity=sline[0]
            entity_id=sline[1]
            entity_to_id[entity]=entity_id
            line=file.readline()
        file.close()

        file=open(entity_relation_to_id_dir+"TrainingData/relation2id.txt","r")
        line=file.readline()
        while line:
            sline=line.strip("\n").split("\t")
            relation=sline[0]
            relation_id=sline[1]
            relation_to_id[relation]=relation_id
            line=file.readline()
        file.close()

        file=open(entity_relation_to_id_dir+"TrainingDataUmls/entity2id.txt","r")
        line=file.readline()
        while line:
            sline=line.strip("\n").split("\t")
            entity=sline[0]
            entity_id=sline[1]
            entity_type_to_id[entity]=entity_id
            line=file.readline()
        file.close()

        file=open(entity_relation_to_id_dir+"TrainingDataUmls/relation2id.txt","r")
        line=file.readline()
        while line:
            sline=line.strip("\n").split("\t")
            relation=sline[0]
            relation_id=sline[1]
            relation_type_to_id[relation]=relation_id
            line=file.readline()
        file.close()
        print("finished mapping ...")
        return entity_to_id,relation_to_id,entity_type_to_id,relation_type_to_id

    def construct_positive_training_cases(self,entity_relation_to_id_dir,KG_file,TTD_triple_cases_file,output_dir):
        entity_to_id,relation_to_id,entity_type_to_id,relation_type_to_id=self.mapping(entity_relation_to_id_dir)
        KG=pickle.load(open(KG_file,"rb"))
        print("starting constrcuting positive cases ...")
        file=open(TTD_triple_cases_file,"r")
        line=file.readline()
        output=open(output_dir+"training_cases_positive.txt","w+")
        count=0
        while line:
            print(str(count)+"\t"+line)
            count += 1
            #Disease:breast_cancer	Target:cli	Drug:ogx-011
            sline=line.strip("\n").split("\t")
            if len(sline[0].split(":")) != 2 or len(sline[1].split(":")) != 2 or len(sline[2].split(":")) != 2:
                line=file.readline()
                count += 1
                continue
            disease=entity_to_id[sline[0].split(":")[1]]
            target=entity_to_id[sline[1].split(":")[1]]
            drug=entity_to_id[sline[2].split(":")[1]]
            for type1 in entity_type_to_id:
                for type2 in entity_type_to_id:
                    for type3 in entity_type_to_id:
                        disease_temp=disease+"_"+entity_type_to_id[type1]
                        target_temp=target+"_"+entity_type_to_id[type2]
                        drug_temp=drug+"_"+entity_type_to_id[type3]
                        self.constrcut_paths(KG,disease_temp,target_temp,drug_temp,output)
            line=file.readline()
        output.close()

    def constrcut_paths(self,KG,disease_temp,target_temp,drug_temp,output):
        if drug_temp not in KG or target_temp not in KG:
            return
        else:
            # 5
            for entity1 in KG[drug_temp]:
                if entity1 in KG and target_temp in KG[entity1]:
                    for entity2 in KG[target_temp]:
                        if entity2 in KG and disease_temp in KG[entity2]:
                            for predicate1 in KG[drug_temp][entity1]:
                                for predicate2 in KG[entity1][target_temp]:
                                    for predicate3 in KG[target_temp][entity2]:
                                        for predicate4 in KG[entity2][disease_temp]:
                                            output.write(drug_temp+"_"+predicate1+"_"+entity1+"_"+predicate2+"_"+target_temp+"_"+predicate3+"_"+entity2+"_"+predicate4+"_"+disease_temp+"\n")
            # 3
            if target_temp in KG[drug_temp] and disease_temp in KG[target_temp]:
                for predicate1 in KG[drug_temp][target_temp]:
                    for predicate2 in KG[target_temp][disease_temp]:
                        output.write(drug_temp+"_"+predicate1+"_"+target_temp+"_"+predicate2+"_"+disease_temp+"\n")

            # 4
            if target_temp in KG[drug_temp] and disease_temp not in KG[target_temp]:
                for entity2 in KG[target_temp]:
                    if entity2 in KG and disease_temp in KG[entity2]:
                        for predicate1 in KG[drug_temp][target_temp]:
                            for predicate2 in KG[target_temp][entity2]:
                                for predicate3 in KG[entity2][disease_temp]:
                                    output.write(drug_temp+"_"+predicate1+"_"+target_temp+"_"+predicate2+"_"+entity2+"_"+predicate3+"_"+disease_temp+"\n")

            # 4
            if target_temp not in KG[drug_temp] and disease_temp in KG[target_temp]:
                for entity1 in KG[drug_temp]:
                    if entity1 in KG and target_temp in KG[entity1]:
                        for predicate1 in KG[drug_temp][entity1]:
                            for predicate2 in KG[entity1][target_temp]:
                                for predicate3 in KG[target_temp][disease_temp]:
                                    output.write(drug_temp+"_"+predicate1+"_"+entity1+"_"+predicate2+"_"+target_temp+"_"+predicate3+"_"+disease_temp+"\n")

    def construct_negative_training_cases(self,entity_relation_to_id_dir,output_dir,selected_data_number):
        file_positive=open(output_dir+"whole_training_cases_positive.txt","r")
        output_file_positive=open(training_cases_dir+"training_cases_positive.txt","w+")
        output_file_negative=open(training_cases_dir+"training_cases_negative.txt","w+")
        entity_to_id,relation_to_id,entity_type_to_id,relation_type_to_id=self.mapping(entity_relation_to_id_dir)
        relation_list=[2,6,10,14]
        entity_list=[0,4,8,12,16]
        line=file_positive.readline()
        total_count=0
        training_data_positive={}
        training_data_negative={}
        while line:
            line=line.strip("\n")
            sline=line.split("_")
            if len(sline) != 18:
                line=file_positive.readline()
                continue
            if total_count%10000==0:
                print("total cases: "+str(total_count))
            #866620_107_7_7_127614_107_7_7_968369_76_14_14_879821_107_48_48_772915_34
            #总共9个  5个entity 4个relation
            if random.randint(0,1)==0:
                relation_list_temp=random.sample(relation_list,1)
                for r in relation_list_temp:
                    sline[r]=random.randint(0,len(relation_to_id)-1)
                    sline[r+1]=sline[r]
            else:
                entity_list_temp=random.sample(entity_list,1)
                for e in entity_list_temp:
                    sline[e]=random.randint(0,len(entity_to_id)-1)
                    sline[e+1]=random.randint(0,len(entity_type_to_id)-1)
            new_case=""
            for i in sline:
                new_case += "_"
                new_case += str(i)
            new_case=new_case.strip("_")
            training_data_positive[total_count]=line
            training_data_negative[total_count]=new_case
            total_count += 1
            line=file_positive.readline()
        file_positive.close()
        print("读完所有文件")
        count=0
        while count < selected_data_number:
            random_select_number=random.randint(0,total_count-1)
            output_file_positive.write(training_data_positive[random_select_number]+"\n")
            output_file_negative.write(training_data_negative[random_select_number]+"\n")
            count += 1
        output_file_positive.close()
        output_file_negative.close()





#KG_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/KnowledgeGraph/KG"
#TTD_triple_cases_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TTD/experimental_disease_target_drug_negative.txt"
#output_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TrainingCases/"
#entity_relation_to_id_dir="/home/a914/zhaodi/Embedding/data/EmbeddingTrainingData 2/1990 to 2013/"

#s=construct_training_cases()
#s.construct_KnowledgeGraph(predication_file,entity_relation_to_id_dir,output_dir)

#s.construct_positive_training_cases(entity_relation_to_id_dir,KG_file,TTD_triple_cases_file,output_dir)
#s.construct_negative_training_cases(entity_relation_to_id_dir,output_dir)
selected_data_number=10000
training_cases_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TrainingCases/"
entity_relation_to_id_dir="/home/a914/zhaodi/Embedding/data/EmbeddingTrainingData 2/1990 to 2013/"
s=construct_training_cases()
s.construct_negative_training_cases(entity_relation_to_id_dir,training_cases_dir,selected_data_number)