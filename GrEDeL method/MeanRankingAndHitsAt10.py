import pickle
import random
import keras
import numpy as np
import copy
from numpy import array
from keras.models import load_model
class DrugRediscover:
    def construct_drug_rediscovery_cases(self,KG_file,TTD_cases_dir,entity_relation_to_id_dir):
        KG_temp=open(KG_file+"KG","rb")
        KG=pickle.load(KG_temp)
        KG_temp.close()
        KG_entities={}
        for subject in KG:
            KG_entities[subject]=1
            for object in KG[subject]:
                KG_entities[object]=1
        print("------------------------Constructing KG: "+str(len(KG))+"----------------------------")
        targets={}
        target_file=open(TTD_cases_dir+"candidate_targets.txt","r")
        line=target_file.readline()
        while line:
            target=line.strip("\n").split("\t")[1]
            targets[target]=1
            line=target_file.readline()
        target_file.close()
        #构造drug-target-disease cases
        entity_to_id,relation_to_id,entity_type_to_id,relation_type_to_id=self.mapping(entity_relation_to_id_dir)
        entity_type_to_id_length=len(entity_type_to_id)
        drug_disease_file=open(TTD_cases_dir+"disease_drug_cases_hitat10_another_part.txt","r")
        output_file=open(TTD_cases_dir+"drug_target_disease_cases_for_drug_rediscovery_another_part.txt","w+")
        count=0
        line=drug_disease_file.readline()
        while line:
            sline=line.strip("\n").split("\t")
            drug=sline[0].split(":")[1]
            disease=sline[1].split(":")[1]
            print(str(count)+":\t"+drug+"\t"+disease)
            count += 1
            if drug in entity_to_id and disease in entity_to_id:
                for target in targets:
                    if target in entity_to_id:
                        self.construct_dtd_candidate_cases_from_one_dtd(KG,KG_entities,entity_to_id[drug],entity_to_id[target],entity_to_id[disease],entity_type_to_id_length,output_file)
            line=drug_disease_file.readline()
        drug_disease_file.close()
        output_file.close()


    def construct_dtd_candidate_cases_from_one_dtd(self,KG,KG_entities,drug,target,disease,entity_type_to_id_length,output_file):
        for drug_t in range(entity_type_to_id_length):
            drug_type=drug+"_"+str(drug_t)
            for target_t in range(entity_type_to_id_length):
                target_type=target+"_"+str(target_t)
                for disease_t in range(entity_type_to_id_length):
                    disease_type=disease+"_"+str(disease_t)
                    if drug_type in KG and target_type in KG and disease_type in KG_entities:
                        for entity_1 in KG[drug_type]:
                            if entity_1 in KG and target_type in KG[entity_1]:
                                for entity_2 in KG[target_type]:
                                    if entity_2 in KG and disease_type in KG[entity_2]:
                                        for predicate_1 in KG[drug_type][entity_1]:
                                            for predicate_2 in KG[entity_1][target_type]:
                                                for predicate_3 in KG[target_type][entity_2]:
                                                    for predicate_4 in KG[entity_2][disease_type]:
                                                        output_file.write(drug_type+"_"+predicate_1+"_"+entity_1+"_"+predicate_2+"_"+target_type+"_"+predicate_3+"_"+entity_2+"_"+predicate_4+"_"+disease_type+"\n")
                                                        return

    def construct_drug_rediscovery_negative_cases(self,KG_file,positive_cases_file,entity_relation_to_id_dir,candidate_drugs_file,output_file):
        KG_temp=open(KG_file+"KG","rb")
        KG=pickle.load(KG_temp)
        KG_temp.close()
        #将entity_to_id等载入
        entity_to_id,relation_to_id,entity_type_to_id,relation_type_to_id=self.mapping(entity_relation_to_id_dir)
        entity_type_to_id_length=len(entity_type_to_id)
        relation_type_to_id_length=len(relation_type_to_id)
        entity_to_id_length=len(entity_to_id)
        # 将候选药物（所有在TTD出现的药物和满足药物语义类型的物质都拿出来）
        drugs=[]
        candidate_drugs=open(candidate_drugs_file,"r")
        line=candidate_drugs.readline()
        while line:
            drug=line.strip("\n")
            if drug in entity_to_id:
                drugs.append(entity_to_id[drug])
            line=candidate_drugs.readline()
        candidate_drugs.close()

        # 根据每一条正例 drug-entity-target-entity-disease 构造100个候选药物的候选例子
        output=open(output_file,"w+")
        positive_cases=open(positive_cases_file,"r")
        line=positive_cases.readline()
        need_to_rediscover_diseases={}
        candidate_drugs={}
        count=0
        while line:
            #916758_9_44_44_316778_88_20_20_149215_104_12_12_677356_34_36_36_772252_34
            sline=line.strip("\n").split("_")
            disease=sline[16]
            other_part=sline[10]+"_"+sline[11]+"_"+sline[12]+"_"+sline[13]+"_"+sline[14]+"_"+sline[15]+"_"+sline[16]+"_"+sline[17]
            if disease not in need_to_rediscover_diseases:
                count += 1
                print(str(count)+":\t"+disease)
                need_to_rediscover_diseases[disease]=1
                candidate_drugs=self.random_choose_drugs(drugs,99)
            target_type=sline[8]+"_"+sline[9]
            for drug in candidate_drugs:
                self.construct_one_candidate_dtd(KG,drug,target_type,other_part,entity_to_id_length,entity_type_to_id_length,relation_type_to_id_length,output)
            line=positive_cases.readline()
        positive_cases.close()
        output.close()


    def construct_one_candidate_dtd(self,KG,drug,target_type,other_part,entity_to_id_length,entity_type_to_id_length,relation_type_to_id_length,output_file):
        for drug_t in range(entity_type_to_id_length):
            drug_type=drug+"_"+str(drug_t)
            if drug_type in KG:
                for entity_1 in KG[drug_type]:
                    if entity_1 in KG and target_type in KG[entity_1]:
                        for predicate_type_1 in KG[drug_type][entity_1]:
                            for predicate_type_2 in KG[entity_1][target_type]:
                                output_file.write(drug_type+"_"+predicate_type_1+"_"+entity_1+"_"+predicate_type_2+"_"+target_type+"_"+other_part+"\n")
                                return
        drug_t=str(random.randint(0,entity_type_to_id_length-1))
        predicate_1=str(random.randint(0,relation_type_to_id_length-1))
        entity=str(random.randint(0,entity_to_id_length-1))
        entity_type=str(random.randint(0,entity_type_to_id_length-1))
        predicate_2=str(random.randint(0,relation_type_to_id_length-1))
        output_file.write(drug+"_"+drug_t+"_"+predicate_1+"_"+predicate_1+"_"+entity+"_"+entity_type+"_"+predicate_2+"_"+predicate_2+"_"+target_type+"_"+other_part+"\n")
        return


    def random_choose_drugs(self,drugs,number_of_choose):
        candidate_drugs={}
        while len(candidate_drugs) < number_of_choose:
            candidate_drugs[str(random.randint(0,len(drugs)-1))]=1
        return candidate_drugs

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


    def drug_rediscovery(self,gold_standard_cases_file,candidate_cases_file,model_file,embedding_dir,output_dir):
        keras.backend.clear_session()
        step_length=9

        OntologyEmbedding_length=0
        print("输入Graph Embedding 的长度:\t25\t50\t75\t100")
        GraphEmbedding_length=int(input())
        print("输入Ontology Embedding 的长度:\t10\t20\t35\t40\t50")
        OntologyEmbedding_length=int(input())
        # load Embedding Data
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
        print("----------------------------FINISHED LOADING GRAPH EMBEDDING DATA----------------------------")

        relation2vec_O={}
        file=open(embedding_dir+str(OntologyEmbedding_length)+"_ontology/relation2vec.bern","r")
        line=file.readline()
        count=0
        while line:
            vector_temp=[]
            sline=line.strip("\t\n").split("\t")
            for i in sline:
                vector_temp.append(float(i))

            relation2vec_O[count]=copy.deepcopy(vector_temp)
            line=file.readline()
            count += 1
        file.close()
        entity2vec_O={}
        file=open(embedding_dir+str(OntologyEmbedding_length)+"_ontology/entity2vec.bern","r")
        line=file.readline()
        count=0
        while line:
            vector_temp=[]
            sline=line.strip("\n\t").split("\t")
            for i in sline:
                vector_temp.append(float(i))

            entity2vec_O[count]=copy.deepcopy(vector_temp)
            line=file.readline()
            count += 1
        file.close()
        print("----------------------------FINISHED LOADING ONTOLOGY EMBEDDING DATA----------------------------")
        # loading LSTM model
        model = load_model(model_file+str(GraphEmbedding_length)+"_"+str(OntologyEmbedding_length)+"_100_1.h5")
        print(model.summary())
        print("----------------------------FINISHED LOADING LSTM MODEL----------------------------")
        # SCORE AND RANK CANDIDATE DRUGS：
        # 1. score gold standard cases
        # score and rank candidate drugs
        gold_standard_drugs_score={}
        file=open(gold_standard_cases_file,"r")
        file_output=open(output_dir+"gold_standard_drugs_scores.txt","w+")
        line=file.readline()
        while line:
            # 6273_36_7_7_274713_107_14_14_473268_107_10_10_354045_118_33_33_106647_126
            sline=line.strip("\n").split("_")
            disease=sline[16]
            drug=sline[0]
            X=array([])
            for i in range(0,len(sline),2):
                if (i/2%2) == 0:
                    X=np.append(X,copy.deepcopy(entity2vec_G[int(sline[i])]),axis=0)
                    X=np.append(X,copy.deepcopy(entity2vec_O[int(sline[i+1])]),axis=0)
                elif (i/2%2) == 1:
                    X=np.append(X,copy.deepcopy(relation2vec_G[int(sline[i])]),axis=0)
                    X=np.append(X,copy.deepcopy(relation2vec_O[int(sline[i+1])]),axis=0)
            X=X.reshape(1, step_length, GraphEmbedding_length+OntologyEmbedding_length)
            predicted=model.predict(X)
            file_output.write(line.strip("\n")+"\t"+str(predicted[0][0])+"\n")
            line=file.readline()
        file.close()
        file_output.close()
        candidate_drugs_score={}
        file=open(candidate_cases_file,"r")
        file_output=open(output_dir+"candidate_drugs_scores.txt","w+")
        line=file.readline()
        while line:
            # 6273_36_7_7_274713_107_14_14_473268_107_10_10_354045_118_33_33_106647_126
            sline=line.strip("\n").split("_")
            disease=sline[16]
            drug=sline[0]
            X=array([])
            for i in range(0,len(sline),2):
                if (i/2%2) == 0:
                    X=np.append(X,copy.deepcopy(entity2vec_G[int(sline[i])]),axis=0)
                    X=np.append(X,copy.deepcopy(entity2vec_O[int(sline[i+1])]),axis=0)
                elif (i/2%2) == 1:
                    X=np.append(X,copy.deepcopy(relation2vec_G[int(sline[i])]),axis=0)
                    X=np.append(X,copy.deepcopy(relation2vec_O[int(sline[i+1])]),axis=0)
            X=X.reshape(1, step_length, GraphEmbedding_length+OntologyEmbedding_length)
            predicted=model.predict(X)
            file_output.write(line.strip("\n")+"\t"+str(predicted[0][0])+"\n")
            line=file.readline()
        file.close()
        file_output.close()

    def analyze_MeanRanking_Hitsat10(self,input_dir,output_dir):
        #candidate_drugs_scores.txt
        #gold_standard_drugs_scores.txt
        #
        file_gold_standard=open(input_dir+"gold_standard_drugs_scores.txt","r")
        disease_drugs={}
        line=file_gold_standard.readline()
        while line:
            #916758_9_44_44_316778_88_20_20_149215_104_12_12_677356_34_36_36_772252_34	0.144068
            sline=line.strip("\n").split("\t")
            score=float(sline[1])
            ssline=sline[0].split("_")
            drug=ssline[0]
            target=ssline[8]
            disease=ssline[16]
            if disease not in disease_drugs:
                disease_drugs[disease]={}
                disease_drugs[disease][drug]={}
                disease_drugs[disease][drug]["score"]=score
                disease_drugs[disease][drug]["association"]=line
                disease_drugs[disease][drug]["rank"]=1
            elif drug not in disease_drugs[disease]:
                disease_drugs[disease][drug]={}
                disease_drugs[disease][drug]["score"]=score
                disease_drugs[disease][drug]["association"]=line
                disease_drugs[disease][drug]["rank"]=1
            elif score > disease_drugs[disease][drug]["score"]:
                disease_drugs[disease][drug]["score"]=score
                disease_drugs[disease][drug]["association"]=line
                disease_drugs[disease][drug]["rank"]=1
            line=file_gold_standard.readline()
        file_gold_standard.close()

        print("finished loading gold standard drugs ...")
        file_candidate_drugs=open(input_dir+"candidate_drugs_scores.txt","r")
        line=file_candidate_drugs.readline()
        disease_drugs_candidate={}
        while line:
            #916758_9_44_44_316778_88_20_20_149215_104_12_12_677356_34_36_36_772252_34	0.144068
            sline=line.strip("\n").split("\t")
            score=float(sline[1])
            ssline=sline[0].split("_")
            drug=ssline[0]
            target=ssline[8]
            disease=ssline[16]
            if disease not in disease_drugs_candidate:
                disease_drugs_candidate[disease]={}
                disease_drugs_candidate[disease][drug]={}
                disease_drugs_candidate[disease][drug]["score"]=score
                disease_drugs_candidate[disease][drug]["association"]=line
                disease_drugs_candidate[disease][drug]["rank"]=1
            elif drug not in disease_drugs_candidate[disease]:
                disease_drugs_candidate[disease][drug]={}
                disease_drugs_candidate[disease][drug]["score"]=score
                disease_drugs_candidate[disease][drug]["association"]=line
                disease_drugs_candidate[disease][drug]["rank"]=1
            elif score > disease_drugs_candidate[disease][drug]["score"]:
                disease_drugs_candidate[disease][drug]["score"]=score
                disease_drugs_candidate[disease][drug]["association"]=line
                disease_drugs_candidate[disease][drug]["rank"]=1
            line=file_candidate_drugs.readline()
        file_gold_standard.close()

        print("finished loading candidate drugs ...")
        output_file=open(output_dir+"disease_drugs_ranking.txt","w")
        disease_drugs_ranking={}
        for disease in disease_drugs:
            for drug in disease_drugs[disease]:
                drug_ranking=1
                drug_score=disease_drugs[disease][drug]["score"]
                drug_association=disease_drugs[disease][drug]["association"]
                for drug_candidate in disease_drugs_candidate[disease]:
                    if disease_drugs_candidate[disease][drug_candidate]["score"] > drug_score:
                        drug_ranking += 1
                if drug_ranking > 100:
                    drug_ranking = 100
                output_file.write(disease+"\t"+drug+"\t"+drug_association.split("\t")[0]+"\t"+str(drug_ranking)+"\t"+str(drug_score)+"\n")
                if disease not in disease_drugs_ranking:
                    disease_drugs_ranking[disease]={}
                    disease_drugs_ranking[disease][drug]=drug_ranking
                elif drug not in disease_drugs_ranking[disease]:
                    disease_drugs_ranking[disease][drug]=drug_ranking
        output_file.close()
        mean_ranking=0
        hitsat10=0
        count=0
        hitsat10_count=0
        for disease in disease_drugs_ranking:
            for drug in disease_drugs_ranking[disease]:
                count += 1
                mean_ranking += disease_drugs_ranking[disease][drug]
                if disease_drugs_ranking[disease][drug] < 11:
                    hitsat10_count += 1
        print("meaning ranking:\t"+str(mean_ranking/count))
        print("hitsat10:\t"+str(hitsat10_count/count))

#KG_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/KnowledgeGraph/"
#TTD_cases_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TTD/"
#entity_relation_to_id_dir="/home/a914/zhaodi/Embedding/data/EmbeddingTrainingData 2/1990 to 2013/"

#KG_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/KnowledgeGraph/"
#positive_cases_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TTD/drug_target_disease_cases_for_drug_rediscovery_positive.txt"
#entity_relation_to_id_dir="/home/a914/zhaodi/Embedding/data/EmbeddingTrainingData 2/1990 to 2013/"
#candidate_drugs_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TTD/candidate_drugs_existsinKG.txt"
#output_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TTD/drug_target_disease_cases_for_drug_rediscovery_negative.txt"
#s=DrugRediscover()
#s.construct_drug_rediscovery_negative_cases(KG_file,positive_cases_file,entity_relation_to_id_dir,candidate_drugs_file,output_file)


#embedding_dir="/home/a914/zhaodi/Embedding/TransE/"
#gold_standard_cases_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TTD/drug_target_disease_cases_for_drug_rediscovery_positive.txt"
#candidate_cases_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/TTD/drug_target_disease_cases_for_drug_rediscovery_negative.txt"
#output_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/MeanRankandHits10/"
#model_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/Training_Models/LSTM_GraphAndOntology_Embedding/"
#s=DrugRediscover()
#s.drug_rediscovery(gold_standard_cases_file,candidate_cases_file,model_dir,embedding_dir,output_dir)
input_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/MeanRankandHits10/"
output_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/MeanRankandHits10/"
s=DrugRediscover()
s.analyze_MeanRanking_Hitsat10(input_dir,output_dir)
