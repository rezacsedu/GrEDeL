import numpy as np
import pickle
import gc
import copy
class Random_Walk_Algorithm:
    ## obtain_test_disease_drug_data函数的功能是获得100个disease—drug的数据，以用于Random Walk的算法来做对比实验
    #  因为使用最开始的方法（基于图的逻辑回归方法）已经得到了一部分实验结果，现在想做一个对比实验，所以需要用和最开始方法相同的数据才行。
    #  在obtain_experimental_data/obtain_experimental_data_(2)_3.py中的construct_test_vectors_of_examples函数因为是随机选的100个disease
    #  作为测试集，所以现在本函数也要选择和之前实验相同的disease作为验证才行。
    ##
    def obtain_test_disease_drug_data(self,KG_file,input_dir_disease_drug,output_dir):
        #load KG
        file=open(KG_file,"rb")
        KG_temp=pickle.load(file)
        file.close()
        print("--------------------------finished loading KG-----------------------------")
        KG={}
        for subject_type in KG_temp:
            subject=subject_type.split("_")[0]
            for object_type in KG_temp[subject_type]:
                object=object_type.split("_")[0]
                if object not in KG:
                    KG[object]={}
                    KG[object][subject]=len(KG_temp[subject_type][object_type])
                elif subject not in KG[object]:
                    KG[object][subject]=len(KG_temp[subject_type][object_type])
                else:
                    KG[object][subject] += len(KG_temp[subject_type][object_type])
        print("--------------------- KG length is:"+str(len(KG))+"-----------------------")

        file_gold_standard=open(input_dir_disease_drug+"gold_standard_drugs_scores.txt","r")
        disease_drugs={}
        line=file_gold_standard.readline()
        while line:
            #916758_9_44_44_316778_88_20_20_149215_104_12_12_677356_34_36_36_772252_34	0.144068
            sline=line.strip("\n").split("\t")
            ssline=sline[0].split("_")
            drug=ssline[0]
            disease=ssline[16]
            if disease not in disease_drugs:
                disease_drugs[disease]={}
                disease_drugs[disease][drug]=1
            else:
                disease_drugs[disease][drug]=1
            line=file_gold_standard.readline()
        file_gold_standard.close()

        print("----------------------finished loading gold standard cases--------------------------------")
        file_candidate_drugs=open(input_dir_disease_drug+"candidate_drugs_scores.txt","r")
        disease_drugs_candidate={}
        line=file_candidate_drugs.readline()
        while line:
            #916758_9_44_44_316778_88_20_20_149215_104_12_12_677356_34_36_36_772252_34	0.144068
            sline=line.strip("\n").split("\t")
            ssline=sline[0].split("_")
            drug=ssline[0]
            disease=ssline[16]
            if disease not in disease_drugs_candidate:
                disease_drugs_candidate[disease]={}
                disease_drugs_candidate[disease][drug]=1
            else:
                disease_drugs_candidate[disease][drug]=1
            line=file_candidate_drugs.readline()
        file_candidate_drugs.close()
        print("----------------------finished loading candidate cases--------------------------------")
        self.RWA(KG,disease_drugs,disease_drugs_candidate,output_dir,6)


    def RWA(self,KG,disease_drugs,disease_drugs_candidate,output_dir,steps):
        count=1
        for disease in disease_drugs:
            print(str(count)+":\t"+disease)
            count += 1
            step_temp={}
            if disease not in KG:
                print("该disease不在KG中:"+disease)
                continue
            else:
                for next_step_entity in KG[disease]:
                    step_temp[next_step_entity]=KG[disease][next_step_entity]
            output_file=open(output_dir+"RWA_"+str(1)+".txt","a+")
            for drug in disease_drugs[disease]:
                if drug not in step_temp:
                    output_file.write(drug+"\t"+"not found"+"\n")
                else:
                    drug_ranking=1
                    for candidate_drug in disease_drugs_candidate[disease]:
                        if candidate_drug in step_temp:
                            if step_temp[candidate_drug] >= step_temp[drug]:
                                drug_ranking += 1
                    if drug_ranking < 101:
                        output_file.write(drug+"\t"+str(drug_ranking)+"\n")
                    else:
                        output_file.write(drug+"\t"+str(101)+"\n")
            output_file.close()
            #还是该disease，开始走第2部以后
            for step in range(steps):
                output_file=open(output_dir+"RWA_"+str(step+2)+".txt","a+")
                step_next={}
                for this_step_entity in step_temp:
                    if this_step_entity in KG:
                        for next_step_entity in KG[this_step_entity]:
                            if next_step_entity not in step_next:
                                step_next[next_step_entity]= step_temp[this_step_entity]*KG[this_step_entity][next_step_entity]
                            else:
                                step_next[next_step_entity] += step_temp[this_step_entity]*KG[this_step_entity][next_step_entity]
                step_temp=copy.deepcopy(step_next)
                for drug in disease_drugs[disease]:
                    if drug not in step_temp:
                        output_file.write(drug+"\tnot found"+"\n")
                    else:
                        drug_ranking=1
                        for candidate_drug in disease_drugs_candidate[disease]:
                            if candidate_drug in step_temp:
                                if step_temp[candidate_drug] >= step_temp[drug]:
                                    drug_ranking += 1
                        if drug_ranking < 101:
                            output_file.write(drug+"\t"+str(drug_ranking)+"\n")
                        else:
                            output_file.write(drug+"\t"+str(101)+"\n")
                output_file.close()

    def analyze_RWA(self,input_dir,output_dir):
        file_output=open(output_dir+"analyze_RWA.txt","w+")
        for i in range(6):
            file_input=open(input_dir+"RWA_"+str(i+1)+".txt","r")
            count=0
            not_found=0
            hitsat10_count=0
            ranking_total=0
            line=file_input.readline()
            while line:
                rank=line.strip("\n").split("\t")[1]
                count += 1
                if rank == "not found":
                    ranking_total += 101
                    not_found += 1
                else:
                    ranking_total += int(rank)
                    if int(rank) < 10:
                        hitsat10_count += 1
                line=file_input.readline()
            file_output.write(str(i+1)+"\t"+str(not_found)+"\t"+str(ranking_total/count)+"\t"+str(hitsat10_count/count)+"\n")


#KG_file="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/KnowledgeGraph/KG"
#input_dir_disease_drug="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/MeanRankandHits10/"
#output_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/DrugDiscoveryRWA/"
#s=Random_Walk_Algorithm()
#s.obtain_test_disease_drug_data(KG_file,input_dir_disease_drug,output_dir)

input_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/DrugDiscoveryRWA/"
output_dir="/home/a914/zhaodi/LBD_paper_GraphEmbedding_LSTM/data/results/DrugDiscoveryRWA/"
s=Random_Walk_Algorithm()
s.analyze_RWA(input_dir,output_dir)

