import os
class analyze_results:
    ##分析 Precision  Recall  F-score
    def analyze_PRF(self,result_dir):
        files=os.listdir(result_dir)
        outputfile=open(result_dir+"PRF.txt","w+")
        for file in files:
            print(file)
            outputfile.write("\n"+file.strip(".txt")+":"+"\n")
            Precision=0
            Recall=0
            F_score=0
            result_file=open(result_dir+file,"r")
            #[1]:0.915404	[0]:0.0150679	[1]:0.968263	[1]:0.937038	[1]:0.940658	[0]:0.104438
            line=result_file.readline()
            total_precision=0
            total_recall=0
            total_f_score=0
            cross_validation_count=0
            while line:
                if line.split("\t")[0]=="cross validation":

                    line=result_file.readline()
                    continue
                else:
                    TP=0
                    TN=0
                    FN=0
                    FP=0
                    sline=line.strip("\n\t ").split("\t")
                    for couple in sline:
                        s_couple=couple.split(":")
                        if len(s_couple)>2:
                            print(couple)
                            continue
                        question=int(s_couple[0].strip("]["))
                        predicate=float(s_couple[1])
                        if question==0 and predicate < 0.5:
                            TN += 1
                        elif question==0 and predicate > 0.5:
                            FP += 1
                        elif question==1 and predicate < 0.5:
                            FN += 1
                        elif question==1 and predicate > 0.5:
                            TP += 1
                        else:
                            print("wrong:"+couple)

                    precision=(TP)/(TP+FP)
                    recall=(TP)/(TP+FN)
                    f_score=2*precision*recall/(precision+recall)
                    #print(str(precision)+"\t"+str(recall)+"\t"+str(f_score))
                    cross_validation_count += 1
                    total_precision += precision
                    total_recall += recall
                    total_f_score += f_score
                line=result_file.readline()
            outputfile.write("Precision: %.3f" %(total_precision/cross_validation_count)+"\n")
            outputfile.write("Recall:%.3f" %(total_recall/cross_validation_count)+"\n")
            outputfile.write("F-score:%.3f" %(total_f_score/cross_validation_count)+"\n")


result_dir="/home/a914/zhaodi/LBD_paper_6_AttentionBasedLSTM/data/results/ComparativeMethods/LSTM/results/"
s=analyze_results()
s.analyze_PRF(result_dir)