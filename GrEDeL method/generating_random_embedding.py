import random
class GeneratingRandom:
    def generating_random_embedding(self,input_dir,output_dir):
        file_in=open(input_dir+"relation2vec.bern","r")
        file_output=open(output_dir+"relation2vec.bern","w+")
        line=file_in.readline()
        while line:
            for i in range(24):
                file_output.write(str("%.5f" %random.uniform(-1,1))+"\t")
            file_output.write(str("%.5f" %random.uniform(-1,1))+"\n")
            line=file_in.readline()
        file_in.close()
        file_output.close()

        file_in=open(input_dir+"entity2vec.bern","r")
        file_output=open(output_dir+"entity2vec.bern","w+")
        line=file_in.readline()
        while line:
            for i in range(24):
                file_output.write(str("%.5f" %random.uniform(-1,1))+"\t")
            file_output.write(str("%.5f" %random.uniform(-1,1))+"\n")
            line=file_in.readline()
        file_in.close()
        file_output.close()

input_dir="/home/a914/zhaodi/Embedding/TransE/25/"
output_dir="/home/a914/zhaodi/Embedding/Random_Embedding/25/"
s=GeneratingRandom()
s.generating_random_embedding(input_dir,output_dir)
