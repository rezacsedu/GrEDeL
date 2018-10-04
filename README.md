# GrEDeL
GrEDeL (Graph Embedding based Deep Learning) is a literature-based discovery framework which is mainly composed of 1) knowledge graph embedding method and 2) deep learning method.

obtain_data file:
obtainPredications.py and obtain_data.py obtains detailed predications from SemmedDB for constrcuting the biomedical knowledge graph, one predication is mainly composed of one subject, one object and the ralation between the subject and object. Specifically, we also obtain the UMLS semantic types of the subject, object and relation.
PMIDs_1991_2010_sentences.txt contains some examples of pubmed abstracts.

knowledge graph embedding file:
constructing_Embedding_data.py is used to prepare data for learning graph embedding by Train_TransE.cpp.
Train_TransE.cpp is the knowledge graph embedding method.

GrEDeL file:
LSTM_training_GraphOntology.py is the main function of our method GrEDeL.


