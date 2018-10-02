__author__ = 'Shengtian'
import os
import pymysql
output_file="../data/PMIDs_1991_2010_sentences.txt"

conn=conn=pymysql.connect(host='localhost',user='root',passwd='123',db='semmeddb',port=3306)
cursor=conn.cursor()
print("successful connect!")

PMID_file="../data/PMIDs_1991_2010.txt"
output_sentence=open(output_file,'w')
rfile=open(PMID_file,'r')
line=rfile.readline()
while line:
    if line[-1] == '\n':
        newline=line[:-1]
    else:
        newline=line
    sql_getSentenceIDs_from_eachPMID='SELECT SENTENCE_ID,SENTENCE FROM SENTENCE WHERE PMID=\''+newline+ '\';'
    PMID=newline

    try:

        cursor.execute(sql_getSentenceIDs_from_eachPMID)
        ## sentencesIDs : ( (SENTENCE_ID,SENTENCE),(SENTENCE_ID,SENTENCE) )
        from_SENTENCES=cursor.fetchall()

        for senID_sentence in from_SENTENCES:
            ## senID_sentence : single (SENTENCE_ID,SENTENCE)
            senID_int=senID_sentence[0]
            senID=str(senID_int)

            sentence=senID_sentence[1]
            output_sentence.write(PMID+'|'+sentence+'|')
            ## QUESTION  if there is a '\n' at the end of senID
            ## get
            # PREDICATION_ID
            # SUBJECT_TEXT
            # SUBJECT_START_INDEX|SUBJECT_END_INDEX
            # INDICATOR_TYPE
            # PREDICATE_START_INDEX|PREDICATE_END_INDEX
            # OBJECT_TEXT
            # OBJECT_START_INDEX|OBJECT_END_INDEX
            sql_getfrom_SENTENCE_PREDICATION='SELECT PREDICATION_ID,SUBJECT_TEXT,SUBJECT_START_INDEX,SUBJECT_END_INDEX,INDICATOR_TYPE,PREDICATE_START_INDEX,PREDICATE_END_INDEX,OBJECT_TEXT,OBJECT_START_INDEX,OBJECT_END_INDEX FROM SENTENCE_PREDICATION WHERE SENTENCE_ID=\''+senID+ '\';'
            ##print("seq_getfrom_SENTENCE_PREDICATION :"+sql_getfrom_SENTENCE_PREDICATION)

            try:
                ##excute the slq_getfrom_SENTENCE_PREDICATION

                cursor.execute(sql_getfrom_SENTENCE_PREDICATION)
                ## from_SENTENCE_PREDICATION:
                # (
                # (PREDICATION_ID,
                # SUBJECT_TEXT,SUBJECT_START_INDEX,SUBJECT_END_INDEX,
                # INDICATOR_TYPE,PREDICATE_START_INDEX,PREDICATE_END_INDEX
                # OBJECT_TEXT,OBJECT_START_INDEX,OBJECT_END_INDEX)
                # ,(another the same format result)
                # )
                from_SENTENCE_PREDICATION=cursor.fetchall()
                if len(from_SENTENCE_PREDICATION) != 0:

                    for each_from_SENTENCE_PREDICATION in from_SENTENCE_PREDICATION:
                        each_PREDICATION_int=each_from_SENTENCE_PREDICATION[0]
                        each_PREDICATION=str(each_PREDICATION_int)

                        try :
                            ## STEP 1: use PREDICATION_ID to get data from PREDICATION
                            sql_getfrom_PREDICATION='SELECT PREDICATE FROM PREDICATION WHERE PREDICATION_ID=\''+each_PREDICATION+ '\';'

                            cursor.execute(sql_getfrom_PREDICATION)

                            PREDICATE_temp=cursor.fetchall()
                            PREDICATE=PREDICATE_temp[0][0]


                            ## STEP 2: use PREDICATION_id to get data from PREDICATION_ARGUMENT
                            sql_getfrom_PREDICATION_ARGUMENT='SELECT CONCEPT_SEMTYPE_ID,TYPE FROM PREDICATION_ARGUMENT WHERE PREDICATION_ID=\''+each_PREDICATION+ '\';'

                            cursor.execute(sql_getfrom_PREDICATION_ARGUMENT)

                            from_PREDICATION_ARGUMENT=cursor.fetchall()

                            ## from_PREDICATION_ARGUMENT :
                            # (
                            # (CONCEPT_SEMTYPE_ID,TYPE),
                            # (CONCEPT_SEMTYPE_ID,TYPE)
                            # )

                            if len(from_PREDICATION_ARGUMENT) != 0:


                                for i in from_PREDICATION_ARGUMENT:
                                    if i[1] == 'S':
                                        SUBJECT_ID_INT=i[0]
                                        SUBJECT_ID=str(SUBJECT_ID_INT)

                                    else:
                                        OBJECT_ID_INT=i[0]
                                        OBJECT_ID=str(OBJECT_ID_INT)


                                ## STEP final: use SUBJECT_ID & OBJECT_ID to get data from CONCEPT_SEMTYPE
                                try:

                                    sql_getfrom_CONCEPT_SEMTYPE_SUBJECT='SELECT SEMTYPE FROM CONCEPT_SEMTYPE WHERE CONCEPT_SEMTYPE_ID=\''+SUBJECT_ID+ '\';'
                                    cursor.execute(sql_getfrom_CONCEPT_SEMTYPE_SUBJECT)
                                    SUBJECT_TYPE_temp=cursor.fetchall()
                                    SUBJECT_TYPE=SUBJECT_TYPE_temp[0][0]

                                    sql_getfrom_CONCEPT_SEMTYPE_OBJECT='SELECT SEMTYPE FROM CONCEPT_SEMTYPE WHERE CONCEPT_SEMTYPE_ID=\''+OBJECT_ID+ '\';'
                                    cursor.execute(sql_getfrom_CONCEPT_SEMTYPE_OBJECT)
                                    OBJECT_TYPE_temp=cursor.fetchall()
                                    OBJECT_TYPE=OBJECT_TYPE_temp[0][0]

                                    ## write to file
                                    for ith_from_SENTENCE_PREDICATION in from_SENTENCE_PREDICATION:

                                        SUBJECT_TEXT=ith_from_SENTENCE_PREDICATION[1]
                                        SUBJECT_START_INDEX=str(ith_from_SENTENCE_PREDICATION[2])
                                        SUBJECT_END_INDEX=str(ith_from_SENTENCE_PREDICATION[3])
                                        INDICATOR_TYPE=ith_from_SENTENCE_PREDICATION[4]
                                        PREDICATE_START_INDEX=str(ith_from_SENTENCE_PREDICATION[5])
                                        PREDICATE_END_INDEX=str(ith_from_SENTENCE_PREDICATION[6])
                                        OBJECT_TEXT=ith_from_SENTENCE_PREDICATION[7]
                                        OBJECT_START_INDEX=str(ith_from_SENTENCE_PREDICATION[8])
                                        OBJECT_END_INDEX=str(ith_from_SENTENCE_PREDICATION[9])
                                        output_sentence.write(SUBJECT_TEXT+'|'+SUBJECT_TYPE+'|'+SUBJECT_START_INDEX+'|'+SUBJECT_END_INDEX+'|')
                                        output_sentence.write(PREDICATE+'|'+INDICATOR_TYPE+'|'+PREDICATE_START_INDEX+'|'+PREDICATE_END_INDEX+'|')
                                        output_sentence.write(OBJECT_TEXT+'|'+OBJECT_TYPE+'|'+OBJECT_START_INDEX+'|'+OBJECT_END_INDEX+'|')

                                        ##output_sentence.write(SUBJECT_TEXT+'|'+SUBJECT_TYPE+'|'+SUBJECT_START_INDEX+'|'+SUBJECT_END_INDEX+'|'+PREDICATE+'|'+INDICATOR_TYPE+'|'+OBJECT_TEXT+'|'+OBJECT_TYPE+'|'+OBJECT_START_INDEX+'|'+OBJECT_END_INDEX+'|')

                                except  Exception:print('ERROR: use SUBJECT_ID & OBJECT_ID to get data from CONCEPT_SEMTYPE\n')
                        except  Exception:print('ERROR: use each_PREDICATION to get data from PREDICATION & PREDICATION_ARGUMENT \n')

                output_sentence.write('\n')
            except  Exception:print('ERROR: use SENTENCE_ID to get data from SENTENCE_PREDICATION\n')
    except  Exception:print('ERROR: use PMID to get data from SENTENCE \n')


    line=rfile.readline()
output_sentence.close()
rfile.close()

cursor.close()
conn.close()
print("Finished")
