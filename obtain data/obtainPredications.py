import os
import pymysql
conn=pymysql.connect(host='localhost',user='root',passwd='123',db='semmeddb',port=3306)
pmidsfile=open("../data/PMIDs_1991_2010.txt","r")
output_file=open("../data/PMIDs_1991_2010_sentences.txt","w")
pmid=pmidsfile.readline().strip("\n")
print "start collecting data from mysql..."
pmid_num=0
cur1=conn.cursor()
while pmid:
    sql='SELECT SENTENCE_ID,SENTENCE FROM SENTENCE WHERE PMID=\''+pmid+ '\';'
    try:
        cur1.execute(sql)
        SENTENCES_SID=cur1.fetchall()
        pmid_num += 1
        if (pmid_num%100)==1:
            print "pmid: "+pmid
            print str(pmid_num)
            print "finish searching from SENTENCE table..."
        for sentence_sid in SENTENCES_SID:
            print "==========================================="
            senID_int=sentence_sid[0]
            senID=str(senID_int)
            sentence=sentence_sid[1]
            print senID+"|"+sentence
            output_file.write(pmid+'|'+sentence+'|')
            sql_getfrom_SENTENCE_PREDICATION='SELECT PREDICATION_ID,SUBJECT_TEXT,SUBJECT_START_INDEX,SUBJECT_END_INDEX,INDICATOR_TYPE,PREDICATE_START_INDEX,PREDICATE_END_INDEX,OBJECT_TEXT,OBJECT_START_INDEX,OBJECT_END_INDEX FROM SENTENCE_PREDICATION WHERE SENTENCE_ID=\''+senID+ '\';'
            try:
                cur1.execute(sql_getfrom_SENTENCE_PREDICATION)
                from_SENTENCE_PREDICATION=cur1.fetchall()
                for each_from_SENTENCE_PREDICATION in from_SENTENCE_PREDICATION:
                    each_PREDICATION_int=each_from_SENTENCE_PREDICATION[0]
                    each_PREDICATION=str(each_PREDICATION_int)
                    try :
                        ## STEP 1: use PREDICATION_ID to get data from PREDICATION
                        sql_getfrom_PREDICATION='SELECT PREDICATE FROM PREDICATION WHERE PREDICATION_ID=\''+each_PREDICATION+ '\';'
                        cur1.execute(sql_getfrom_PREDICATION)
                        PREDICATE_temp=cur1.fetchall()
                        PREDICATE=PREDICATE_temp[0][0]
                        ## STEP 2: use PREDICATION_id to get data from PREDICATION_ARGUMENT
                        sql_getfrom_PREDICATION_ARGUMENT='SELECT CONCEPT_SEMTYPE_ID,TYPE FROM PREDICATION_ARGUMENT WHERE PREDICATION_ID=\''+each_PREDICATION+ '\';'
                        cur1.execute(sql_getfrom_PREDICATION_ARGUMENT)
                        from_PREDICATION_ARGUMENT=cur1.fetchall()
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
                                        cur1.execute(sql_getfrom_CONCEPT_SEMTYPE_SUBJECT)
                                        SUBJECT_TYPE_temp=cur1.fetchall()
                                        SUBJECT_TYPE=SUBJECT_TYPE_temp[0][0]
                                        sql_getfrom_CONCEPT_SEMTYPE_OBJECT='SELECT SEMTYPE FROM CONCEPT_SEMTYPE WHERE CONCEPT_SEMTYPE_ID=\''+OBJECT_ID+ '\';'
                                        cur1.execute(sql_getfrom_CONCEPT_SEMTYPE_OBJECT)
                                        OBJECT_TYPE_temp=cur1.fetchall()
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
                                            output_file.write(SUBJECT_TEXT+'|'+SUBJECT_TYPE+'|'+SUBJECT_START_INDEX+'|'+SUBJECT_END_INDEX+'|')
                                            output_file.write(PREDICATE+'|'+INDICATOR_TYPE+'|'+PREDICATE_START_INDEX+'|'+PREDICATE_END_INDEX+'|')
                                            output_file.write(OBJECT_TEXT+'|'+OBJECT_TYPE+'|'+OBJECT_START_INDEX+'|'+OBJECT_END_INDEX+'|\n')

                                    except Exception: print "errors: when retrieving details from CONCEPT_SEMTYPE\n"

                    except Exception: print "errors: when retrieving details from PREDICATION\n"

            except Exception: print "errors: when retrieving details from SENTENCE_PREDICATION\n"

    except  Exception: print "errors: when retrieving pmid from SENTENCE\n"
    pmid=pmidsfile.readline()

cur1.close()
conn.close()
output_file.close()
pmidsfile.close()
print "there are "+str(pmid_nums)+" abstracts\n"


