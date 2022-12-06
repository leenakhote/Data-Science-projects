import json
import time
import sys
from elasticsearch import Elasticsearch 
from elasticsearch.helpers import bulk
import csv
import tensorflow as tf
import tensorflow_hub as hub


def indexQuestions(es,embed):
    # index in ES = DB in an RDBMS
    # Read each question and index into an index called questions
    # Indexing only titles for this example to improve speed. In practice, its good to index CONCATENATE(title+body)
    # Define the index
    
    if es.indices.exists(index="questions-index")==False:

        #Refer: https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html
        # Mapping: Structure of the index
        # Property/Field: name and type  
        b = {"mappings": {
            "properties": {
                    "title": {
                        "type": "text"
                    },
                    "title_vector": {
                        "type": "dense_vector",
                        "dims": 512
                }
            }
            }
        }


        ret = es.indices.create(index='questions-index',ignore=400 , body=b) #400 caused by IndexAlreadyExistsException, 
        print(json.dumps(ret,indent=4))

        # TRY this in browser: http://localhost:9200/questions-index

        print("questions-index created")
        print("*********************************************************************************")

        #load USE4 model
        #print("Loading USE model......");
        #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        #embed = hub.load("pretrained_models/universal-sentence-encoder_4")


        # CONSTANTS
        NUM_QUESTIONS_INDEXED = 1000

        # Col-Names: Id,OwnerUserId,CreationDate,ClosedDate,Score,Title,Body
        cnt=0

        with open('./data/Questions.csv', encoding="latin1") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',' )
            next(readCSV, None)  # skip the headers 
            for row in readCSV:
                #print(row[0], row[1])
                doc_id = row[0];
                title = row[1];
                vec = tf.make_ndarray(tf.make_tensor_proto(embed([title]))).tolist()[0]		
                
                b = {"title":title,
                    "title_vector":vec,
                    }	
                #print(json.dumps(tmp,indent=4))		
                
                res = es.index(index="questions-index", id=doc_id, body=b)
                #print(res)
                

                # keep count of # rows processed
                cnt += 1
                if cnt%100==0:
                    print(cnt)
                
                if cnt == NUM_QUESTIONS_INDEXED:
                    break;

            print("Completed indexing....")

            print("*********************************************************************************")
        
    else:
        print("questions-index already present so indexing was skipped")
        print("*********************************************************************************")
        pass

    return
