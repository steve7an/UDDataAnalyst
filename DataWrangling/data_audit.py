
import pprint
def run_aggregate_result(pipeline, db):
    results = db.osm.aggregate(pipeline)
    #print type(results)
    
    for result in results:
        pprint.pprint(result)


from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client.sample

####
query_title = "#### Incorrect postal code - checking"
project = {"$project": 
          {"postcode":"$address.postcode", "_id":0}
          }
match = {"$match": 
          {"address.postcode" : {"$exists":"true"}}
         }

print query_title
pipeline = [match, project]
run_aggregate_result(pipeline, db)


####
query_title = "#### Looking into address details of three selected postcode"
match = {"$match": 
          {"$or": [{"address.postcode" : "81300"},{"address.postcode" : "688688"}
                   ,{"address.postcode" : "<different>"}]}
         }
project = {"$project": 
          {"address":"$address", "_id":0}
          }
sort = {"$sort": {"count" : -1}}
group = { "$group": { "_id": "$address",
                      "count": {"$sum":1}}}

print query_title
pipeline = [match, project, group, sort]
run_aggregate_result(pipeline, db)


####
query_title = "#### Abbrevation used in the road names or building name - Checking"
project = {"$project": 
          {"street":"$address.street","_id":0}
          }
match = {"$match": 
          {"$and":[{"address.street" : {"$exists":"true"}},{"address.postcode":{"$regex":"[0-9]{6}"}} ]}
         }
sort = {"$sort": {"count" : -1}}
group = { "$group": { "_id": "$street",
                    "count": {"$sum":1}}}

print query_title
pipeline = [match, project, sort, group]
run_aggregate_result(pipeline, db)


####
query_title = "#### Inconsistencies on other part of the address - Checking"
project = {"$project": 
          {"address":"$address", "_id": 0}
          }
match = {"$match": 
          {"address" : {"$exists":"true"}}
         }

print query_title
pipeline = [match, project]
run_aggregate_result(pipeline, db)


query_title = "#### Getting the details for the way which contains only the house number or house name:"
print query_title

result = db.osm.find_one({"address.housenumber":"58"})
pprint.pprint(result)

result = db.osm.find_one({"address.housename":"XB1"})
pprint.pprint(result)


#### 
query_title = "#### Check if we're able to pull address near to house number 58 using node ref"

project = {"$project": 
          {"address":"$address", "id":"$id", "type":"$type"}
          }
match = {"$match": 
          {"$and":[{"address" : {"$exists":"true"}}, 
                   {"node_refs":{"$gte":1826326424}}, {"node_refs":{"$lte":1826326478}} ]}
         }

print query_title
pipeline = [match, project]
run_aggregate_result(pipeline, db)