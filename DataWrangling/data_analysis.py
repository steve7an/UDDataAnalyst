import pprint
def run_aggregate_result(pipeline, db):
    results = db.osm.aggregate(pipeline)
    #print type(results)
    
    for result in results:
        pprint.pprint(result)

####
query_title = "#### initialize the db"
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db_sg = client.singapore

print query_title
db_sg.osm.find_one()

####
## https://code.activestate.com/recipes/577081-humanized-representation-of-a-number-of-bytes/
def humanize_bytes(bytes, precision=1):
    """Return a humanized string representation of a number of bytes.

    Assumes `from __future__ import division`.

    >>> humanize_bytes(1)
    '1 byte'
    >>> humanize_bytes(1024)
    '1.0 kB'
    >>> humanize_bytes(1024*123)
    '123.0 kB'
    >>> humanize_bytes(1024*12342)
    '12.1 MB'
    >>> humanize_bytes(1024*12342,2)
    '12.05 MB'
    >>> humanize_bytes(1024*1234,2)
    '1.21 MB'
    >>> humanize_bytes(1024*1234*1111,2)
    '1.31 GB'
    >>> humanize_bytes(1024*1234*1111,1)
    '1.3 GB'
    """
    abbrevs = (
        (1<<50L, 'PB'),
        (1<<40L, 'TB'),
        (1<<30L, 'GB'),
        (1<<20L, 'MB'),
        (1<<10L, 'kB'),
        (1, 'bytes')
    )
    if bytes == 1:
        return '1 byte'
    for factor, suffix in abbrevs:
        if bytes >= factor:
            break
    return '%.*f %s' % (precision, bytes / factor, suffix)

####
query_title = "#### Checking osm and json file size"
import os
osmfilename = 'singapore.osm'
jsonfilename = "singapore.osm2.json"
print query_title
print osmfilename, humanize_bytes(os.path.getsize(osmfilename))
print jsonfilename, humanize_bytes(os.stat(jsonfilename).st_size)

####
print "dbstats():"
dbstats = db_sg.command("dbstats")
#pprint.pprint (dbstats)

for k, v in dbstats.iteritems():
    if "Size" in k:
        print k, humanize_bytes(v)
    else:
        print k, (v)


####
print "#### Number of nodes and ways:"
print "Number of documents:", "{:,}".format(db_sg.osm.find().count())
print "Number of nodes:", "{:,}".format(db_sg.osm.find({"type":"node"}).count())
print "Number of ways:", "{:,}".format(db_sg.osm.find({"type":"way"}).count())
print "Number of unique users:", "{:,}".format(len(db_sg.osm.find().distinct("created.user")))


####
match = {"$match": {'created.user': {"$exists": 1} }}
group = {"$group": {"_id": '$created.user', "count": {"$sum": 1}}}
sort = {"$sort": {'count': -1}}
limit = {"$limit": 10}

print "#### Top ten users that contributed the most:"
pipeline = [match,group,sort,limit]
run_aggregate_result(pipeline, db_sg)

####
project = {"$project": 
          {"street":"$address.street","_id":0}
          }
match = {"$match": 
          {"$and":[{"address.street" : {"$exists":"true"}},{"address.postcode":{"$regex":"^[0-9]{6}$"}} ]}
         }
sort = {"$sort": {"count" : -1}}
group = { "$group": { "_id": "$street",
                    "count": {"$sum":1}}}
limit = {"$limit": 10}

print "#### Top ten street that appears the most in this data set:"
pipeline = [match, project, group, sort, limit]
run_aggregate_result(pipeline, db_sg)


####
match = {"$match": {'amenity': {"$exists": 1} }}
group = {"$group": {"_id": '$amenity', "count": {"$sum": 1}}}
sort = {"$sort": {'count': -1}}
limit = {"$limit": 10}

print "#### Top ten amenities that appeared the most in this data set:"
pipeline = [match, group, sort, limit]
run_aggregate_result(pipeline, db_sg)


####
query_title = "#### Finding the number of malls in Singapore"
match = {"$match": {'shop': {"$exists": 1} }}
group = {"$group": {"_id": '$shop', "count": {"$sum": 1}}}
sort = {"$sort": {'count': -1}}
limit = {"$limit": 10}

print query_title
pipeline = [match, group, sort, limit]
run_aggregate_result(pipeline, db_sg)


####
query_title = "#### Finding out what shops are labeled with yes:"
match = {"$match": {'shop': "yes"}}
limit = {"$limit": 10}
group = {"$group": {"_id": '$name', "count": {"$sum": 1}}}
sort = {"$sort": {'count': -1}}

print query_title
pipeline = [match, group, sort, limit]
run_aggregate_result(pipeline, db_sg)


####
query_title = "#### Finding the top three bah kut teh restaurants in Singapore:"
match = {"$match": 
         {"$or":[{"name": {"$regex":"k.+t teh", "$options": "-i"}},
          {"website": "http://www.songfa.com.sg/"}]
        }}
group = {"$group": {"_id": '$name', "count": {"$sum": 1}}}
sort = {"$sort": {'count': -1}}

print query_title
pipeline = [match, group, sort]
run_aggregate_result(pipeline, db_sg)