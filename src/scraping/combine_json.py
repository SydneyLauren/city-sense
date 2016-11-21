import json
import glob
import pymongo
from pymongo.errors import DuplicateKeyError, CollectionInvalid

# Define the MongoDB database and table
conn = pymongo.MongoClient()
db = conn.tarvwstsyd2
collection = db.tarvwstsyd2_data
from collections import defaultdict
from bson.json_util import dumps

filenames = ['europe_city_reviews3.json', 'europe_city_reviews5.json', 'europe_city_reviews8.json']


# result = []
# for f in glob.glob("*.json"):
#     with open(f, "rb") as infile:
#         result.append(json.load(infile))
#
# with open("merged_ta.json", "wb") as outfile:
#      json.dump(result, outfile)

#
overall_dict = defaultdict(list)
for filename in filenames:
    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            for key in item.keys():
                if key == "_id":
                    continue
                overall_dict[key.strip().strip('\n').encode('ascii', 'ignore')].extend(item[key])
# print len(overall_dict[''])
# print len(overall_dict['Dresden, Germany'])
# asdasd
for key in overall_dict:
    doc = {key: overall_dict[key]}
    collection.insert(doc)

with open("ta_combined4.json", "wb") as f:
    f.write(dumps(list(collection.find())))
