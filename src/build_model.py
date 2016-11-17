import json
import pandas as pd
from collections import defaultdict
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import string
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from geopy.geocoders import Nominatim
import pickle
from mpl_toolkits.basemap import Basemap
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import itertools
import numpy.linalg as LA


def json_to_dict(filename):
    '''
    INPUT: name of json file
    OUTPUT: dictionary with city keys and description values
    take json file and return dictionary
    '''
    doc_dict = {}
    with open('../data/wordlist.txt') as f:
        english = set([line.strip('\r\n') for line in f])

    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            for key in item.keys():
                unique_text = set(item[key])  # check for non-unique entries (duplicate reviews)

                txt = ''.join(ch.lower() for ch in unique_text  # remove punctuation, make everything lower case
                              if ch not in set(string.punctuation))

                txt = ''.join(c for c in txt if c.isdigit() is False)  # remove numbers

                txt = ' '.join(c for c in txt.split() if  # remove the city and country name from its own description
                               c.lower() != key.split(', ')[0].lower() and c.lower() != key.split(', ')[-1].lower())

                txt = ' '.join(c for c in txt.split() if c in english)  # remove non-english words

                doc_dict[key.strip().strip('\n').encode('ascii', 'ignore')] = txt  # populate dictionary
    return doc_dict


def combine_dictionaries(dict1, dict2):
    '''
    INPUT: two dictionaries
    OUTPUT: one dictionary that combines the two dictionaries wherever keys match
    take two dictionaries and merge them
    '''
    key_list = set(rs_dict.keys() + ta_dict.keys())
    merged_dict = dict()
    for key in key_list:
        merged_dict[key] = str(rs_dict.get(key)) + str(ta_dict.get(key))
    return merged_dict


def generate_dataframe(d):
    df = pd.DataFrame.from_dict(d, orient='index', dtype=None)
    df.columns = ['description']
    df.to_pickle('../data/cities_dataframe.pkl')
    return df


def generate_basemap(cities):
    '''
    INPUT: list of cities
    OUTPUT: pickled basemap object and text file containing basemap coordinates for cities
    generate a basemap object and list of city basemap coordinates for future plotting
    '''
    m = Basemap(projection='stere', lon_0=5, lat_0=72.0, rsphere=6371200., llcrnrlon=-15.0,
                urcrnrlon=74.0, llcrnrlat=32.0, urcrnrlat=55.0, resolution='l')
    pickle.dump(m, file('../data/base_map.pkl', 'w'))
    geolocator = Nominatim()

    with open('../data/coordinates.txt', 'w') as f:
        for city in cities:
            location = geolocator.geocode(city, timeout=10)
            xpt, ypt = m(location.longitude, location.latitude)
            f.write('{}|{}|{}\n'.format(city, xpt, ypt))


if __name__ == '__main__':
    # read json files into dictionary format
    rs_dict = json_to_dict('../data/ricksteves_articles_blogs_R01.json')
    ta_dict = json_to_dict('../data/europe_city_reviews2.json')

    # combine dictionaries
    europe_dict = combine_dictionaries(rs_dict, ta_dict)

    # remove cities which contain little or no text
    europe_dict = {key: value for key, value in europe_dict.items() if len(value) > 200}

    # Convert dictionary into dataframe and save
    cities_df = generate_dataframe(europe_dict)

    # build and save vectorizer
    stops = []
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english') + stops)
    with open('../data/my_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # generate basemap object and city coordinates for plotting
    generate_basemap(cities_df.index.values)
