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
import matplotlib.pyplot as plt
import unicodedata


def json_to_sentencedict(filename):
    '''
    INPUT: name of json file
    OUTPUT: dictionary with city keys and description values
    take json file and return dictionary or original text
    '''
    di = dict()
    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            for key in item.keys():
                unique_text = (set(item[key]))
                txt = ''.join(c for c in unique_text)
                txt = ' '.join(ch for ch in txt.split() if ch != '\n')
                txt.replace('\u2019', "'")
                txt = ''.join((c for c in unicodedata.normalize('NFD', unicode(txt)) if unicodedata.category(c) != 'Mn'))
                di[key.strip().strip('\n').encode('ascii', 'ignore')] = txt.encode('ascii', 'ignore')
    return di


def json_to_dict(filename):
    '''
    INPUT: name of json file
    OUTPUT: dictionary with city keys and description values
    take json file and return dictionary of cleaned up text
    '''
    di = {}
    english = []
    with open('../data/wordlist.txt') as f:
        for line in f:
            english.append(line.strip('\r\n'))
            english.append(' ')

    english = set(english)

    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            for key in item.keys():
                unique_text = set(item[key])
                # remove punctuation, make everything lower case
                txt = ''.join(ch.lower() for ch in unique_text if ch not in set(string.punctuation))
                # remove numbers
                txt = ''.join(c for c in txt if c.isdigit() is False)
                # remove the city and country name from its own description
                keyparts = key.split(',')
                txt = ' '.join(c for c in txt.split() if c.lower() != keyparts[0].lower() and c.lower() != keyparts[-1].lower())
                # remove non-english words
                txt = ' '.join(c for c in txt.split() if c in english)
                # populate dictionary
                di[key.strip().strip('\n').encode('ascii', 'ignore')] = txt
    return di


def calculate_cosine_sim(doc_bodies, check_words, stops):
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english') + stops)

    X = vectorizer.fit_transform(np.append(doc_bodies, check_words))
    cosine_similarities = linear_kernel(X, X)

    return cosine_similarities[-1, :-1]


def plot_personality_map(m, cosine_similarities, xpts, ypts, citylist):
    fig = plt.figure(figsize=(19, 9))
    cosine_similarities = np.array(cosine_similarities)
    sz = 20. * cosine_similarities / max(cosine_similarities) + 6
    alp = (cosine_similarities / max(cosine_similarities))/1.5 + 0.3
    # fs = 15. * cosine_similarities / max(cosine_similarities)
    fs = np.zeros(len(cosine_similarities))
    alp = np.zeros(len(cosine_similarities))
    cs_sort = np.argsort(cosine_similarities)[::-1]

    n1 = 4
    n2 = 12
    n3 = 40
    alp[cs_sort[:n1]] = 1
    alp[cs_sort[n1:n2]] = 0.50
    alp[cs_sort[n2:n3]] = 0.40
    alp[cs_sort[n3:]] = 0.2

    fs[cs_sort[:n1]] = 10
    fs[cs_sort[n1:n2]] = 8
    fs[cs_sort[n2:n3]] = 6
    fs[cs_sort[n3:]] = 4



    for i in xrange(len(xpts)):
        plt.plot(xpts[i], ypts[i], '.', markersize=13, color='#81D8D0', alpha=0.8)
        # plt.plot(xpts[i], ypts[i], '.', markersize=sz[i], color='#81D8D0', alpha=alp[i])
        # plt.text(xpts[i], ypts[i], citylist[i], fontsize=int(fs[i]), alpha=alp[i], color=[1,1,1])
    m.drawcoastlines(linewidth=0.2)
    m.drawcountries(linewidth=0.2)
    m.drawmapboundary(fill_color='#2B3856')
    # fill continents, set lake color same as ocean color.
    m.fillcontinents(color='#786D5F', lake_color='#2B3856')

    plt.savefig('city_sample.png', bbox_inches='tight')
    plt.show()


rs_dict = json_to_dict('../data/ricksteves_articles_blogs_R01.json')
ta_dict = json_to_dict('../data/europe_city_reviews2.json')

rs_dict_comp = json_to_sentencedict('../data/ricksteves_articles_blogs_R01.json')
ta_dict_comp = json_to_sentencedict('../data/europe_city_reviews2.json')




key_list_comp = set(rs_dict_comp.keys() + ta_dict_comp.keys())
europe_dict_comp = dict()
for key in key_list_comp:
    l1 = str(rs_dict_comp.get(key)).split('. ')
    l2 = str(ta_dict_comp.get(key)).split('. ')
    europe_dict_comp[key] = l1 + l2


key_list = set(rs_dict.keys() + ta_dict.keys())
europe_dict = dict()

for key in key_list:
    europe_dict[key] = str(rs_dict.get(key)) + str(ta_dict.get(key))

# remove cities which contain little or no text
europe_dict = {key: value for key, value in europe_dict.items() if len(value) > 200}
europe_dict_comp = {key: value for key, value in europe_dict_comp.items() if key in europe_dict}
for k in europe_dict:
    if len(europe_dict[k]) < 100:
        print '\n', k
        print europe_dict[k]
print len(europe_dict.keys())
print len(europe_dict_comp.keys())

# Convert dictionary into dataframe
# europe_dict['Washington, England'] = europe_dict.pop('Washington, United Kingdom')
# europe_dict['London, England'] = europe_dict.pop('London England United Kingdom')
print 'check for text scrambling:'
print europe_dict_comp['Belgrade, Serbia']
asdasdad
cities_df = pd.DataFrame.from_dict(europe_dict, orient='index', dtype=None)
cities_df.columns = ['description']

# Extract cities and their descripitions from dataframe
# doc_bodies = cities_df['description']
doc_bodies = cities_df['description'].values
tokenized_corpus = [word_tokenize(content.lower()) for content in doc_bodies]
cities = cities_df.index.values

with open('../data/personalities.txt') as f:
    for line in f:
        personality = line.strip('\n').split(', ') # read words on each line of file
#         personality = ['great', 'awesome']
        cosine_similarities = calculate_cosine_sim(doc_bodies, ' '.join(personality), [])
        top5 = np.argsort(cosine_similarities)[::-1][:5]

        # pull from original calculation
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        Xorig = vectorizer.fit_transform(np.append(doc_bodies, ' '.join(personality)))
        cs = linear_kernel(Xorig, Xorig)

        # test a new calculation
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        X = vectorizer.fit_transform(doc_bodies)

        c = vectorizer.transform(np.array([' '.join(personality), ]))

        csnew = X*c.T
        data = cosine_similarity(X, c)


# with open('../data/base_map.pkl') as f:
#     m = pickle.load(f)

# this was pretty good
# m = Basemap(projection='stere', lon_0=5, lat_0=72.0, rsphere=6371200., llcrnrlon=-15.0,
#             urcrnrlon=72.0, llcrnrlat=32.0, urcrnrlat=50.0, resolution='l')

m = Basemap(projection='stere', lon_0=5, lat_0=60.0, rsphere=6371200., llcrnrlon=-15.0,
            urcrnrlon=60.0, llcrnrlat=32.0, urcrnrlat=56.0, resolution='l')

# load the city coordinates
# with open('../data/coordinates.txt') as f:
#     coords = np.array([[float(coord) for coord in line.strip('\n').split('|')[1:]] for line in f])

x = []
y = []
geolocator = Nominatim()
for city in cities:
    location = geolocator.geocode(city, timeout=10)
    xpt, ypt = m(location.longitude, location.latitude)
    x.append(xpt)
    y.append(ypt)

cs = np.ones(len(cosine_similarities))
# plot_personality_map(m, cs, coords[:, 0], coords[:, 1], cities)
plot_personality_map(m, cs, x, y, cities)
plt.show()
