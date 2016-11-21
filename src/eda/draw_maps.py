import json
from collections import defaultdict
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
import networkx as nx
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import string
from nltk.tokenize import word_tokenize
from mpl_toolkits.basemap import Basemap
from sklearn.metrics.pairwise import linear_kernel
from geopy.geocoders import Nominatim
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import itertools
import matplotlib.pyplot as plt


def json_to_dict(filename):
    '''
    INPUT: name of json file
    OUTPUT: dictionary with city keys and description values
    take json file and return dictionary
    '''
    di = {}
    english = []
    with open('wordlist.txt') as f:
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
                keypts = key.split(',')
                txt = ' '.join(c for c in txt.split() if c.lower() != keypts[0].lower() and c.lower() != keypts[-1].lower())
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

    cosine_similarities = np.array(cosine_similarities)
    sz = 20. * cosine_similarities / max(cosine_similarities) + 6
    alp = (cosine_similarities / max(cosine_similarities))/1.5 + 0.3
    fs = 15 * cosine_similarities / max(cosine_similarities)

    for i in xrange(len(xpts)):
        plt.plot(xpts[i], ypts[i], '.', markersize=sz[i], color='#81D8D0', alpha=alp[i])
        plt.text(xpts[i], ypts[i], citylist[i], fontsize=int(fs[i]), alpha=alp[i], color=[1, 1, 1])

    m.drawcoastlines(linewidth=0.2)
    m.drawcountries(linewidth=0.2)
    m.drawmapboundary(fill_color='#2B3856')
    # fill continents, set lake color same as ocean color.
    m.fillcontinents(color='#786D5F', lake_color='#2B3856')

    plt.savefig('city_sample.jpg', bbox_inches='tight')
    plt.show()


def plot_initializer(coordinates):
    '''
    INPUT: coordinates for a set of shapes
    OUTPUT: plot handle and basemap object

    Takes coordinates for a set of shapes and initializes a basemap figure'''

    w, h = coordinates[2] - coordinates[0], coordinates[3] - coordinates[1]
    extra = 0.01

    figwidth = 8
    fig = plt.figure(figsize=(figwidth, figwidth*h/w))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    m = Basemap(
        projection='tmerc', ellps='WGS84',
        lon_0=np.mean([coordinates[0], coordinates[2]]),
        lat_0=np.mean([coordinates[1], coordinates[3]]),
        llcrnrlon=coordinates[0] - extra * w,
        llcrnrlat=coordinates[1] - (extra * h),
        urcrnrlon=coordinates[2] + extra * w,
        urcrnrlat=coordinates[3] + (extra * h),
        resolution='i',  suppress_ticks=True)
    return m, ax

rs_dict = json_to_dict('../data/ricksteves_articles_blogs_R01.json')
ta_dict = json_to_dict('../data/europe_city_reviews2.json')

empty_count = 0
for k in ta_dict:
    if len(ta_dict[k]) < 100:
        empty_count += 1
# print empty_count

key_list = set(rs_dict.keys() + ta_dict.keys())
europe_dict = dict()
for key in key_list:
    europe_dict[key] = str(rs_dict.get(key)) + str(ta_dict.get(key))

# remove cities which contain little or no text
europe_dict = {key: value for key, value in europe_dict.items() if len(value) > 200}
for k in europe_dict:
    if len(europe_dict[k]) < 100:
        print '\n', k
        print europe_dict[k]

# Convert dictionary into dataframe
cities_df = pd.DataFrame.from_dict(europe_dict, orient='index', dtype=None)
cities_df.columns = ['description']

# Extract cities and their descripitions from dataframe
# doc_bodies = cities_df['description']
doc_bodies = cities_df['description'].values
tokenized_corpus = [word_tokenize(content.lower()) for content in doc_bodies]
cities = cities_df.index.values


with open('personalities.txt') as f:
    for line in f:
        personality = line.strip('\n').split(', ')  # read words on each line of file
        cosine_similarities = calculate_cosine_sim(doc_bodies, ' '.join(personality), [])
        top5 = np.argsort(cosine_similarities)[::-1][:5]
        for num in top5:
            pass


coordinates = [-15.0, 60.0, 32.0, 60.0]

m, ax = plot_initializer(coordinates)

geolocator = Nominatim()
xpts = []
ypts = []
sim = []
for i, city in enumerate(cities):
    location = geolocator.geocode(city, timeout=10)
    if location is not None:
        xpt, ypt = m(location.longitude, location.latitude)
        xpts.append(xpt)
        ypts.append(ypt)
    else:
        print city
print 'got here'
plot_personality_map(m, cosine_similarities, xpts, ypts, cities)
