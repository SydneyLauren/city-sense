from __future__ import division
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import itertools
import cStringIO
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt



IMAGE_PATH = 'final_model/mapimage_{}.png'
# load the dataframe that was built in build_model.py
cities_df = pd.read_pickle('../data/cities_dataframe.pkl')
sentence_df = pd.read_pickle('../data/sentence_dataframe.pkl')

# load the vectorized you pickled in build_model.py
with open('../data/my_vectorizer.pkl') as f:
    tfidf = pickle.load(f)
with open('../data/sentence_vectorizer.pkl') as f:
    tfidf_sentence = pickle.load(f)

# extract cities and their descripitions from dataframe
doc_bodies = cities_df['description'].values
sent_bodies = sentence_df['description'].values

cities = cities_df.index.values
sent_cities = sentence_df.index.values

X = tfidf.fit_transform(doc_bodies)

# read personality words from text file
personality_dict = dict()

# load the basemap object for plotting
with open('../data/base_map.pkl') as f:
    m = pickle.load(f)

# load the city coordinates
with open('../data/coordinates.txt') as f:
    coords = np.array([[float(coord) for coord in line.strip('\n').split('|')[1:]] for line in f])


def plot_personality_map(m, cosine_similarities, xpts, ypts, citylist, image_num):

    fig = plt.figure(figsize=(19, 9))
    cosine_similarities = np.array(cosine_similarities)
    sz = 20. * cosine_similarities / max(cosine_similarities) + 6
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
        plt.plot(xpts[i], ypts[i], '.', markersize=sz[i], color='#81D8D0', alpha=alp[i])
        plt.text(xpts[i], ypts[i], citylist[i], fontsize=int(fs[i]), alpha=alp[i], color=[1, 1, 1])

    m.drawcoastlines(linewidth=0.2)
    m.drawcountries(linewidth=0.2)
    m.drawmapboundary(fill_color='#2B3856')
    # fill continents, set lake color same as ocean color.
    m.fillcontinents(color='#786D5F', lake_color='#2B3856')
    image_filename = IMAGE_PATH.format(image_num)
    plt.savefig(image_filename, bbox_inches='tight')

    return image_filename


with open('../data/personalities_R03.txt') as f:
    for line in f:
        kv_split = line.split(': ')
        personality_dict[kv_split[0]] = kv_split[1].strip('\n').split(', ')

categories = ['relaxing', 'entertaining', 'intellectual', 'spiritual', 'uplifting',
              'friendly', 'profound', 'creative', 'surreal', 'accepting', 'funny',
              'exciting', 'lively', 'natural', 'sophisticated']

count = 0
for i in xrange(2):
    for subset in itertools.combinations(categories, i+1):
        imagenum = ''.join([str((cat in subset) * 1) for cat in categories])
        pers_list = [personality_dict[sc] for sc in subset]
        personality_list = [item for sublist in pers_list for item in sublist]
        personality_vector = tfidf.transform(np.array([' '.join(set(personality_list)), ]))
        cos_sims = cosine_similarity(X, personality_vector)
        image_filename = plot_personality_map(m, cos_sims[:, 0], coords[:, 0], coords[:, 1], cities, imagenum)
        count += 1

asdads

image_filename = plot_personality_map(m, cos_sims[:, 0], coords[:, 0], coords[:, 1], cities, imagenum)
