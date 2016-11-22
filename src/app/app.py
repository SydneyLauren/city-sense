from __future__ import division
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify, make_response
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import cStringIO
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
app = Flask(__name__)


IMAGE_PATH = 'static/images/mapimage_{}.png'
# load the dataframe that was built in build_model.py
cities_df = pd.read_pickle('../../data/cities_dataframe.pkl')
sentence_df = pd.read_pickle('../../data/sentence_dataframe.pkl')

# load the vectorized you pickled in build_model.py
with open('../../data/my_vectorizer.pkl') as f:
    tfidf = pickle.load(f)
with open('../../data/sentence_vectorizer.pkl') as f:
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
with open('../../data/base_map.pkl') as f:
    m = pickle.load(f)

# load the city coordinates
with open('../../data/coordinates.txt') as f:
    coords = np.array([[float(coord) for coord in line.strip('\n').split('|')[1:]] for line in f])


def plot_personality_map(m, cosine_similarities, xpts, ypts, citylist, image_num):
    print 'calculating sizing parameters'
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
    print 'plotting the points'
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
    print 'plot saved'
    return image_filename


def get_city_image(cities):
    return ['static/images/city_images_3/{}.png'.format(city) for city in cities]


with open('../../data/personalities_R03.txt') as f:
    for line in f:
        kv_split = line.split(': ')
        personality_dict[kv_split[0]] = kv_split[1].strip('\n').split(', ')


@app.route('/', methods=['GET', 'POST'])
def index():
    pagetitle = {'name': "CitySense"}
    intro = {'txt': 'A destination for every personality'}
    return render_template('index.html',
                           intro=intro,
                           pagetitle=pagetitle)


@app.route('/solve', methods=['GET', 'POST'])
def solve():
    user_data = request.json
    selections = np.array(user_data.values())
    categories = np.array(user_data.keys())
    selected_categories = categories[selections]
    imagenum = 0
    for i, key in enumerate(sorted(user_data.keys())):
        imagenum += (i + 1) * user_data[key]

    pers_list = [personality_dict[sc] for sc in selected_categories]
    personality_list = [item for sublist in pers_list for item in sublist]

    personality_vector = tfidf.transform(np.array([' '.join(set(personality_list)), ]))
    cos_sims = cosine_similarity(X, personality_vector)
    top4 = np.argsort(cos_sims[:, 0])[::-1][:4]
    top_cities = [cities[n] for n in top4]

    image_paths = get_city_image(top_cities)
    sentence = []
    for i, num in enumerate(top4):
        s_array = re.split('[.!]', sent_bodies[num])
        # r'(?<=\w)\.(?!\..)|!'
        # s_array = sent_bodies[num].split('.')

        s_match = 0
        top_sentence = ''
        for s in s_array:
            s = s.lstrip('read more ')
            s = s.lstrip('More ')
            slen = len(set(s.split()) & set(personality_list))
            if slen > s_match:
                s_match = slen
                top_sentence = s
        sentence.append('"{}"'.format(top_sentence.strip()))
    print 'trying to plot'

    image_filename = plot_personality_map(m, cos_sims[:, 0], coords[:, 0], coords[:, 1], cities, imagenum)
    print image_filename
    return jsonify({'city_1': top_cities[0], 'city_2': top_cities[1],
                    'city_3': top_cities[2], 'city_4': top_cities[3],
                    'c1text': sentence[0], 'c2text': sentence[1],
                    'c3text': sentence[2], 'c4text': sentence[3],
                    'map_image': image_filename,
                    'city1pic': image_paths[0], 'city2pic': image_paths[1],
                    'city3pic': image_paths[2], 'city4pic': image_paths[3],
                    })


if __name__ == '__main__':
    app.run(debug=True)
