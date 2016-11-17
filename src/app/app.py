from __future__ import division
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)

# load the dataframe that was built in build_model.py
cities_df = pd.read_pickle('../../data/cities_dataframe.pkl')

# load the vectorized you pickled in build_model.py
with open('../../data/my_vectorizer.pkl') as f:
    tfidf = pickle.load(f)

# extract cities and their descripitions from dataframe
doc_bodies = cities_df['description'].values
cities = cities_df.index.values
X = tfidf.fit_transform(doc_bodies)

# read personality words from text file
personality_dict = dict()


def plot_personality_map(m, cosine_similarities, xpts, ypts, citylist):
    fig = plt.figure(figsize=(19, 9))
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

    plt.savefig('city_calculated.jpg', bbox_inches='tight')


with open('../../data/personalities_R01.txt') as f:
    for line in f:
        kv_split = line.split(': ')
        personality_dict[kv_split[0]] = kv_split[1].strip('\n').split(', ')


@app.route('/', methods=['GET', 'POST'])
def index():
    pagetitle = {'name': "CitySense"}
    intro = {'txt': 'Some intro text will go here'}
    return render_template('index.html',
                           intro=intro,
                           pagetitle=pagetitle)


@app.route('/solve', methods=['GET', 'POST'])
def solve():
    user_data = request.json
    selections = np.array(user_data.values())
    categories = np.array(user_data.keys())
    selected_categories = categories[selections==True]

    pers_list = [personality_dict[sc] for sc in selected_categories]
    personality_list = [item for sublist in pers_list for item in sublist]

    personality_vector = tfidf.transform(np.array([' '.join(personality_list), ]))
    cos_sims = cosine_similarity(X, personality_vector)
    top4 = np.argsort(cos_sims[:, 0])[::-1][:4]
    top_cities = [cities[n] for n in top4]

    return jsonify({'city_1': top_cities[0], 'city_2': top_cities[1],
                    'city_3': top_cities[2], 'city_4': top_cities[3]})


if __name__ == '__main__':
    app.run(debug=True)
