from __future__ import division
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify, make_response
import numpy as np
import setup_app
import numpy as np
app = Flask(__name__)


# Load the dataframes containing corpus
cities_df, sentence_df = setup_app.load_dataframes(['../../data/cities_dataframe.pkl',
                                                    '../../data/sentence_dataframe.pkl'])
# extract cities and their descripitions from dataframe
doc_bodies = cities_df['description'].values
sent_bodies = sentence_df['description'].values
cities = cities_df.index.values
sent_cities = sentence_df.index.values

# load the vectorized models pickled in build_model.py
tfidf, tfidf_sentence = setup_app.load_models(['../../data/my_vectorizer.pkl',
                                               '../../data/sentence_vectorizer.pkl'])
# fit the model
X = tfidf.fit_transform(doc_bodies)

# read personality words from text file
personality_dict = dict()

# load the basemap object and city coordinates for plotting
m, coords = setup_app.load_map('../../data/base_map.pkl',
                               '../../data/coordinates.txt')


with open('../../data/personalities_R05.txt') as f:
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


@app.route('/home', methods=['GET', 'POST'])
def home():
    return index()


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('threecolumn.html')


@app.route('/solve', methods=['GET', 'POST'])
def solve():
    user_data = request.json  # record user-selected personality trait(s)

    # get list of personality words and vectorized personality
    personality_vector, personality_list = setup_app.get_personality_vector(user_data, tfidf, personality_dict)

    # calculate cosine similarity and get top four cities
    cosine_similarities = cosine_similarity(X, personality_vector)
    top4 = np.argsort(cosine_similarities[:, 0])[::-1][:4]
    top_cities = [cities[n] for n in top4]

    # get representative text snippet for top cities
    sentence = setup_app.top_city_sentence(top4, sent_bodies, personality_list)
    # get map plot
    image_filename = setup_app.get_map_image(cosine_similarities, cities, m, coords, user_data)
    # get city image paths
    image_paths = setup_app.get_city_image(top_cities)

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
