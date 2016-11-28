from __future__ import division
import pickle
import pandas as pd
import numpy as np
import re
import os.path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def load_dataframes(dataframes):
    '''
    INPUT: list of paths to picked dataframes
    OUTPUT: dataframe(s)
    take paths to pickled dataframes and return dataframes
    '''
    return (pd.read_pickle(df) for df in dataframes)


def load_models(pickle_files):
    '''
    INPUT: list of pickled models
    OUTPUT: unpickled models
    take list of pickled models and return the models
    '''
    models = []
    for p_model in pickle_files:
        with open(p_model) as f:
            models.append(pickle.load(f))
    return tuple(models)


def load_map(pickled_map, coordinates_file):
    '''
    INPUT: pickled basemap object, city coordinates file
    OUTPUT: basemap object, basemap coordinates of cities
    '''
    # load the basemap object for plotting
    with open('../../data/base_map.pkl') as f:
        m = pickle.load(f)

    # load the city coordinates
    with open('../../data/coordinates.txt') as f:
        coords = np.array([[float(coord) for coord in line.strip('\n').split('|')[1:]] for line in f])

    return m, coords


def plot_personality_map(m, cosine_similarities, xpts, ypts, citylist, image_num, image_path):
    '''
    INPUT: basemap, cosine similarites, city coordinates and names, image name and paths
    OUTPUT: map image path
    create map image by cosine similarity
    '''
    fig = plt.figure(figsize=(19, 9))  # initialize figure
    # set markersize, font size and alpha by cosine similarity
    sz = 20. * cosine_similarities / max(cosine_similarities) + 6
    fs = 9. * cosine_similarities / max(cosine_similarities) + 4
    alp = 0.5 * cosine_similarities / max(cosine_similarities)
    cs_sort = np.argsort(cosine_similarities)[::-1]
    alp[cs_sort[:4]] = 1
    fs[cs_sort[:4]] = 13

    top4cities = citylist[cs_sort[:4]]
    top4x = xpts[cs_sort[:4]]
    order = list(top4cities[np.argsort(top4x)])
    alignment = ['left', 'right']
    for i in xrange(len(xpts)):
        plt.plot(xpts[i], ypts[i], '.', markersize=sz[i], color='#81D8D0', alpha=alp[i])
        if citylist[i] in top4cities:
            plt.text(xpts[i], ypts[i], citylist[i], fontsize=int(fs[i]), alpha=alp[i],
                     color=[1, 1, 1], ha=alignment[order.index(citylist[i]) % 2])
        else:
            plt.text(xpts[i], ypts[i], citylist[i], fontsize=int(fs[i]), alpha=alp[i],
                     color=[1, 1, 1], ha=alignment[i % 2])

    # fill in countries and water
    m.drawcoastlines(linewidth=0.2)
    m.drawcountries(linewidth=0.2)
    m.drawmapboundary(fill_color='#2B3856')
    m.fillcontinents(color='#786D5F', lake_color='#2B3856')

    image_filename = image_path.format(image_num)
    plt.savefig(image_filename, bbox_inches='tight')
    plt.close('all')
    return image_filename


def get_map_image(cos_sims, cities, m, coords, user_data):
    '''
    INPUT: cosine similiarites, cities list and coordinates, basemap images, user selections
    OUTPUT: path to image
    '''
    imagenum = ''.join([str(val * 1) for val in user_data.values()])
    if os.path.isfile('static/images/map_images/mapimg_{}.png'.format(imagenum)):
        return 'static/images/map_images/mapimg_{}.png'.format(imagenum)
    else:
        return plot_personality_map(m, np.array(cos_sims[:, 0]), coords[:, 0], coords[:, 1],
                                    cities, imagenum, 'static/images/map_images/mapimg_{}.png')
    return image_file


def get_personality_vector(user_data, tfidf, personality_dict):
    '''
    INPUT: user data, model, personality dictionary
    OUTPUT: personality vector and personality list
    '''
    selections = np.array(user_data.values())
    categories = np.array(user_data.keys())
    selected_categories = categories[selections]

    pers_list = [personality_dict[sc] for sc in selected_categories]
    personality_list = [item for sublist in pers_list for item in sublist]

    personality_vector = tfidf.transform(np.array([' '.join(set(personality_list)), ]))
    return personality_vector, personality_list


def top_city_sentence(top_inds, sent_bodies, personality_list):
    '''
    INPUT: city indices, text, personality list
    OUTPUT: text snippets for top cities
    '''
    sentence = []
    stopwds = ['Rick,', 'Gyugyi', 'killing']
    for i, num in enumerate(top_inds):
        s_array = re.split('[.!?]', sent_bodies[num])
        s_match = 0
        top_sentence = ''
        for s in s_array:
            s = s.replace('read more ', '').replace('More ', '')
            slen = len(set(s.split()) & set(personality_list))
            out_count = len(set(s.split()) & set(stopwds))
            if slen > s_match and out_count == 0:
                s_match = slen
                top_sentence = s
            elif slen >= s_match and len(top_sentence) > 4 and len(s) < len(top_sentence) and \
                    len(s) > 75 and out_count == 0:
                s_match = slen
                top_sentence = s
        tops_sentence = s if len(top_sentence) == 0 else top_sentence
        sentence.append('"{}"'.format(top_sentence.strip()))
    return sentence


def get_city_image(cities):
    '''
    INPUT: cities
    OUTPUT: path to city images
    '''
    return ['static/images/city_images/{}.png'.format(city) for city in cities]
