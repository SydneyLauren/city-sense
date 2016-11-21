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
import matplotlib.pyplot as plt
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
import unicodedata


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


def json_to_sentencedict(filename):
    '''
    INPUT: name of json file
    OUTPUT: dictionary with city keys and description values
    take json file and return dictionary or original text
    '''
    sentence_dict = dict()
    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            for key in item.keys():
                unique_text = (set(item[key]))
                txt = ''.join(c for c in unique_text)
                txt = ' '.join(ch for ch in txt.split() if ch != '\n')
                txt.replace('\u2019', "'")
                txt = ''.join((c for c in unicodedata.normalize('NFD',
                                                                unicode(txt)) if unicodedata.category(c) != 'Mn'))
                sentence_dict[key.strip().strip('\n').encode('ascii', 'ignore')] = txt.encode('ascii', 'ignore')
    return sentence_dict


def combine_dictionaries(dict1, dict2):
    '''
    INPUT: two dictionaries
    OUTPUT: one dictionary that combines the two dictionaries wherever keys match
    take two dictionaries and merge them
    '''
    key_list = set(dict1.keys() + dict2.keys())
    merged_dict = dict()
    for key in key_list:
        merged_dict[key] = str(dict1.get(key)) + str(dict2.get(key))
    return merged_dict


def generate_dataframe(d, name):
    df = pd.DataFrame.from_dict(d, orient='index', dtype=None)
    df.columns = ['description']
    df.to_pickle(name)
    return df


def get_stopwords(speech_dicts, doc_frequency, word_pos, greater_than=True):
    vals = speech_dicts[word_pos].values()
    wds = np.array(speech_dicts[word_pos].keys())
    doc_freqs = np.array([len(set(val)) for val in vals])
    if greater_than:
        return wds[doc_freqs >= doc_frequency]
    else:
        return wds[doc_freqs <= doc_frequency]


def get_pos_dicts(tokenized_corpus):
    noun_dict = defaultdict(list)
    verb_dict = defaultdict(list)
    adj_dict = defaultdict(list)
    adv_dict = defaultdict(list)

    for i, doc in enumerate(tokenized_corpus):
        for word in doc:
            tag = pos_tag([word])[0][1]
            if 'NN' in tag:
                noun_dict[word].append(i)
            if 'JJ' in tag:
                adj_dict[word].append(i)
            if 'VB' in tag:
                verb_dict[word].append(i)
            if 'RB' in tag:
                adv_dict[word].append(i)
    return noun_dict, verb_dict, adj_dict, adv_dict


def get_stopwords_count(speech_dicts, word_count, word_pos, greater_than=True):
    vals = speech_dicts[word_pos].values()
    wds = np.array(speech_dicts[word_pos].keys())
    word_counts = np.array([len(val) for val in vals])
    if greater_than:
        return wds[word_counts >= word_count]
    else:
        return wds[word_counts <= word_count]


def generate_basemap(cities):
    '''
    INPUT: list of cities
    OUTPUT: pickled basemap object and text file containing basemap coordinates for cities
    generate a basemap object and list of city basemap coordinates for future plotting
    '''
    # m = Basemap(projection='stere', lon_0=5, lat_0=72.0, rsphere=6371200., llcrnrlon=-15.0,
    #             urcrnrlon=74.0, llcrnrlat=32.0, urcrnrlat=55.0, resolution='l')
    m = Basemap(projection='stere', lon_0=5, lat_0=60.0, rsphere=6371200., llcrnrlon=-15.0,
                urcrnrlon=60.0, llcrnrlat=32.0, urcrnrlat=56.0, resolution='l')
    m.drawcoastlines(linewidth=0.2)
    m.drawcountries(linewidth=0.2)
    m.drawmapboundary(fill_color='#2B3856')
    # fill continents, set lake color same as ocean color.
    m.fillcontinents(color='#786D5F', lake_color='#2B3856')
    pickle.dump(m, file('../data/base_map.pkl', 'w'))
    geolocator = Nominatim()

    with open('../data/coordinates.txt', 'w') as f:
        for city in cities:
            location = geolocator.geocode(city, timeout=10)
            xpt, ypt = m(location.longitude, location.latitude)
            f.write('{}|{}|{}\n'.format(city, xpt, ypt))
    plt.plot(xpts, ypts, '.', markersize=13, color='#81D8D0', alpha=0.8)
    plt.savefig('app/static/images/citymap_baseline.png')


if __name__ == '__main__':
    # read json files into dictionary format
    # rs_dict = json_to_dict('../data/ricksteves_articles_blogs_R01.json')
    # ta_dict = json_to_dict('../data/europe_city_reviews2.json')

    rs_dict = json_to_dict('scraping/ricksteves_articles_blogs_R02.json')
    ta_dict = json_to_dict('scraping/ta_combined4.json')

    # rs_dict = json_to_dict('scraping/')
    # combine dictionaries
    europe_dict = combine_dictionaries(rs_dict, ta_dict)

    # get dictionary with complete sentences maintained
    # rs_dict_comp = json_to_sentencedict('../data/ricksteves_articles_blogs_R01.json')
    # ta_dict_comp = json_to_sentencedict('../data/europe_city_reviews2.json')

    rs_dict_comp = json_to_sentencedict('../data/ricksteves_articles_blogs_R02.json')
    ta_dict_comp = json_to_sentencedict('../data/ta_combined4.json')
    # combine dictionaries
    europe_dict_comp = combine_dictionaries(rs_dict_comp, ta_dict_comp)

    # remove cities which contain little or no text
    europe_dict['Belfast, Northern Ireland'] = europe_dict['Belfast, England']
    europe_dict_comp['Belfast, Northern Ireland'] = europe_dict['Belfast, England']
    remove_cities = ['Ostrava, Czech Republic', 'Belfast, England']
    europe_dict = {key: value for key, value in europe_dict.items() if len(value) > 200 and key not in remove_cities}
    europe_dict_comp = {key: value for key, value in europe_dict_comp.items() if key in europe_dict}

    # Convert dictionary into dataframe and save
    cities_df = generate_dataframe(europe_dict, '../data/cities_dataframe.pkl')
    sentence_df = generate_dataframe(europe_dict_comp, '../data/sentence_dataframe.pkl')

    doc_bodies = cities_df['description'].values
    tokenized_corpus = [word_tokenize(content.lower()) for content in doc_bodies]
    # generate list of stop words
    noun_dict, verb_dict, adj_dict, adv_dict = get_pos_dicts(tokenized_corpus)
    dict_array = np.array([noun_dict, verb_dict, adj_dict, adv_dict])
    stp_wds = []
    # nouns
    stp_wds.extend(get_stopwords(dict_array, 181, 0, greater_than=True))
    stp_wds.extend(get_stopwords_count(dict_array, 4485, 0, greater_than=True))
    stp_wds.extend(get_stopwords(dict_array, 1, 0, greater_than=False))
    stp_wds.extend(get_stopwords_count(dict_array, 0, 0, greater_than=False))

    # verbs
    stp_wds.extend(get_stopwords(dict_array, 194, 1, greater_than=True))
    stp_wds.extend(get_stopwords_count(dict_array, 4968, 1, greater_than=True))
    stp_wds.extend(get_stopwords(dict_array, 0, 1, greater_than=False))
    stp_wds.extend(get_stopwords_count(dict_array, 0, 1, greater_than=False))

    # adjectives
    stp_wds.extend(get_stopwords(dict_array, 155, 2, greater_than=True))
    stp_wds.extend(get_stopwords_count(dict_array, 929, 2, greater_than=True))
    stp_wds.extend(get_stopwords(dict_array, 0, 2, greater_than=False))
    stp_wds.extend(get_stopwords_count(dict_array, 0, 2, greater_than=False))

    # adverbs
    stp_wds.extend(get_stopwords(dict_array, 192, 3, greater_than=True))
    stp_wds.extend(get_stopwords_count(dict_array, 2258, 3, greater_than=True))
    stp_wds.extend(get_stopwords(dict_array, 0, 3, greater_than=False))
    stp_wds.extend(get_stopwords_count(dict_array, 0, 3, greater_than=False))

    # Two letter words
    for lst in tokenized_corpus:
        stp_wds.extend([word for word in lst if len(word) < 3])
    stp_wds = list(set(stp_wds))
    # build and save vectorizer
    # stops = []
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english') + stp_wds)
    sentence_vectorizer = TfidfVectorizer()
    with open('../data/my_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('../data/sentence_vectorizer.pkl', 'wb') as f:
        pickle.dump(sentence_vectorizer, f)

    # generate basemap object and city coordinates for plotting
    generate_basemap(cities_df.index.values)
