from __future__ import division
from mechanize import Browser
from bs4 import BeautifulSoup
import requests
import mechanize
import urllib
import re
import urllib2
from urllib2 import HTTPError
from PIL import Image
import os


def find_next_text(soup, txt):
    '''
    INPUT: soup, text to search for
    OUPPUT: text appearing after searched text
    take beautifulsoup output and search for text
    '''
    soupstr = str(soup)
    start = soupstr.find(txt) + len(txt)
    end = soupstr[start:].find('"') + start
    return soupstr[start:end]


def get_soup(url):
    '''
    INPUT: url
    OUTPUT beautiful soup element
    take url and return soup
    '''
    data = requests.get(url).content
    soup = BeautifulSoup(data, 'html.parser')
    return soup


def get_image_url(soup):
    '''
    INPUT: BeautifulSoup
    OUTPUT: url of jpg image
    take BeautifulSoup output and return the first jpg image
    '''
    image_url = None
    try1 = soup.find_all('table', attrs={'class': 'infobox geography vcard'})
    try2 = soup.find_all('table', attrs={'class': 'infobox geography'})
    if len(try2) > len(try1):
        use_soup = try2
    else:
        use_soup = try1
    for s in use_soup:
        img = s.find_all('a', {'class': 'image'})
        max_ar = 0

        for i in img:
            page = find_next_text(i, 'src="')
            if '.jpg' in page.lower():
                return page
            elif '.png' in page.lower():
                return page

            # width = int(find_next_text(i, 'width="'))
            # height = int(find_next_text(i, 'height="'))
            # page = find_next_text(i, 'src="')
            # if '.jpg' in page.lower():
            #     ar = width / height
            #     if ar > max_ar:
            #         max_ar = ar
            #         image_url = page
            # elif '.png' in page.lower():
            #     ar = width / height
            #     if ar > max_ar:
            #         max_ar = ar
            #         image_url = page
    return image_url


def save_image(image_url, path, city_key):
    '''
    INPUT: image url, path
    OUTPUT: none
    take image url and save image to user-specified path
    '''
    filename = '{}.png'.format(city_key)
    try:
        urllib.urlretrieve('https:'+image_url, '../app/static/images/city_images/'+filename)
    except TypeError:
        print 'skipping ' + city_key


if __name__ == '__main__':
    cities = []
    keys = []
    with open('European_cities2.txt') as f:
        for line in f:
            splitline = line.split('|')
            city = splitline[0].replace('+', '_')
            city = '_'.join([word.capitalize() for word in city.split('_')])

            if city == 'Jerez_De_La_Frontera':
                 city = 'Jerez_de_la_Frontera'
            key = splitline[1].strip('\n')
            i = 0

            image_url = None
            urls = ['https://en.wikipedia.org/wiki/{}'.format(city),
                    'https://en.wikipedia.org/wiki/{}_(city)'.format(city),
                    'https://en.wikipedia.org/wiki/{},_{}'.format(city, key.split(', ')[-1])
                    ]
            while image_url is None and i < 3:
                url = urls[i]
                soup = get_soup(url)
                image_url = get_image_url(soup)
                i += 1
            save_image(image_url, '../app/static/images/city_images/', key)
