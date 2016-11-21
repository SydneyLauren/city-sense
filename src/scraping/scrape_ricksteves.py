from mechanize import Browser
from bs4 import BeautifulSoup
import requests
import mechanize
from urllib2 import HTTPError
from bson.json_util import dumps
import json
import pymongo


def open_browser(url):
    '''
    INPUT: string containing url to open
    OUTPUT: browser object
    open the requested page and return a browser object
    '''
    br = Browser()  # Initialize browser object
    br.set_handle_robots(False)
    br.addheaders = [('User-agent', 'Firefox')]
    br.open(url)  # Retrieve the requested page
    br.select_form(nr=0)
    return br


def get_article_text(city):
    url = 'https://search.ricksteves.com/?button=&filter=Tips+%26+Articles&query={}&utf8=%E2%9C%93'.format(city)
    response = requests.get(url)
    br = open_browser(url)  # call open_browser to get browser object
    description = []
    for link in br.links():
        linkstr = link.url.lower()
        if ('/articles/' in linkstr or '/watch-read-listen/' in linkstr) and city.lower() in linkstr:
            try:
                data = br.follow_link(link)  # click the link
            except HTTPError:
                continue
            for item in BeautifulSoup(data, 'html.parser').findAll('div', {'class': 'wysiwyg'}):
                for paragraph in item.findAll('p'):
                    description.append(paragraph.text)
    return description


def get_blog_text(city):
    url = 'https://search.ricksteves.com/?button=&filter=Rick%27s+Blog&query={}&utf8=%E2%9C%93'.format(city)
    response = requests.get(url)
    br = open_browser(url)  # call open_browser to get browser object
    description = []
    for link in br.links():
        linkstr = link.url.lower()
        if '/blog/' in linkstr and city.lower() in linkstr:
            try:
                data = br.follow_link(link)  # click the link
            except HTTPError:
                continue
            for item in BeautifulSoup(data, 'html.parser').findAll('div', {'id': 'content'}):
                for paragraph in item.findAll('p'):
                    description.append(paragraph.text)
    return description


def get_destination_text(city):
    url = 'https://search.ricksteves.com/?button=&filter=Destinations&query={}&utf8=%E2%9C%93'.format(city)
    response = requests.get(url)
    br = open_browser(url)  # call open_browser to get browser object
    description = []
    for link in br.links():
        linkstr = link.url.lower()
        if '/europe/' in linkstr and city.lower() in linkstr:
            try:
                data = br.follow_link(link)  # click the link
            except HTTPError:
                continue
            for item in BeautifulSoup(data, 'html.parser').findAll('div', {'class': 'wysiwyg'}):
                for paragraph in item.findAll('p'):
                    description.append(paragraph.text)
    return description
# Define the MongoDB database and table
conn = pymongo.MongoClient()
db = conn.rsscc2
collection = db.rss_datacc2

cities = []
keys = []
with open('European_cities2.txt') as f:
    for line in f:
        splitline = line.split('|')
        cities.append(splitline[0].lower())
        keys.append(splitline[1].strip('\n'))

city_count = 0
for key, city in zip(keys, cities):
    print key, city
    description = get_article_text(city)
    blog = get_blog_text(city)
    destination = get_destination_text(city)
    doc = {key: description + blog + destination}
    print '{} articles, {} blogs, {} destination'.format(len(description), len(blog), len(destination))
    collection.insert(doc)

with open("ricksteves_articles_blogs_R03.json", "wb") as f:
    f.write(dumps(list(collection.find())))
