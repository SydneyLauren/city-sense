from selenium import webdriver
import selenium
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from mechanize import Browser
from bs4 import BeautifulSoup
from bson.json_util import dumps
import time
import json
import pymongo
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt

# Define the MongoDB database and table
conn = pymongo.MongoClient()
db = conn.tarvwss
collection = db.tarvws_data
chromedriver = '/Users/sydneydecoto/bin/chromedriver'


def search_city(url, city):
    '''
    INPUT: url as string, city as string
    OUTPUT: html for city page, current url, driver object
    open the url and search for the city
    '''
    # Initialize a chrome driver and go to url
    driver = webdriver.Chrome(chromedriver)
    driver.get(url)

    # insert the city in the search field
    searchbox = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'GEO_SCOPED_SEARCH_INPUT')))
    searchbox.send_keys(city)
    time.sleep(0.1)
    # select the city overview option and click
    mainsearch = driver.find_element_by_xpath('//input[starts-with(@id, "mainSearc")]')
    time.sleep(0.1)
    mainsearch.send_keys('Things to Do')


    try:
        # click the search button to go to next page
        driver.find_elements_by_class_name('inner')[0].click()
    except selenium.common.exceptions.ElementNotVisibleException:
        print 'retry ', city
        return None, None, None
    except selenium.common.exceptions.WebDriverException:
        print 'retry ', city
        return None, None, None

    time.sleep(3)  # sleep so that the page has time to load
    driver.switch_to_alert()  # ignore the popup
    # return the new page html, current url, and driver object
    return driver.page_source, driver.current_url, driver


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


def get_review_locations(current_url):
    '''
    INPUT: url
    OUTPUT: city itinerary text as string
    take a url and return city itinerary
    '''
    br = open_browser(current_url)  # call open_browser to get browser object
    review_list = []
    for i, link in enumerate(br.links()):
        time.sleep(0.1)
        if 'attraction_review' in str(link).lower():
            data = br.follow_link(link)  # click the link
            for item in BeautifulSoup(data, 'html.parser').findAll('div', {'class': 'entry'}):
                review_list.append(item.text)
                if len(review_list) > 10000:
                    return review_list
    return review_list

if __name__ == "__main__":
    cities = []
    citykeys = []
    with open('European_cities2.txt') as f:
        for line in f:
            splitline = line.split('|')
            cities.append(splitline[0].lower())
            citykeys.append(splitline[1].strip('\n'))

    url = 'https://www.tripadvisor.com'

    for key in citykeys:
        html, current_url, driver = search_city(url, key)
        if html is None:
            continue
        review_list = get_review_locations(current_url)
        print '{} has {} review locations'.format(key, len(review_list))

        doc = {key: review_list}
        collection.insert(doc)
        driver.quit()
    with open("europe_city_reviews2.json", "wb") as f:
        f.write(dumps(list(collection.find())))
