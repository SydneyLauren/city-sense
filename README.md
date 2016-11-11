# City Sense
Identify a city’s mood to provide insight to travelers

### Business Understanding
Before a trip, travelers can find all sorts of information about what to do and where to stay at their destination, but it can be difficult to get a sense of a city’s ambience or feeling. This project will focus on identifying the personalities of European cities to provide insight to potential visitors.

### Data Understanding
Existing datasets are not readily available, but first person anecdotes about European cities can be found on ricksteves.com and tripadvisor.com. I will scrape the two sites for major cities in Europe and combine the text from all of the articles and reviews scraped for a given city.  Some ricksteves articles have already been scraped and are provided in a json file called ‘ricksteves_articles.json’.

### Data Preparation
The text data will be cleaned of punctuation and non-ascii characters in order to process as a corpus using nltk. Exploratory data analysis will be done to view the top features for different cities by term frequency and term frequency - inverse document frequency, over the complete documents and subsets of the documents separated by part of speech.

### Data Pipeline:
Scrape data from ricksteves.com and tripadvisor.com store in mongodb and dump to json
Read json files into dictionary where city names are the keys and text from reviews/articles are the values
Make a tokenized version of the corpus using word tokenize from nltk.tokenize and plot a histogram of the word frequencies to help identify additional stop words
Vectorize using nltk TfidfVectorizer
Explore different subsets of the text by limiting the documents to different parts of speech

### Modeling
The results of the exploratory data analysis will be used to determine which stop words and/or parts of speech will be removed from the dataset. The stop words and number of features will be refined until the model produces a collection of words for each city that identify its personality.  Personality traits may include characteristics like ‘intellectual,’ ‘peaceful,’ ‘down-to-earth’, ‘spiritual,’ ‘friendly’ etc.

### Evaluation
The model will be evaluated by extracting the top features for randomly selected cities in the corpus and checking them for their relevance in describing personality. The evaluation and modeling will be done iteratively.  A potential problem is that I may not arrive at a model which can successfully pull meaningful features that identify a city’s personality.  If this is the case after many iterations, I will need to change the scope of my project to work with the data I have or abandon it altogether.

### Deployment
The project will be deployed as a web app in which the user can input city and find out about the atmosphere there or input conditions (‘relaxing’, ‘vibrant’, ‘friendly’ etc) and get a list of cities that meet their criteria.
