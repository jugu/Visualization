import os
import ast
import nltk
#from nltk.stem import *
import string
#from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from nltk.stem.porter import PorterStemmer
from sklearn.utils import extmath
import pandas as pd


rootdir = ''
movielinesfile = rootdir + "movie_lines_sm.txt"
movietitlesfile = rootdir + "movie_titles_metadata.txt"
'''
Which words appear in the dialoges of movies beloging to a particular genre
'''

def preprocess_prepareMovieGenres():
    movieToGenre = dict()
    genreToMovieTitles = dict()
    lines = [line.rstrip('\n') for line in open(movietitlesfile)]
    for line in lines:
        arr = line.split("+++$+++")
        genres = ast.literal_eval((arr[len(arr) - 1]).strip())
        movieToGenre[arr[0].strip()] = genres
        for genre in genres:
            if not genre in genreToMovieTitles.keys():
                genreToMovieTitles[genre] = list()
            genreToMovieTitles[genre].append(arr[0].strip())
    return movieToGenre, genreToMovieTitles

def preprocess_extractMovieLines():
   movieToDialogs = dict()
   lines = [line.rstrip('\n') for line in open(movielinesfile)]
   stemmer = PorterStemmer()
   printable = set(string.printable)
   for line in lines:
        arr = line.split("+++$+++")
        title = arr[2].strip()
        line =  arr[len(arr)-1].strip()
        try:
            line = line.encode('utf-8',errors='ignore')
        except:
            line = filter(lambda x: x in printable, line)
        line = stemmer.stem(line)
        if not title in movieToDialogs.keys():
            movieToDialogs[title] = line
        movieToDialogs[title] = " ".join([movieToDialogs[title],line])
   return movieToDialogs

def tf_idf_movielines(movieToDialogs):
   #movieToDialogs = preprocess_extractMovieLines()
   #stops = set(stopwords.words("english"))
   movieToWords = dict()
   remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
   for movie,lines in movieToDialogs.iteritems():
        lines = lines.lower()
        no_punctuation = lines.translate(remove_punctuation_map)
        movieToWords[movie] = no_punctuation
        #filtered_words = [word.lower() for word in line.split(" ") if word not in stops]
   tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
   tfs = tfidf.fit_transform(movieToWords.values())
   print tfidf
   return tfs

def tf_idf_genrelines():
   movieToGenre, genreToMovieTitles = preprocess_prepareMovieGenres()
   movieToDialogs = preprocess_extractMovieLines()
   #tfsmovielines = tf_idf_movielines(movieToDialogs)
   #stops = set(stopwords.words("english"))
   movieToWords = dict()
   genreToWords = dict()
   remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
   genreList = list()
   genreWordsList = list()
   for genre,movies in genreToMovieTitles.iteritems():
        alllines = ""
        for movie in movies:
            if (not movieToDialogs.has_key(movie)):
                continue
            lines = movieToDialogs[movie]
            lines = lines.lower()
            no_punctuation = lines.translate(remove_punctuation_map)
            movieToWords[movie] = no_punctuation
            alllines = " ".join([alllines, no_punctuation])
        if (len(alllines) > 0):
            genreToWords[genre] = alllines
            genreList = genreList + [genre]
            genreWordsList = genreWordsList + [alllines]
   return genreWordsList, genreList


def tf_idf_test():
    titles = ["The Neatest Little Guide to Stock Market Investing",
          "Investing For Dummies, 4th Edition",
          "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
          "The Little Book of Value Investing",
          "Value Investing: From Graham to Buffett and Beyond",
          "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
          "Investing in Real Estate, 5th Edition",
          "Stock Investing For Dummies",
          "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
          ]
    exclude = set(string.punctuation)
    processed = []
    for title in titles:
        title = title.lower()
        title = ''.join(ch for ch in title if ch not in exclude)
        processed = processed + [title]
    labels = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]
    return processed, labels

def datacoordinates(datavaluelist, datalabellist):
        #filtered_words = [word.lower() for word in line.split(" ") if word not in stops]
   stopwords = nltk.corpus.stopwords.words('english')
   tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=stopwords, max_features=100)
   #tfs_movieline = tfidf.fit_transform(movieToWords.values())
   tfs_genreline = tfidf.fit_transform(datavaluelist)
   U, S, V = extmath.randomized_svd(tfs_genreline, n_components=3)
   coords = []
   count = 0
   for xy in U:
       row = {}
       row['yvalue'] = xy[1]
       row['xvalue'] = xy[2]
       row['pointname'] = datalabellist[count]
       row['cluster'] = 'genre'
       count += 1
       coords.append(row)
   count = 0
   for x in V[1]:
       row = {}
       row['yvalue'] = x
       row['xvalue'] = V[2][count]
       row['pointname'] = tfidf.get_feature_names()[count]
       row['cluster'] = 'movie words'
       count += 1
       coords.append(row)
   #lsa = TruncatedSVD(2, algorithm = 'randomized')
   #dtm_lsa_genreline = lsa.fit_transform(tfs_genreline)
   #dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa_genreline)
   #df = pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns =tfidf.get_feature_names())
   return coords


stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        item = stemmer.stem(item)
        if len(item) > 3:
            stemmed.append(item)
    return stemmed

def tokenize(text):
    import re
    token_pattern = r"(?u)\b\w\w\w+\b"
    token_pattern = re.compile(token_pattern)
    text = " ".join(token_pattern.findall(text))
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#tf_idf_genrelines()
#stemmer = PorterStemmer()
#print stemmer.stem("Headquarters!? What is it?")