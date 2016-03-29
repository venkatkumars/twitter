import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def stopNgram(string):
    stopset = set(stopwords.words('english'))
    tokens=word_tokenize(string.decode('utf-8'))
    tokens = [w for w in tokens if not w.lower() in stopset]
    trigrams=ngrams(tokens,3)
    #print type(trigrams), trigrams
    return trigrams
    
def stop(string):
    stopset = set(stopwords.words('english'))
    tokens=word_tokenize(string.decode('utf-8'))
    tokens = [w for w in tokens if not w.lower() in stopset]
    #trigrams=ngrams(tokens,3)
    #print type(trigrams), trigrams
    return ' '.join(tokens)
    
    



pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

neu_tweets = []

with open("input.csv","r") as f:
   for line in f:
       line12=line.split(",",1)
       sent=line12[0]
       print sent
       text=line12[1]
       print text[1:-3]
       print stop(text[1:-3])
       gen=stopNgram(text[1:-3])
       text=[i for i in gen]
       print text
       if sent.find("positive")!=-1:
           pos_tweets.append((text,sent[1:-1]))
       elif sent.find("negative")!=-1:
           neg_tweets.append((text,sent[1:-1]))
       elif sent.find("neutral")!=-1:
           neu_tweets.append((text,sent[1:-1]))

input()

'''
with open('input.csv', 'r') as text_file:
    text = text_file.read()
    tokens=word_tokenize(text.decode('utf-8'))
    tokens = [w for w in tokens if not w in stopset]
    print tokens'''
print pos_tweets
print neg_tweets
print neu_tweets

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets + neu_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))

word_features = get_word_features(get_words_in_tweets(tweets))


training_set = nltk.classify.apply_features(extract_features, tweets)


classifier = nltk.NaiveBayesClassifier.train(training_set)

print classifier.show_most_informative_features(32)

while(1):
  try:
    tweet=input("Enter Tweet::")
    print classifier.classify(extract_features(tweet.split()))
  except StandardError,e:
    print e
