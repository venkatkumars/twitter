import nltk
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

with open("input.csv","r") as f:
    for line in f:
       line12=line.split(",",1)
       sent=line12[0]
       print sent
       text=line12[1]
       print text
       if sent.find("positive")!=-1:
           pos_tweets.append((text[1:-3],sent[1:-1]))
       elif sent.find("negative")!=-1:
           neg_tweets.append((text[1:-3],sent[1:-1]))

print pos_tweets
print neg_tweets

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))

word_features = get_word_features(get_words_in_tweets(tweets))


training_set = nltk.classify.apply_features(extract_features, tweets)


classifier = nltk.NaiveBayesClassifier.train(training_set)

print classifier.show_most_informative_features(32)

tweet=input("Enter Tweet::")
print classifier.classify(extract_features(tweet.split()))
