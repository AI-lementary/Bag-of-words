import pandas as pd
#Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
#Download the punkt tokenizer for sentence splitting
import nltk.data
#nltk.download()


#Read data from files
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
#verify the number of reviews that were read
print ("Read %d labeled train reviews, %d labeled test reviews," \
       "and %d unlabeled reviews\n" % (train['review'].size, test['review'].size,unlabeled_train['review'].size))



def review_to_wordlist(review, remove_stopwords=False):
    #Function to convert a dcoument to a sequence of words,
    #optionally removing stopwords
    #
    #1. remove html
    review_text = BeautifulSoup(review,"lxml").get_text()
    
    #2. remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    
    #3. Convert words to lower case and split them
    words = review_text.lower().split()
    
    #4. optionally remove stopwords (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #5. Return a list of words
    return(words)

#print(train.loc[0]['review'])
#print(review_to_wordlist(train.loc[0]['review']))

#Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')        

#Define a function to split a review into parsed sentences
def review_to_sentences(review,tokenizer,remove_stopwords=False):
    ## Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #This is the format that word2vec requires - a list (sentences) of lists (words)
    
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    
    # 2. Loop over each sentence
    sentences =[]
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence)>0:
            #Call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    # Return the list of sentences - each sentence is a list of words, so this returns a list of lists        
    return sentences

#Now we can apply this function to prepare our data for input to Word2Vec (this will take a couple minutes):
sentences = []


print("Parsing sentences from Trained set")
for review in range(train["review"].size):
    #print("This is review ", review)
    if review%100==0:
        print(" %d percent progress so far" % (review/train["review"].size*100))
    sentences += review_to_sentences(train["review"][review], tokenizer)

print("Parsing sentences from Unlabeled set")
for review in range(unlabeled_train["review"].size):
    if review%100==0:
        print(" %d percent progress so far" % (review/unlabeled_train["review"].size*100))
    sentences += review_to_sentences(unlabeled_train["review"][review], tokenizer)

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    