import pandas as pd
import numpy as np
import json
import nltk
import string
from sklearn.metrics import confusion_matrix

# Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

stop_words = stopwords.words('english')
snowball_stemmer = SnowballStemmer('english')



class Processor():

    def __init__(self):
        self.sentiment_words = None 

    
    def pre_processing(self, sentiment, review):
        words = review.split()
        stemmed_words = []
        stemmed_sample = {'Sentiment': sentiment,
                          'Word Table': {} }
    
        # Loop through all words in sample 
        for word in words:

            # Remove cappital sensitivity 
            word = word.lower()

            # Removes all punctuation 
            word = ''.join(char for char in word if char not in string.punctuation)

            # Remove numbers and filter out stop words 
            if word not in stop_words:
                try:
                    int(word)
                    continue
                except:
                    pass
        
                # Applies the Snoball Stemmer from the NLTK package to the list of words in the sample 
                stemmed_word = snowball_stemmer.stem(word)
                stemmed_words.append(stemmed_word)
    
        # Create word count table of filtered stemmed words 
        for stemmed_word in stemmed_words:
            try:
                stemmed_sample['Word Table'][stemmed_word] = stemmed_sample['Word Table'][stemmed_word] + 1 
            except:
                stemmed_sample['Word Table'][stemmed_word] = 0

        return stemmed_sample



    
    def word_processing(self, training_data, testing):
        processed_samples = []
        all_words = None
    
        # Run each sample though pre-processing to filter and stem the words. Table returned
        for index, row in training_data.iterrows():
            sentiment = row['Sentiment']
            review = row['Review']
            sample = self.pre_processing(sentiment, review)
            processed_samples.append(sample)  

        # Itterate through each processed sample and add new words the all_words list 
        for idx, sample in enumerate(processed_samples):
            words = list(sample['Word Table'].keys())
            
            if idx == 0:
                all_words = words
            else:
                for word in words:
                    if word not in all_words:
                        all_words.append(word)

        return all_words, processed_samples



    
    def sentiment_filter(self, all_words):
        
        # Initialise the sentiment intensity analyser
        sia = SentimentIntensityAnalyzer()
        sentiment_words = []
        
        for word in all_words:
            # Find the sentiment intensity score of the word
            score = sia.polarity_scores(word)['compound']
            
            # Filter words based on sentiment being over or under 0. 0 being a neutral word
            if score != 0:
                sentiment_words.append(word)

        return sentiment_words

    
    
    
    def process_binary_data(self, processed_samples):
        all_data = []
        sample_data = []
        sample_count = 0

        # Convert word data to binary lists. First element is the label, the rest are the features 
        for sample in processed_samples:
            sentiment = sample['Sentiment']
            sample_data.append(sentiment)
            
            for word in self.sentiment_words:
                if word in list(sample['Word Table'].keys()):
                    sample_data.append(1)
                else:
                    sample_data.append(0)
                    
            all_data.append(sample_data) 
            sample_data = []
    
        return np.array(all_data)


    
    
    def process(self, raw_data, testing=False):
        # Controller function that passes the raw text data to pre-processing and filters the word table for non-neutral words and outputs a binary data table for learning
        
        all_words, processed_samples = self.word_processing(raw_data, testing)

        if testing == False:
            self.sentiment_words = self.sentiment_filter(all_words)
            
        binary_data = self.process_binary_data(processed_samples)
        
        return binary_data

    