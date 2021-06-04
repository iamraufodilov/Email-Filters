# import libraries

#Load-data Libraries
import pandas as pd

#Text Processing libraries
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


#Model libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#Save the model
import joblib
from joblib import dump, load

#Evaluate the model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#load dataset
data = pd.read_csv('G:/rauf/STEPBYSTEP/Data/spambase_ds/Email spam.csv')
#_>print(data.head(20))
#Spam Example
#_>print(data['text'][1])
df = data[data['spam'] ==0]
#Non-Spam example
#_>print(df['text'][1369])
#Non-Spam example
data['spam'].unique()


##Check if we have missing values
#_>print(data.isnull().sum())

#Check if we have duplicates values
#_>print(data.duplicated().sum()) # here we can see we have 33 duplicated emails lets remove them

# drop duplicates
data.drop_duplicates(inplace=True)

# Save a CSV file 
data.to_csv('G:/rauf/STEPBYSTEP/Projects/NLP/Email Filters/Naive Bayes Model/cleaned_data.csv', index=False)


# retrieve data
data = pd.read_csv('G:/rauf/STEPBYSTEP/Projects/NLP/Email Filters/Naive Bayes Model/cleaned_data.csv')

# change the name of columns
data.rename(columns = {'spam': 'Label', 'text':'Email'}, inplace=True)


# text preprocessing

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


#Write a tokenization function to process/ clean text data (email)
def tokenize(text):

   #1. Normalize: Convert to lower case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
   #2. Tokenizing: split text into words
    tokens = word_tokenize(text)
    
   #3. Remove stop words: if a token is a stop word, then remove it
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    #4. Lemmatize and Stemming
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    clean_tokens = []
    
    for i in lemmed_words:
        clean_tokens.append(i)
        
        ## back to string from list
    text = " ".join(clean_tokens)
    return text

    #return clean_tokens


# split data for train and test
# assign the independent features (text) to X variable and the target to y
X = data['Email']
y = data['Label']

#split data into training 77% and test 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# create train pipeline
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])

#train pipeline
pipeline.fit(X_train, y_train)


# test the pipeline
predicted = pipeline.predict(X_test)


# evaluate the model
accuracy = accuracy_score(y_test, predicted)

print("Accuracy:",  round(accuracy,2))

print("Other Metrics:")
print(classification_report(y_test, predicted))


#
from sklearn.metrics import confusion_matrix
## Plot confusion matrix
print(confusion_matrix(y_test, predicted))


# save the model
import pickle
file_name = 'pipelinemodel'
pickle.dump(pipeline, open(file_name, 'wb'))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
in this project we load dataset
we cleaned it and save to local dir
then we load cleaned data, and we tokenize, normalize lemmatize, remove stopwords
then split data to tran and test
create naive bayes classifier model pipeline and train it
then we evaluate test data with 74% accuracy
finally we save model as binary file
'''