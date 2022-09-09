from ast import keyword
from statistics import mode
from numpy import pad
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#get the data
print(train_data.shape)
print(train_data.head())

#how much is missing

def missingData():
    missing_test_data = train_data.isnull().sum()
    print(missing_test_data)

#given that a lot of values in the location column are not real places or are either to general like "Milky Way". Plus
#adding on to the fact that anyone at any location can tweet of a disaster from another location in the world, and that a disaster
#can happen anywhere on earth gives to my opinion that the column should be drop

#Data cleaning
train_data = train_data.drop(columns=['location'])
test_data = test_data.drop(columns=['location'])

print("keyword mode value:", str(train_data['keyword'].mode()))
train_data_keyword_mode = train_data['keyword'].mode()
test_data_keyword_mode = test_data['keyword'].mode()
#we fill the na values on keyword with the mode
train_data['keyword'] = train_data['keyword'].fillna(str(train_data_keyword_mode))
test_data['keyword'] = test_data['keyword'].fillna(str(test_data_keyword_mode))

replace_dict = {}
for index, keyword in enumerate(train_data['keyword'].unique()):
    replace_dict[keyword] = index
train_data['keyword'] = train_data['keyword'].replace(replace_dict)

train_features = train_data.copy()
train_labels = train_features.pop('target')

#preprocessing with padding and tokenizer
tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(train_features['text'])
word_index = tokenizer.word_index
train_features['text'] = tokenizer.texts_to_sequences(train_features['text'])
print(train_features['text'][:5])
padded = pad_sequences(train_features['text'],padding='post')
print(padded[:2])
print(padded.shape)

#different data types on columns

