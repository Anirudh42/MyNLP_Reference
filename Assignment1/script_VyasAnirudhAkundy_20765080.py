import sys
import numpy as np
import pandas as pd
import nltk
import random
nltk.download('stopwords')
from nltk.corpus import stopwords
#Obtaining Stop Words from nltk library
stop_words = set(stopwords.words('english'))
#Storing special characters to be removed
special_chars = ['!', '\"', '#', '$', '%', '&', '(', ')', '*', '+', '/', ':', ';',
                 '<', '=', '>', '@', '[', '\\', ']', '^', '`', '{', '|', '}', '~', '\t', '\n']


#WILL TAKE ABOUT 1 MINUTE TO RUN

#Function to tokenize the given corpus
def tokenize_corpus(my_text):
    tokenized_sentence = []
    for i in range(my_text.shape[0]):
        chars = []
        for c in my_text.iloc[i]:
            if c.isalpha():
                chars.append(c.lower())
            else:
                chars.append(c)
        count = 0
        pos = 0
        tokens = []
        for j in range(len(chars)):
            if chars[j] not in ["-", ",", ".", "\'", " ", '!', '\"', '#', '$', '%', '&', '(', ')', '*', '+', '/', ':', ';', '<', '=', '>', '@', '[', '\\', ']', '^', '`', '{', '|', '}', '~', "\n"] and j != len(chars)-1:
                count += 1
            else:
                tokens.append("".join(chars[pos:count]))
                pos = count+1
                count += 1
                if (chars[j].isspace() == False):
                    tokens.append(chars[pos-1])
        filtered_tokens = []
        for k in range(len(tokens)):
            if tokens[k] != "":
                filtered_tokens.append(tokens[k])
        tokenized_sentence.append(filtered_tokens)
    return tokenized_sentence



#Function to remove special characters after tokenization
def remove_specialChars(corpus):
    finalList = []
    for text in corpus:
        temp = []
        current_text = text
        for words in current_text:
            if words not in special_chars:
                temp.append(words)
        finalList.append(temp)
    return finalList



#Function to remove stop words
def remove_stopWords(corpus):
    removed = []
    for text1 in corpus:
        temp1 = []
        curr_text = text1
        for words1 in curr_text:
            if words1 not in stop_words:
                temp1.append(words1)
        removed.append(temp1)
    return removed



#Function to split data into Train(80%), Validation(10%), Test(10%) randomly
def split_data(data):
    train_data = []
    val_data = []
    test_data = []
    datapoints = np.arange(0, len(data))
    train_points = random.sample(set(datapoints), int(0.8*len(data)))
    remaining_pointsForVal = set(np.arange(0, len(data)))-set(train_points)
    validation_points = random.sample(set(remaining_pointsForVal), int(0.1*len(data)))
    remaining_pointsForTest = set(remaining_pointsForVal)-set(validation_points)
    for a in train_points:
        train_data.append(data[a])
    for b in validation_points:
        val_data.append(data[b])
    for c in remaining_pointsForTest:
        test_data.append(data[c])
    return train_data, val_data, test_data







if __name__ == "__main__":
    input_path = sys.argv[1]
    my_data = pd.read_csv(input_path,sep="\n", header=None)
    my_data = my_data[0].apply(str)
    after_tokenizing=tokenize_corpus(my_data)
    specialCharacters_removed=remove_specialChars(after_tokenizing)
    stopWords_removed=remove_stopWords(specialCharacters_removed)
    train_data_noStopWords, validation_data_noStopWords, test_data_noStopWords = split_data(stopWords_removed)
    train_data_withStopWords, validation_data_withStopWords, test_data_withStopWords=split_data(specialCharacters_removed)
    

    np.savetxt("train_no_stopword.csv",train_data_noStopWords, delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", validation_data_noStopWords,delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_data_noStopWords,delimiter=",", fmt='%s')

    np.savetxt("train.csv", train_data_withStopWords,delimiter=",", fmt='%s')
    np.savetxt("val.csv", validation_data_withStopWords,delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_data_withStopWords,delimiter=",", fmt='%s')
