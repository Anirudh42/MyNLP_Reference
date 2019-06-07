import sys
import pandas as pd
import numpy as np
import random
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import OrderedDict
#It will take 5 minutes to execute
#Funtion to create Unigrams and Bigrams
def create_ngrams(my_data,n):
  temp_list=[]
  for i in range(len(my_data)):
      temp_list.append([list(my_data[i][j:j+n]) for j in range(len(my_data[i])-(n-1))])

  for i in range(len(temp_list)):
    for j in range(len(temp_list[i])):
        temp_list[i][j]=" ".join(temp_list[i][j])
  return temp_list


#Function used to label data and create dictionaries to input into the classifier
def label_data(my_text,data1):
  temp=[]
  for i in range(len(my_text)):
      a=OrderedDict()
      for j in range(len(my_text[i])):
          a[my_text[i][j]]=True
      if i<len(data1):
          temp.append((a,1))
      else:
          temp.append((a,0))
      del(a)
  return temp


if __name__ == "__main__":

	#Training data
	pos_data=pd.read_csv(sys.argv[1],sep="\n",names=['text'])
	pos_data['label']=pd.Series([int(1)]*len(pos_data))
	neg_data=pd.read_csv(sys.argv[2],sep="\n",names=['text'])
	neg_data['label']=pd.Series([int(0)]*len(neg_data))
	train_data=pd.concat((pos_data,neg_data),axis=0)
	#Testing data
	pos_testdata=pd.read_csv(sys.argv[5],sep="\n",names=['text'])
	pos_testdata['label']=pd.Series([int(1)]*len(pos_testdata))
	neg_testdata=pd.read_csv(sys.argv[6],sep="\n",names=['text'])
	neg_testdata['label']=pd.Series([int(0)]*len(neg_testdata))
	test_data=pd.concat((pos_testdata,neg_testdata),axis=0)

	#Validation data
	pos_valdata=pd.read_csv(sys.argv[3],sep="\n",names=['text'])
	pos_valdata['label']=pd.Series([int(1)]*len(pos_valdata))
	neg_valdata=pd.read_csv(sys.argv[4],sep="\n",names=['text'])
	neg_valdata['label']=pd.Series([int(0)]*len(neg_valdata))
	val_data=pd.concat((pos_valdata,neg_valdata),axis=0)


	train_data=[eval(train_data.text.iloc[i]) for i in range(len(train_data))]
	val_data=[eval(val_data.text.iloc[i]) for i in range(len(val_data))]
	test_data=[eval(test_data.text.iloc[i]) for i in range(len(test_data))]


	#Creating Unigrams and Bigrams
	train_unigrams = create_ngrams(train_data,1)
	val_unigrams = create_ngrams(val_data,1)
	test_unigrams = create_ngrams(test_data,1)

	train_bigrams = create_ngrams(train_data,2)
	val_bigrams = create_ngrams(val_data,2)
	test_bigrams = create_ngrams(test_data,2)

	#Creating Unigrams+Bigrams
	train_mix = [train_unigrams[i]+train_bigrams[i] for i in range(len(train_data))]
	val_mix = [val_unigrams[i]+val_bigrams[i] for i in range(len(val_data))]
	test_mix = [test_unigrams[i]+test_bigrams[i] for i in range(len(test_data))]


	#Labeling the data
	trainInput_unigrams=label_data(train_unigrams,pos_data)
	trainInput_bigrams=label_data(train_bigrams,pos_data)
	trainInput_mix = label_data(train_mix,pos_data)

	valInput_unigrams=label_data(val_unigrams,pos_valdata)
	valInput_bigrams=label_data(val_bigrams,pos_valdata)
	valInput_mix=label_data(val_mix,pos_valdata)

	testInput_unigrams=label_data(test_unigrams,pos_testdata)
	testInput_bigrams=label_data(test_bigrams,pos_testdata)
	testInput_mix=label_data(test_mix,pos_testdata)


	full_data=[trainInput_unigrams,trainInput_bigrams,trainInput_mix,valInput_unigrams,valInput_bigrams,valInput_mix]

	#Shuffling the data
	for i in full_data:
	  random.shuffle(i)



#This code was used to tune the model by tring different alpha values for each case.
#I have commented this code because it takes a lot of time to execute and
#also because it is not necessary because it is only used for tuning
	# a=np.arange(0.1,1.4,0.1)
	# accuracies=[]
	# for i in range(0,6):
	#   values=[]
	#   for j in a:
	#     mnb= SklearnClassifier(MultinomialNB(alpha=j))
	#     classifier=mnb.train(full_data[i])
	#     values.append(nltk.classify.accuracy(classifier,full_data[i+6]))
	#   accuracies.append(values)


	full_data_test=[testInput_unigrams,testInput_bigrams,testInput_mix]
	test_accuracies=[]
	alpha_values_withStopWords=[0.6,0.4,0.2]
	# alpha_values_withoutStopWords=[0.5,0.6,0.5]
	for i in range(len(alpha_values_withStopWords)):
	  mnb= SklearnClassifier(MultinomialNB(alpha=alpha_values_withStopWords[i]))
	  classifier=mnb.train(full_data[i])
	  test_accuracies.append(100*(nltk.classify.accuracy(classifier,full_data_test[i])))

	print("Accuracy Values:")
	print("Unigrams : " + str(np.round(test_accuracies[0],2)) + "%" )
	print("Bigrams : " + str(np.round(test_accuracies[1],2)) + "%" )
	print("Unigrams + Bigrams : " + str(np.round(test_accuracies[2],2)) + "%" )


