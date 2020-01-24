import pandas as pd
import numpy as np
import io
import random
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, Activation, Flatten
from keras import regularizers


train_pos=pd.read_csv('./train_positive_no_stopword.csv',sep="\n",names=['sent'])
train_neg=pd.read_csv('./train_negative_no_stopword.csv',sep="\n",names=['sent'])
val_pos=pd.read_csv('./val_positive_no_stopword.csv',sep="\n",names=['sent'])
val_neg=pd.read_csv('./val_negative_no_stopword.csv',sep="\n",names=['sent'])
test_pos=pd.read_csv('./test_positive_no_stopword.csv',sep="\n",names=['sent'])
test_neg=pd.read_csv('./test_negative_no_stopword.csv',sep="\n",names=['sent'])

embeddings = Word2Vec.load('./assignment3word2vec.model')

token=Tokenizer(num_words=27000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

train_pos['label']=len(train_pos)*[1]
train_neg['label']=len(train_neg)*[0]
val_pos['label']=len(val_pos)*[1]
val_neg['label']=len(val_neg)*[0]
test_pos['label']=len(test_pos)*[1]
test_neg['label']=len(test_neg)*[0]

train_data=pd.concat((train_pos,train_neg),axis=0)
train_data=train_data.sample(frac=1)
val_data=pd.concat((val_pos,val_neg),axis=0)
val_data=val_data.sample(frac=1)
test_data=pd.concat((test_pos,test_neg),axis=0)
test_data=test_data.sample(frac=1)

X_train=train_data['sent']
y_train=train_data['label']
X_val=val_data['sent']
y_val=val_data['label']
X_test=test_data['sent']
y_test=test_data['label']

full_data=pd.concat((X_train,X_val,X_test),axis=0)

def stringtolist(data):
  data=[eval(data.iloc[i]) for i in range(len(data))]
  return data

X_train=stringtolist(X_train)
X_val=stringtolist(X_val)
X_test=stringtolist(X_test)
full_data=stringtolist(full_data)

sentlength = int(np.percentile([len(seq) for seq in full_data], 95))

token.fit_on_texts([' '.join(seq[:sentlength]) for seq in full_data])
X_train = token.texts_to_sequences([' '.join(seq[:sentlength]) for seq in X_train])
X_val = token.texts_to_sequences([' '.join(seq[:sentlength]) for seq in X_val])
X_test = token.texts_to_sequences([' '.join(seq[:sentlength]) for seq in X_test])

X_train = pad_sequences(X_train, maxlen=sentlength, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=sentlength, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=sentlength, padding='post', truncating='post')

EMB_DIM=embeddings.vector_size
VOCAB_SIZE=len(token.word_index)+1

embedding_matrix=np.random.randn(VOCAB_SIZE,EMB_DIM)

for word,i in token.word_index.items():
  if word in embeddings.wv.vocab:
    embedding_matrix[i]=embeddings[word]
  else:
    embedding_matrix[i]=np.random.randn(1,EMB_DIM)



y_train=np_utils.to_categorical(y_train)
y_val=np_utils.to_categorical(y_val)
y_test=np_utils.to_categorical(y_test)


classifier=Sequential()
classifier.add(Embedding(input_dim=VOCAB_SIZE,output_dim=EMB_DIM,weights=[embedding_matrix], input_length=sentlength,
                         trainable=False,name='Embedding_Layer'))
classifier.add(Flatten())
classifier.add(Dense(150,activation='sigmoid',kernel_regularizer=regularizers.l2(0.005),name='Hidden_Layer'))#With L2
#classifier.add(Dense(150,activation='tanh',name='Hidden_Layer'))#Without L2
classifier.add(Dropout(rate=0.9,name='Dropout'))
classifier.add(Dense(2,activation='softmax',name='Output_Layer'))
classifier.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

classifier.fit(X_train, y_train,
                  batch_size=1024,
                  epochs=10,
                  validation_data=(X_val, y_val))
print("Test Accuracy : " + str(classifier.evaluate(X_test,y_test)[1]*100))
