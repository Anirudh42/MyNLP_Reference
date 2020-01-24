import pandas as pd
import numpy as np
from gensim.models import Word2Vec


#Function to print most similar words to any given word within the vocabulary
def similarWordTo(word, n):
  for i in range(n):
    print(w2v_model.wv.most_similar(positive=[word], topn=n)[i][0])



if __name__ == "__main__":
    
  #Concatenating all the data
  pos_data_nosw=pd.read_csv("train_positive_no_stopword.csv",sep="\n",names=['text'])
  neg_data_nosw=pd.read_csv("train_negative_no_stopword.csv",sep="\n",names=['text'])
  pos_valdata_nosw=pd.read_csv("val_positive_no_stopword.csv",sep="\n",names=['text'])
  neg_valdata_nosw=pd.read_csv("val_negative_no_stopword.csv",sep="\n",names=['text'])
  pos_testdata_nosw=pd.read_csv("test_positive_no_stopword.csv",sep="\n",names=['text'])
  neg_testdata_nosw=pd.read_csv("test_negative_no_stopword.csv",sep="\n",names=['text'])
  full_data=pd.concat((pos_data_nosw,neg_data_nosw,pos_valdata_nosw,neg_valdata_nosw,pos_testdata_nosw,neg_testdata_nosw),axis=0)

  full_data=[eval(full_data.text.iloc[i]) for i in range(len(full_data))]

  #Creating the Word2Vec Model
  w2v_model = Word2Vec(min_count=10,
                      window=2,
                      size=300,
                      sample=6e-5, 
                      alpha=0.03, 
                      min_alpha=0.0007, 
                      negative=20)
  #Setting up the vocabulary for the model
  w2v_model.build_vocab(full_data, progress_per=10000)
  #Training the model
  w2v_model.train(full_data, total_examples=w2v_model.corpus_count, epochs=15, report_delay=1)

  print("20 most similar words to good:")
  similarWordTo("good",20)
  print()
  print()
  print("20 most similar words to bad:")
  similarWordTo("bad",20)



