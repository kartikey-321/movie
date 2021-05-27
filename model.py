import numpy as np
import pandas as pd
import joblib
import nltk
# nltk.download("wordnet", "./nltk_data/")
# nltk.data.path.append('./nltk_data/')
df=pd.read_csv('./mv_train.csv')
d={'pos' : 1,'neg' : 0}
df['label']=df['label'].map(d)
x=df['review'].tolist()
y=df[['label']]

from nltk.corpus import stopwords
sw=set(stopwords.words('english'))

def filtering(text,sw):
    for i in sw:
        for j in text:
            if i==j:
                text.remove(i)
    return text

from nltk.stem.snowball import SnowballStemmer,PorterStemmer
ps=PorterStemmer()
from nltk.tokenize import RegexpTokenizer
token=RegexpTokenizer(r'\w+')

def mymovie(text):
    movie_list=[]
    for document in text:
        words1=[]
        words=document.lower()
        words=token.tokenize(words)
        words=filtering(words,sw)
        for i in words:
            words1.append(ps.stem(i))
        words=' '.join(words1)
        movie_list.append(words)
    return movie_list

result=mymovie(x)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(1,3))
x=cv.fit_transform(result)
y=np.array(y)
joblib.dump(cv,'transform.pkl')

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(x,y)
joblib.dump(mnb,'movie.pkl')