import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis


newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
docs_raw = newsgroups.data
print(len(docs_raw))

#cleaning and creating token
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b',
                                max_df = .5, 
                                min_df = 1)

#creating the numeric representation
dtm = tf_vectorizer.fit_transform(docs_raw)
print(dtm.shape)

#the lda model
lda_tf = LatentDirichletAllocation(n_components=10, max_iter=50, random_state=0)
lda_tf.fit(dtm)

#creating visualization and showing in browser
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
visData = pyLDAvis.sklearn.prepare(lda_tf, dtm, tf_vectorizer)
pyLDAvis.show(visData)