
# Machine Learning testbed for sklearn and finances 


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import gensim 


# #create some features
def flatten_list_of_lists_clean(lol):
    temp=[]
    for l in lol:
        temp.extend(l)
        
    temp2=[]
    for x in temp:
        try:
            temp2.append(x.encode('utf-8'))
        except:
            continue
    
    return temp2

def cat_mapper(cats):
    cats_unique = df.sort_values('category').category.drop_duplicates().values

    cat_map={}
    for i,c in enumerate(cats_unique):
        
        cat_map[c]=i
        cat_map[i]=c
        
    return cat_map
        
    
        
def temporal_features(dates):
    """ input: dates should be a timestamp of somekind,
    this function should output:
    day of the week fraction
    day of the month fraction
    day of the year fraction
    juliandate date
    """
    
    #initilise a data frame to hold tempral featues
    features=pd.DataFrame()
    
    #create features
    features['day_of_week'] = pd.to_datetime(dates).apply(lambda x: x.dayofweek/7.0)
    features['day_of_month'] = pd.to_datetime(dates).apply(lambda x: x.day*1.0/x.daysinmonth)
    features['day_of_year'] = pd.to_datetime(dates).apply(lambda x: x.dayofyear/365.0)
    features['julian_date']=pd.to_datetime(df.date).apply(lambda x: x.to_julian_date())-2456658.5 

    return features

def money_features(ammounts,accounts):
    """creates some catgorical features as 
    well as some relating to finantial ammount"""
    
    #initilise a data frame to hold features
    features=pd.DataFrame()

    features['ammounts'] = ammounts.apply(lambda x:np.float(x))
    features = pd.concat([features,pd.get_dummies(accounts)],axis=1)

    return features

def create_text_base_models(comments, cats, num_topics):
    """given a set of comments makes some 
    text models such that they can be run on training 
    data only"""
    
    
    #create a dataframe to hold models
    models=pd.DataFrame()
    
    #count the occorances of each word in each cat
    text=pd.concat([comments.apply(lambda x : x.split()),cats],axis=1)


    cat_text = text.groupby('category').agg({'description': lambda x: [x for x in list(x)]})
    cat_text['raw_tokens'] = cat_text.description.apply(flatten_list_of_lists_clean)
    cat_text['token_counts'] = cat_text.raw_tokens.apply(Counter)
    
    cat_text['top_5_tokens'] = cat_text.token_counts.apply(lambda x:
    [token for token,occ in sorted(x.items(),key=lambda (x,y): -y)][:5])
    
    cat_text['top_30per_tokens']= cat_text.token_counts.apply(lambda x: 
    [token for token,occ in sorted(x.items(),key=lambda (x,y): -y)][:int(len(list(x))*0.3)])
    
    #create documents
    docs=cat_text.raw_tokens.values 

    #map documents to a dictionary
    dictionary = gensim.corpora.Dictionary(docs)
    dictionary.save('/tmp/ml_data.dict') # store the dictionary, for future reference
    #print(dictionary)
    
    #make corpus -- A list of vectorised documents
    corpus = [dictionary.doc2bow(d) for d in docs]
    gensim.corpora.MmCorpus.serialize('/tmp/ml_data.mm', corpus) 

    #create lsi model crom corpus
    lsi = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

    index = gensim.similarities.MatrixSimilarity(lsi[corpus])
    index.save('/tmp/deerwester.index')
    index = gensim.similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    
    #text model for documents
    text_mod = (index,dictionary,lsi)
    
    
    return text_mod

###text processing ####
def text_features(comments,text_model):
    """return simple text features"""
    
    #initilise a data frame to hold features
    features=pd.DataFrame()
    
    tokenised=comments.apply(lambda x:x.split())
    
    features['num_words'] = tokenised.apply(len)
    
    #create text features based on topic models
    index,dictionary,lsi = text_model
    
    features_topics_sims=[]
    
    for comment in comments:
        try:
            doc = comment.encode('utf-8').split()
            vec_bow = dictionary.doc2bow(doc)
            vec_lsi = lsi[vec_bow] # convert the query to LSI space

            #document similaraties  
            sims = index[vec_lsi]
            
            features_topics_sims.append(sims)
        
        except:
            features_topics_sims.append([-1]*len(sims))
        
    
    features = pd.concat([features,pd.DataFrame(features_topics_sims)],axis=1)
    
    return features

def features_extraction(comments,accounts,ammounts,dates,
                        train=False,cats=None,text_model=None,
                        num_topics=20):
    """extracts features from data
    
    if data_type is test, then it will look for a text_model,
    if data_type is train, then it will create a text_model"""
    
    features = pd.DataFrame()
    
    if train:
        print 'creating topic models...'
        text_model = create_text_base_models(comments,cats,num_topics)
        
    #extract text features
    df_text_features = text_features(df.description,text_model)
    
    #extract money features
    df_money_features = money_features(ammounts,accounts)
    
    #extract temporal features
    df_temporal_features = temporal_features(dates)
    
    features = pd.concat([df_text_features,df_money_features,df_temporal_features],axis=1)
    
    return features

def create_report(model): 
    pass        




#load in data
df_raw = pd.read_csv('new_cats_as of_21_210316.csv')

#rename cols
cois = [u'date', u'account', u'description', u'payee', u'new_ammounts', u'new_new_cats']
df = df_raw[cois]
df.columns =[u'date', u'account', u'description', u'payee', u'ammount',u'category']


from sklearn import preprocessing
from sklearn import ensemble, neighbors, linear_model, grid_search
from sklearn.cross_validation import train_test_split,_num_samples
import sklearn
from sklearn.pipeline import Pipeline


#feature creation
features = features_extraction(comments=df.description, 
                               accounts=df.account, 
                               ammounts=df.ammount, 
                               dates=df.date, 
                               train=True, 
                               cats=df.category, 
                               text_model=None,
                               num_topics=20)

#object maps ints to cats or cats to ints
cat_map = cat_mapper(df.category)


#sklearn bit
X = np.array(features.values, dtype=np.float)
T = np.array(pd.get_dummies(df['category']).values,dtype=np.float)
t = np.array(df.category.apply(lambda x: cat_map[x]).values,dtype=np.float)



#scale data
scaler_data = preprocessing.StandardScaler().fit(X)
scaler_target = preprocessing.StandardScaler().fit(T)

X_scaled = scaler_data.transform(X)
T_scaled = scaler_target.transform(T)

#test train split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,t, test_size=0.2)


#ml
#clf = ensemble.RandomForestClassifier(n_estimators=20,n_jobs=4)
#clf = linear_model.LogisticRegression()
#clf = linear_model.RidgeClassifierCV()
#clf = neighbors.KNeighborsClassifier(n_neighbors=1)
#clf = ensemble.GradientBoostingClassifier(n_estimators=100, 
#                                          learning_rate=1.0,
#                                          max_depth=1, 
#                                          random_state=0)



clf = ensemble.RandomForestClassifier()
#clf = ensemble.ExtraTreesClassifier()
#clf = ensemble.GradientBoostingClassifier()

#clf.fit(X_train,y_train)

# specify parameters and distributions to sample from
param_dist = {    #"learning_rate": [0.05,0.1,0.5], #GBM only
                  "n_estimators": [60,100,200,500,1000],
                  "max_depth": [2,10,40,60,80,120,200]
                  }




print('performing grid search...')
grid = grid_search.GridSearchCV(clf,param_dist,n_jobs=-1,cv=10,verbose=1)#,n_iter=10,cv=10)

grid.fit(X_train,y_train)

print 
print 'best score: ', grid.best_score_
print 
print 'best parameters', grid.best_params_


clf_final=grid

if False:
    print 
    print 'logloss: (train) ', sklearn.metrics.log_loss(y_train,clf_final.predict(X_train))
    print 'logloss: (test) ', sklearn.metrics.log_loss(y_test,clf_final.predict(X_test))
    print 
    
if False:
    print 
    print 'jackard: (train) ', sklearn.metrics.jaccard_similarity_score(y_train,clf_final.predict(X_train))
    print 'jackard: (test) ', sklearn.metrics.jaccard_similarity_score(y_test,clf_final.predict(X_test))
    print 
    
if True:
    print 
    print 'accuracy_score: (train) ', sklearn.metrics.accuracy_score(y_train,clf_final.predict(X_train))
    print 'accuracy_score: (test) ', sklearn.metrics.accuracy_score(y_test,clf_final.predict(X_test))
    print 




if False:
    clf = ensemble.GradientBoostingClassifier(n_estimators=2000, 
                                              learning_rate=.05,
                                              max_depth=2, 
                                              random_state=0).fit(X_train,y_train)

    print 'accuracy_score: (test) ', sklearn.metrics.accuracy_score(y_test,clf.predict(X_test))