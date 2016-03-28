
# Machine Learning testbed for sklearn and finances 


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import gensim 
from time import time
import logging
from sklearn.metrics import confusion_matrix

from sklearn import ensemble, neighbors, linear_model, grid_search,preprocessing, decomposition
from sklearn.cross_validation import train_test_split,_num_samples
import sklearn
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline



# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)




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
    names=[]
    for i,c in enumerate(cats_unique):
        
        cat_map[c]=i
        cat_map[i]=c
        
        names.append(c)
        
    return cat_map,names
        
    
        
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

    features['ammounts'] = ammounts.apply(lambda x:np.float(x)).values
    features = pd.concat([features,pd.DataFrame(pd.get_dummies(accounts).values)],axis=1)

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


    ct=0
    ce=0
    c=0
    for comment in comments:
        c+=1
        try:
            doc = comment.encode('utf-8').split()
            vec_bow = dictionary.doc2bow(doc)
            vec_lsi = lsi[vec_bow] # convert the query to LSI space

            #document similaraties  
            sims = index[vec_lsi]
            
            features_topics_sims.append(sims)

            ct+=1
        
        except:
            ce+=1
            features_topics_sims.append([-1]*len(sims))

    logger.debug('1 len features {0}'.format(len(features)))
    logger.debug('1 len features_topics_sims {0}'.format(len(pd.DataFrame(features_topics_sims))))
    
    features = pd.concat([pd.DataFrame(features.values),pd.DataFrame(features_topics_sims)],axis=1)

    logger.debug('2 len features {0}'.format(len(features)))

    logger.debug('text_features loop, c_try: {0}, c_except: {1}, total: {2}'.format(ct,ce,c))
    
    return features

def features_extraction(comments,accounts,ammounts,dates,
                        train=False,cats=None,text_model=None,
                        num_topics=20):
    """extracts features from data
    
    if data_type is test, then it will look for a text_model,
    if data_type is train, then it will create a text_model"""
    
    features = pd.DataFrame()

    logger.debug(len(features))
    
    if train:
        logger.info('creating topic models')
        text_model = create_text_base_models(comments,cats,num_topics)
        
    #extract text features
    df_text_features = text_features(comments,text_model)

    logger.debug('len text_features: {0}'.format(len(df_text_features)))
    
    #extract money features
    df_money_features = money_features(ammounts,accounts)
    
    logger.debug('len money_features: {0}'.format(len(df_money_features)))

    #extract temporal features
    df_temporal_features = temporal_features(dates)
    

    logger.debug('len temporal_features: {0}'.format(len(df_temporal_features)))

    features = pd.concat([
        pd.DataFrame(df_text_features.values),
        pd.DataFrame(df_money_features.values),
        pd.DataFrame(df_temporal_features.values)],axis=1)

    logger.debug('len all_features: {0}'.format(len(features)))
    
    return features


def fit_model_lab(X,y,names,optamise=False, report=False):

    """ This function is for playing around with models for the 
    primary purpos of doing parameter and model optermisation
    in the full machine_learning part of the code it may become
    redudant. However, it offers a platform for experimenting with
    sklearn and xgboost, as well as hyperparameters.

    X is a array of features 
    y is a set of labels in the form of a vectorised
    names are the catgorical names


    optamise runs a RandomizedSearchCV over a parameter space
    note this must be hard coded for each different type of 
    classifier. The below is setup for xgboost.
    report presentes a print out of the how well the classifier
    works in each category as well as plotting confusion_matrixies
    for test and train.

    """


    #scale data
    #scaler_data = preprocessing.StandardScaler().fit(X)
    #X_scaled = scaler_data.transform(X)

    #pca decomposition
    #pca = decomposition.PCA(n_components=10).fit(X)
    #X_scaled_pca = scaler_data.transform(X_scaled)


    #select classifier
    #clf = sklearn.ensemble.RandomForestClassifier()
    #clf = sklearn.ensemble.ExtraTreesClassifier()
    #clf = sklearn.ensemble.GradientBoostingClassifier()
    clf = xgb.sklearn.XGBClassifier()#learning_rate=0.1,silent=1,n_estimators=1000)

    #create pipline
    estimators=[ 
                ('scaler', preprocessing.StandardScaler()),
                ('reduce_dim', decomposition.PCA()), 
                ('model', clf)]

    pipline = Pipeline(estimators)


    #test/train split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


    best_params =  {'model__min_child_weight': 2,
                                'reduce_dim__n_components': 14,
                                'model__gamma': 0.22222222222222221,
                                'model__learning_rate': 0.01,
                                'model__max_delta_step': 4,
                                'model__max_depth': 27,
                                'model__n_estimators': 1000,
                                'model__subsample': 0.66666666666666663}
    



    if optamise:

        # specify parameters and distributions to sample from 

        params_dist = dict(
            reduce_dim__n_components=list(np.linspace(2,len(set(y_train)),len(set(y_train))).astype(np.int)),
            model__n_estimators     =list(np.logspace(1,3,10).astype(np.int)),
            model__learning_rate    =list(np.linspace(0.01,1.,10)),
            model__max_depth        =list(np.linspace(1,40,10).astype(np.int)),
            model__gamma            =np.linspace(0.,1.,10),
            #model__colsample_bytree =np.linspace(0.1,1.,3),
            #model__colsample_bylevel=np.linspace(0.1,1.,3),
            model__max_delta_step   =np.linspace(0,10,10).astype(np.int),
            model__subsample        =np.linspace(0.,1.,10),
            model__min_child_weight =np.linspace(0,10,10).astype(np.int)
            )


        logger.info('performing grid search')
        #grid = grid_search.RandomizedSearchCV(clf,param_dist,n_jobs=-1,verbose=1,n_iter=10,cv=10)

        grid = grid_search.RandomizedSearchCV(
                                    pipline,
                                    params_dist,
                                    n_jobs=1,
                                    verbose=1,
                                    n_iter=10000,
                                    cv=5)

        grid.fit(X_train,y_train)

        logger.info('best_params: '+str(grid.best_params_))
        logger.info('best_score: '+ str(grid.best_score_))

        grid.best_score_


        clf_final=grid


    
    else:


        logger.info('skipping optamisation phase and fitting model')

        pipline.fit(X_train,y_train.ravel())

        clf_final=pipline

    

    train_score = sklearn.metrics.accuracy_score(y_train,clf_final.predict(X_train))
    test_score = sklearn.metrics.accuracy_score(y_test,clf_final.predict(X_test))
    #report on accuary of model and 

    logger.info('accuracy_score: (test)  {0:.3f}'.format(test_score))
    logger.info('accuracy_score: (train) {0:.3f}'.format(train_score))   


    #print '\ntrain\n',sklearn.metrics.classification_report(y_train,clf_final.predict(X_train))

    #print '\ntest\n',sklearn.metrics.classification_report(y_test, clf_final.predict(X_test))

    if report:
        class_report(clf_final,y_train,clf_final.predict(X_train), ttype='train')
        class_report(clf_final,y_test,clf_final.predict(X_test), ttype='test')
 
    return clf_final

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
     

def class_report(clf,p,t, plot=True,ttype='unknown'):
    #caculate the number of correct classfications
    res=t-p
    per_correct_all = len(res[res==0])*1./len(t)
    logging.info('fraction of correct classfications {0:.2f}'.format(per_correct_all))

    cats=[]
    f_correct=[]
    n_correct=[]
    n_cat=[]

    print '\nreport type '+ ttype +'\n'+'-'*80

    print '{0:<40} {1:<10} {2:<10} {3:<10}'.format('cat', 'f_correct', 'N_correct','N_total')

    #find faction of correct 
    for int_cat, name in cat_map.items():
        if type(int_cat) == int:
            try:

                w_cat = t==int_cat

                temp_res=t[w_cat]-p[w_cat]
                print '{0:<40} {1:<10.2f} {2:<10} {3:<10}'.format(name, 
                                                          len(temp_res[temp_res==0])*1./len(p[w_cat]),
                                                          len(temp_res[temp_res==0]),
                                                          len(p[w_cat]))

                n_cat.append(len(p[w_cat]))
                f_correct.append(len(temp_res[temp_res==0])*1./len(p[w_cat]))
                n_correct.append(len(temp_res[temp_res==0]))
                
            except:
                
                w_cat = t==int_cat

                temp_res=t[w_cat]-p[w_cat]
                print '{0:<40} {1:<10.2f} {2:<10} {3:<10}'.format(name, 
                                                          0.,
                                                          0,
                                                          len(p[w_cat]))

                n_cat.append(len(p[w_cat]))
                f_correct.append(0)
                n_correct.append(0.)

    print '\n{0:<40} {1:<10.2f} {2:<10} {3:<10}'.format('ave/total', 
                                                        np.mean(f_correct), 
                                                        np.sum(n_correct),
                                                        np.sum(n_cat))

    print '-'*80+'\n'*2
    
    if plot:
        # Compute confusion matrix
        cm = confusion_matrix(p,t)
        np.set_printoptions(precision=2)

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(figsize=(14,14))
        plot_confusion_matrix(cm_normalized, names, title='Normalized confusion matrix '+ttype)
        fig.savefig('/Users/chrisfuller/Desktop/'+ttype+'.jpg')




if __name__ == '__main__':  
    #load in data
    df_raw = pd.read_csv('~/Documents/Programs/finance_mk2/machine_learning/new_cats_as of_21_210316.csv')

    #rename cols
    cois = [u'date', u'account', u'description', u'payee', u'new_ammounts', u'new_new_cats']
    df_raw2 = df_raw[cois]
    df_raw2.columns =[u'date', u'account', u'description', u'payee', u'ammount',u'category']

        #remove low value cats that will make model fitting hard
    selection = (df_raw2.category.value_counts() <25).reset_index()
    low_val_cats = selection[selection.category]['index'].values
    logger.info('number of transaction removed as less than threshold: {0}'.format(
            len(df_raw2[df_raw2.category.apply(lambda x:x in low_val_cats)])))
                
            
    df=df_raw2[df_raw2.category.apply(lambda x:x not in low_val_cats)]


    

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
    cat_map, names = cat_mapper(df.category)


    #sklearn bit
    X = np.array(features.values, dtype=np.float)
    T = np.array(pd.get_dummies(df['category']).values,dtype=np.float)
    t = np.array(df.category.apply(lambda x: cat_map[x]).values,dtype=np.float)
    #t = t.reshape(len(t),1)

    logger.debug('len(X): {0}'.format(len(X)))
    logger.debug('len(t): {0}'.format(len(t)))
    clf = fit_model_lab(X,t,names, optamise=True,report=True)






