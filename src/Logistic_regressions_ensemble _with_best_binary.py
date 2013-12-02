#submitted . 
# CV Score # 0.148805764072
#Public leader board score # 0.15621
#Author : Chitrasen
#Date : 11/1/2013
# TF (no idf + with idf) , best p value, 

import pandas as pd
from sklearn import cross_validation,metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error
import re
#import PorterStemmer

def clean(s):
    s = s.lower()
    s = re.sub('n\'t',' not',s)
    #s = re.sub(r'\W+', ' ', s)
    #s = re.sub('\s+', ' ', s)
    #s = re.sub('\d+','00',s)
    return s

def removeStopWords(s):
    stopWordsStr = 'a about above after again against all am an and any are as at be because been before being below between both but by cannot could did do does doing down during each few for from further had has have having he he\'d he\'ll he\'s her here here\'s hers herself him himself his how how\'s i i\'d i\'ll i\'m i\'ve if in into is it it\'s its itself let\'s me more most my myself no nor not of off on once only or other ought our ours  ourselves out over own rt same she she\'d she\'ll she\'s should so some such than that that\'s the their theirs them themselves then there there\'s these they they\'d they\'ll they\'re they\'ve this those through to too under until up very was we we\'d we\'ll we\'re we\'ve were what what\'s when when\'s where where\'s which while who who\'s whom why why\'s with would you you\'d you\'ll you\'re you\'ve your yours yourself yourselves'
    stopWords = re.split('\s+',stopWordsStr)
    cleanwords = [w for w in re.split('\W+',s) if w not in stopWords]
    
    p = PorterStemmer()
    
    s = ''
    for word in cleanwords:
        s += " " +p.stem(word,0,len(word)-1)
    return s

def y_binary_p(y,p):
    if y <= p:
        return 0    
    else:
        return 1


def main():
    
    print 'reading the datasets ....'
    train = pd.read_csv('/home/trupti/data_local/kaggle/partly_sunny/train.csv',header = 0)
    test = pd.read_csv('/home/trupti/data_local/kaggle/partly_sunny/test.csv', header = 0)
    
    train['tweet'] = [tweet+' '+str(state) for tweet,state in zip(train.tweet,train.state)]
    test['tweet'] = [tweet+' '+str(state) for tweet,state in zip(test.tweet,test.state)]

    pred1 = pd.DataFrame(np.zeros((test.shape[0], train.shape[1]- test.shape[1])), columns = train.columns[4:])
    pred1.index = test['id'] 
    
    pred2 = pd.DataFrame(np.zeros((test.shape[0], train.shape[1]- test.shape[1])), columns = train.columns[4:])
    pred2.index = test['id']   
    
    
    tweets_train = train['tweet']
    tweets_test = test['tweet']


    tfidf1 = TfidfVectorizer(min_df=5,max_df = .95,max_features=10000, strip_accents='unicode',  
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,2), use_idf=False,sublinear_tf=1)
    
    tfidf2 = TfidfVectorizer(min_df=5,max_df = .95,max_features=10000, strip_accents='unicode',  
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,2), use_idf=True,sublinear_tf=1)
    
    
    lr = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                           C=1, fit_intercept=True, intercept_scaling=1.0, 
                           class_weight=None, random_state=None)
    
    print 'fitting TfIDF...'
    trainLen = tweets_train.shape[0]
    X_all = list(tweets_train)+list(tweets_test)
    X_all = [clean(s) for s in X_all]

    tfidf1.fit(X_all)
    X_all1 = tfidf1.transform(X_all)
    
    tfidf2.fit(X_all)
    X_all2 = tfidf2.transform(X_all)
    
    X_train1 = X_all1[:trainLen]
    X_test1 = X_all1[trainLen:]
    
    X_train2 = X_all2[:trainLen]
    X_test2 = X_all2[trainLen:]
    
    vec_y_binary_p = np.vectorize(y_binary_p)
    
    final_cv_error = 0
    
    #optim_p = {}
   
    
    optim_p = {'s1':0.3,'s2':0.5,'s3':0.4,'s4':0.4,'s5':0.4,
         'w1':0.6,'w2':0.4,'w3':0.4,'w4':0.4,
         'k1':0.3,'k2':0.3,'k3':0.3,'k4':0.4,'k5':0.5,'k6':0.1,'k7':0.5,'k8':0.3,'k9':0.3,'k10':0.4,
            'k11':0.1,'k12':0.4,'k13':0.3,'k14':0.1,'k15':0.5}
    print 'Finding the optimum p ... with 10 fold CV'
    
    #kfs = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10,test_size=.25, random_state=0)
    
    
    #for y_col in train.columns[4:]:
    #    min_error = 100
    #    for i in linspace(0.1,1.0,num = 10):
    #        mse_measure = []
    #        for train_idx,cv_idx in kfs:
    #            X_train_kf = X_train[train_idx,:]
    #            X_cv_kf = X_train[cv_idx,:]
    #            
    #            y = train[y_col]
    #            y_train_kf = y[train_idx]
    #            y_cv_kf = y[cv_idx]
    #            
    #            y_train_kf = vec_y_binary_p(y_train_kf,i)
    #            lr.fit(X_train_kf,y_train_kf)
    #                           
    #            mse_measure.append(mean_squared_error(y_cv_kf,lr.predict_proba(X_cv_kf)[:,1]))
    #        current_error =  np.mean(mse_measure)
    #        if current_error < min_error:
    #            min_error = current_error
    #            optim_p[y_col] = i
    #        else:
    #            print 'optimum p for ',y_col,optim_p[y_col]
    #            break        
    
    print 'CV score with optimun p'
    
    kf = cross_validation.KFold(X_train1.shape[0],n_folds = 10)
    
    final_error = []
    j = 0
    for train_idx,cv_idx in kf: 
        j +=1
        print 'prcessing CV :',j
        X_train_kf1 = X_train1[train_idx,:]
        X_train_kf2 = X_train2[train_idx,:]
        
        X_cv_kf1 = X_train1[cv_idx,:]
        X_cv_kf2 = X_train2[cv_idx,:]
        
        
        y_cv_kf = train.ix[cv_idx,4:]
        #creating a place holder of cv predictions
        pred_cv1 = pd.DataFrame(np.zeros(y_cv_kf.shape), columns = train.columns[4:])
        pred_cv2 = pd.DataFrame(np.zeros(y_cv_kf.shape), columns = train.columns[4:])
        
        for y_col in train.columns[4:]:
            y = train[y_col]            
            y_train_kf = y[train_idx]            
            y_train_kf = vec_y_binary_p(y_train_kf,optim_p[y_col])
            
            lr.fit(X_train_kf1,y_train_kf)
            pred_cv1[y_col] = lr.predict_proba(X_cv_kf1)[:,1]
            
            lr.fit(X_train_kf2,y_train_kf)
            pred_cv2[y_col] = lr.predict_proba(X_cv_kf2)[:,1]        
            #print '    error for',y_col,sqrt(mean_squared_error(y_cv_kf[y_col],(pred_cv1[y_col] + pred_cv2[y_col])/2))
           
        pred_cv =  (pred_cv1 + pred_cv2)/2  
        error = sqrt(mean_squared_error(y_cv_kf,pred_cv))
        final_error.append(error)
        print 'final error for cv',j, error
    
    print 'Mean error for 10 fold CV is ',np.mean(final_error)
    
    
    for y_col in train.columns[4:]:
        y = train[y_col]
        y = vec_y_binary_p(y,optim_p[y_col])
        
        lr.fit( X_train1, y)        
        pred1[y_col] = lr.predict_proba(X_test1)[:,1]
        
        lr.fit( X_train2, y)        
        pred2[y_col] = lr.predict_proba(X_test2)[:,1]
        
    pred = (pred1 + pred2)/2
    pred.to_csv('benchmark_lr_ensemble_optim_p.csv')
    print 'submission file created'

if __name__ == "__main__":
    main()
