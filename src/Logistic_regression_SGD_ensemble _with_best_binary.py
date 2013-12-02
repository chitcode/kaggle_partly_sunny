#submited , leader board score =  	0.15610
#Author : Chitrasen
#Date : 11/1/2013
# TF (no idf) , best p value, 
#combning all the classifiers

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


def y_binary_p(y,p):
    if y <= p:
        return 0    
    else:
        return 1
    
def scaleCols(df):
    s_total = df['s1']+df['s2']+df['s3']+df['s4']+df['s5']
    df['s1'] = df['s1']/s_total
    df['s2'] = df['s2']/s_total
    df['s3'] = df['s3']/s_total
    df['s4'] = df['s4']/s_total
    df['s5'] = df['s5']/s_total
    
    w_total = df['w1']+df['w2']+df['w3']+df['w4']
    df['w1'] = df['w1']/w_total
    df['w2'] = df['w2']/w_total
    df['w3'] = df['w3']/w_total
    df['w4'] = df['w4']/w_total
    
    return df


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
    
    pred3 = pd.DataFrame(np.zeros((test.shape[0], train.shape[1]- test.shape[1])), columns = train.columns[4:])
    pred3.index = test['id'] 
    
    pred4 = pd.DataFrame(np.zeros((test.shape[0], train.shape[1]- test.shape[1])), columns = train.columns[4:])
    pred4.index = test['id']
    
    
    tweets_train = train['tweet']
    tweets_test = test['tweet']


    tfidf1 = TfidfVectorizer(min_df=5,max_df = .95,max_features=10000, strip_accents='unicode',  
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,2), use_idf=False,sublinear_tf=1)
    
    tfidf2 = TfidfVectorizer(min_df=5,max_df = .95,max_features=10000, strip_accents='unicode',  
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,2), use_idf=True,sublinear_tf=1)
    
    
    lr = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                           C=1, fit_intercept=True, intercept_scaling=1.0, 
                           class_weight=None, random_state=None)
    
    clf = lm.SGDClassifier(loss='modified_huber',n_iter= 100)
    
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
   
    
    optim_p1 = {'s1':0.3,'s2':0.5,'s3':0.4,'s4':0.4,'s5':0.4,
         'w1':0.6,'w2':0.4,'w3':0.4,'w4':0.4,
         'k1':0.3,'k2':0.3,'k3':0.3,'k4':0.4,'k5':0.5,'k6':0.1,'k7':0.5,'k8':0.3,'k9':0.3,'k10':0.4,
            'k11':0.1,'k12':0.4,'k13':0.3,'k14':0.1,'k15':0.5}
    
    optim_p2 = {'s1':0.3,'s2':0.5,'s3':0.4,'s4':0.5,'s5':0.5,
         'w1':0.5,'w2':0.4,'w3':0.4,'w4':0.4,
         'k1':0.4,'k2':0.4,'k3':0.4,'k4':0.4,'k5':0.5,'k6':0.3,'k7':0.5,'k8':0.3,'k9':0.4,'k10':0.4,'k11':0.4,'k12':0.4,'k13':0.4,'k14':0.4,'k15':0.5}
    
    print 'Finding the optimum p ... with 10 fold CV'
    
    
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
        
        pred_cv3 = pd.DataFrame(np.zeros(y_cv_kf.shape), columns = train.columns[4:])
        pred_cv4 = pd.DataFrame(np.zeros(y_cv_kf.shape), columns = train.columns[4:])
        
        for y_col in train.columns[4:]:
            y = train[y_col]            
            y_train_kf = y[train_idx]            
            y_train_kf1 = vec_y_binary_p(y_train_kf,optim_p1[y_col])
            
            y_train_kf2 = vec_y_binary_p(y_train_kf,optim_p2[y_col])
            
            lr.fit(X_train_kf1,y_train_kf1)
            pred_cv1[y_col] = lr.predict_proba(X_cv_kf1)[:,1]
            
            lr.fit(X_train_kf2,y_train_kf1)
            pred_cv2[y_col] = lr.predict_proba(X_cv_kf2)[:,1]        
            #print '    error for',y_col,sqrt(mean_squared_error(y_cv_kf[y_col],(pred_cv1[y_col] + pred_cv2[y_col])/2))
            
            clf.fit(X_train_kf1,y_train_kf2.values)
            pred_cv3[y_col] = clf.predict_proba(X_cv_kf1)[:,1]
            
            clf.fit(X_train_kf2,y_train_kf2.values)
            pred_cv4[y_col] = clf.predict_proba(X_cv_kf2)[:,1]
           
        pred_cv =  0.25 * pred_cv1 + 0.25 * pred_cv2 + 0.25 * pred_cv3 + 0.25 * pred_cv4
        error = sqrt(mean_squared_error(y_cv_kf,scaleCols(pred_cv)))
        final_error.append(error)
        print 'final error for cv',j, error
    
    print 'Mean error for 10 fold CV is ',np.mean(final_error)
    
    
    for y_col in train.columns[4:]:
        y = train[y_col]
        y1 = vec_y_binary_p(y,optim_p1[y_col])
        
        lr.fit( X_train1, y1)        
        pred1[y_col] = lr.predict_proba(X_test1)[:,1]
        
        lr.fit( X_train2, y1)        
        pred2[y_col] = lr.predict_proba(X_test2)[:,1]
        
        
        y2 = vec_y_binary_p(y,optim_p2[y_col])
        
        clf.fit( X_train1, y2.values)        
        pred3[y_col] = clf.predict_proba(X_test1)[:,1]
        
        clf.fit( X_train2, y2.values)        
        pred4[y_col] = clf.predict_proba(X_test2)[:,1]
        
    pred = scaleCols((pred1 + pred2 + pred3 + pred4)/4)    
    pred.to_csv('benchmark_lr_sgd_ensemble_optim_p.csv')
    print 'submission file created'

if __name__ == "__main__":
    main()
