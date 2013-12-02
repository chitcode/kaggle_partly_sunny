#Author : Chitrasen
#Date : 11/1/2013
# Basic benchmark code

import pandas as pd
from sklearn import cross_validation,metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model as lm


train = pd.read_csv('/home/chitra/data_local/kaggle_data/partly_sunny/train.csv',header=0)
test = pd.read_csv('/home/chitra/data_local/kaggle_data/partly_sunny/test.csv',header=0)

pred = pd.DataFrame(np.zeros((test.shape[0], train.shape[1]- test.shape[1])), columns = train.columns[4:])
pred.index = test['id']


def y_binary(y):
    if y >= 0.5:
        return 1
    else:
        return 0
    

def main():
    print 'reading the datasets ....'
    
    
    tweets_train = train['tweet']
    tweets_test = test['tweet']


    tfidf = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
    
    #lr = lm.Ridge(normalize =True, alpha = 10,solver = "lsqr")
    lr = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                           C=1, fit_intercept=True, intercept_scaling=1.0, 
                           class_weight=None, random_state=None)
    
    print 'fitting TfIDF...'
    trainLen = tweets_train.shape[0]
    X_all = list(tweets_train)+list(tweets_test)

    tfidf.fit(X_all)
    X_all = tfidf.transform(X_all)
    
    X_train = X_all[:trainLen]
    X_test = X_all[trainLen:]
    
    vec_y_binary = np.vectorize(y_binary)
    
    final_cv_error = 0
    
    for i in train.columns[4:]:
        y = train[i]
        y = vec_y_binary(y)
        cv_score = np.mean(cross_validation.cross_val_score(lr, X_train, y, cv=20, scoring='mean_squared_error'))
        print "20 Fold CV Score for: ",i, cv_score
        final_cv_error += np.abs(cv_score)
        
        lr.fit( X_train, y)
        pred[i] = lr.predict_proba(X_test)[:,1]
        
    print 'Final Mean Squared Error for CV ',np.sqrt(final_cv_error/24)
    pred.to_csv('benchmark.csv')
    print 'submission file created'

if __name__ == "__main__":
    main()

