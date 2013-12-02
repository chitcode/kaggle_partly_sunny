#Author : Chitrasen
#Date : 11/1/2013
# Basic benchmark code LogisticRegression with sampling agents

import pandas as pd
import re
from sklearn import cross_validation,metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier

#import PorterStemmer

def clean(s):
    s = s.lower()
    s = re.sub('n\'t',' not',s)
    s = re.sub('{link}',' ',s)
    s = re.sub('#\w+',' ',s)
    s = re.sub('@\w+',' ',s)
    s = re.sub('[\W]',' ',s)
    s = re.sub('\s+',' ',s)
    s = removeStopWords(s)
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


def normalize_prob(a):
    row_sums = a.sum(axis =1)
    return a/row_sums[:,np.newaxis]




def isItRetweet(df):
    return [len(re.findall('^rt', tweet)) > 0 for tweet in df.tweet]

def hashTagsCount(df):
    return [len(re.findall('\#.',tweet)) for tweet in df.tweet]

def mentionCount(df):
    return [len(re.findall('\@.',tweet)) for tweet in df.tweet]

def numericContains(df):
    return [len(re.findall('[0-9]+[:.]*[0-9]*.',tweet)) for tweet in df.tweet]
    
def containsAMPM(df):
    return [len(re.findall('[0-9]+\s*am|pm',tweet)) for tweet in df.tweet]
    
def linksCount(df):
    return [len(re.findall('{link}',tweet)) for tweet in df.tweet]

def containsHashWeather(df):
    return [len(re.findall('\#weather',tweet)) for tweet in df.tweet]

def tweetLength(df):
    return [len(tweet) for tweet in df.tweet]

def getHighFeqWords(df,col,percent = 0.85, stop_words = 'english',max_features = 50):
    df1 = df['tweet'][df[col] > percent]
    countVect = CountVectorizer(strip_accents = 'ascii',analyzer='word',stop_words = stop_words,
                                max_features = max_features,binary = True)    
    
    countVect.fit(df1.values)
    maxFeatures = countVect.fit(df['tweet'][df[col] > percent]).get_feature_names()
    print maxFeatures
    countVect = CountVectorizer(vocabulary=maxFeatures)
    return countVect.fit_transform(df.tweet).toarray()

def getFeatureVecture(df,function_list = None,col_category=None):
    
    returnVals = None
    
    if(function_list == None):
        function_chain = [isItRetweet,hashTagsCount,mentionCount,numericContains,containsAMPM,linksCount,
                      containsHashWeather,tweetLength]
        returnVals = np.hstack([np.matrix(func(df)).T for func in function_chain])
        
    return returnVals


def main():
    print 'Reading datasets...'
    train = pd.read_csv('/home/trupti/data_local/kaggle/partly_sunny/train.csv',header = 0)
    test = pd.read_csv('/home/trupti/data_local/kaggle/partly_sunny/test.csv', header = 0)


    pred = pd.DataFrame(np.zeros((test.shape[0], train.shape[1]- test.shape[1])), columns = train.columns[4:])
    pred.index = test['id']   


    tweets_train = train['tweet']
    tweets_test = test['tweet']
    #X_all = list(tweets_train)+list(tweets_test)
    
    X_all_a = list(tweets_train)+list(tweets_test)
    X_all = [clean(s) for s in X_all_a]
    
    tfidf = CountVectorizer(X_all,min_df=5,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,1),stop_words = 'english')
    
    lr = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                           C=1, fit_intercept=True, intercept_scaling=1.0, 
                           class_weight=None, random_state=None)

    clf = DecisionTreeClassifier()
    
    print 'fitting TfIDF...'
    trainLen = tweets_train.shape[0]


    tfidf.fit(X_all)
    X_all = tfidf.transform(X_all)
    lsa = TruncatedSVD(n_components = 100)

    print 'Fitting in LSA'
    X_all = lsa.fit_transform(X_all)

    X_train = X_all[:trainLen]
    X_test = X_all[trainLen:]
    
    X_train = np.hstack((X_train,getFeatureVecture(train)))


    # sampling from the data and fitting indivisual samples linear model and average the results

    final_cv_error = 0
    sampling_count = 10
 
    kf = cross_validation.KFold(X_train.shape[0],n_folds = 10)
    cols = {'s':['s1','s2','s3','s4','s5'],
        'w':['w1','w2','w3','w4'],
        'k':['k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']}

    col_keys = ['s','w','k']
    final_error = []
    j = 0

    for train_idx,cv_idx in kf: 
        j +=1
        print 'prcessing CV :',j
        X_train_kf = X_train[train_idx,:]
        X_cv_kf = X_train[cv_idx,:]
        
        y_train_kf = train.ix[train_idx,4:]
        y_cv_kf = train.ix[cv_idx,4:]
        #creating a place holder of cv predictions
        pred_cv = pd.DataFrame(np.zeros(y_cv_kf.shape), columns = train.columns[4:])
        
        error_pred = 100
        
        for y_col in col_keys:
            #do this for 100s simulation
            
            print '    Normalizing prediction values'
            y_train_kf_norm = normalize_prob(y_train_kf[cols[y_col]].values)
            
            pred_sampling = np.zeros((len(cv_idx),sampling_count))
            
            #print 'sampling for ',y_col,sampling
            sim_data = [np.random.choice(np.arange(len(cols[y_col])),
                                     sampling_count,replace = True, 
                                     p = y_train_kf_norm[i,:]) for i in np.arange(len(train_idx))]
            sim_data = np.array(sim_data)
            print '    sim_data shape', sim_data.shape
            #print sim_data[:10]
            for sampling in range(sampling_count): 
                #print 'sim_data', sim_data[sampling]
                clf.fit(X_train_kf,sim_data[:,sampling])
                pred_sampling[:,sampling] = clf.predict(X_cv_kf)
            pred_sampling = pd.DataFrame(pred_sampling)
            
            print '    processed all the samplings'
            
            col_index = 0
            for col in cols[y_col]:            
                col_val = pred_sampling == col_index            
                pred_cv[col] = np.array(col_val.sum(axis = 1), dtype = float)/sampling_count
                col_index +=1
        print 'Score CV',j, sqrt(mean_squared_error(y_cv_kf,pred_cv))

if __name__ == '__main__':
    main()
