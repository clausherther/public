import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve, auc

from sklearn import metrics
pd.options.mode.chained_assignment = None 

sns.set(color_codes=True)

#------------------------------------------------------------------------------
# Plotting helper functions
#------------------------------------------------------------------------------
def plot_confusion_matrix(cm, title='Confusion matrix', labels=['True', 'False'], cmap=plt.cm.Blues):
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc(actual, predictions):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)

    roc_auc = auc(false_positive_rate, true_positive_rate)
    print "thresholds:", thresholds
    print "AUC:", roc_auc
    
    plt.figure(figsize=(12,9))
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    return false_positive_rate, true_positive_rate, thresholds, roc_auc

def plot_histogram(data, bins=50):
    f = plt.figure(figsize=(12,9))
    plt.title('Histogram')
    ax = f.add_subplot(111)
    ax.hist(data, bins=bins)
    plt.show()

def run_prediction(X, y, model, test_size=.25, Xtransformer=None):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    print '----------------------------'
    print 'X_train\t',X_train.shape
    print 'X_test\t',X_test.shape
    print 'y_train\t',y_train.shape
    print 'y_test\t',y_test.shape
    print '----------------------------'
    
    print '\n'
    print '-----------------------------------------'
    print 'Running model with train/test split...'
    print '-----------------------------------------'

    if not Xtransformer is None:
        X_train = Xtransformer.fit_transform(X_train)
        X_test = Xtransformer.transform(X_test)

    #fit on training data
    fit = model.fit(X_train, y_train)


    # if hasattr(model, 'coef_'):
    #     print '\n'
    #     print 'Coefficients for each X:\n'
    #     coeff = zip(cols, model.coef_[0])
    #     for c, e in coeff:
    #         print c, '\t', e

#     y_pred = None
#     if hasattr(model, 'predict_proba'):
#         y_pred = model.predict_proba(X_test_feat)
#     else:

	# predict on test data
    y_pred = model.predict(X_test)
    
    y_score = None
    if hasattr(model, 'decision_function'):
        y_score = fit.decision_function(X_test)
    else:
        print 'model does not have decision_function()'

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print "Accuracy of model:\t", accuracy.ravel()
  
    results = { 
                'X_train': X_train, 
                'X_test': X_test, 
                'y_train': y_train, 
                'y_test': y_test, 
                'y_pred': y_pred
           }

    return model, results, y_score

def confusion_matrix(y_test, y_pred):

    cm = metrics.confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=3)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm, cm_normalized

def model_metrics(model, X, y, data_split):
    
    print '-----------------------------------------'
    print 'Metrics:'
    print '-----------------------------------------'
    y_test = data_split['y_test']
    y_pred = data_split['y_pred']
    
    X_train = data_split['X_train']
    X_test = data_split['X_test']
    y_train = data_split['y_train']
    
    print 'MSE\t', metrics.mean_squared_error(y_test, y_pred)
    print 'RMSE\t', np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    
    print '\n'
    print '-----------------------------------------'
    print 'Scores:'
    print '-----------------------------------------'
    print 'Train\t', score_train
    print 'Test\t', score_test
    print '-----------------------------------------\n'

    return score_test

def cross_validation(model, X, y, n_jobs=1):
    
    print '-----------------------------------------'
    print 'Running model using Cross Validation:'
    print '-----------------------------------------'

    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=n_jobs, verbose=1)
    score_cross_val_mean = scores.mean()
    print 'Mean Cross-Val Score:', score_cross_val_mean

    return scores        