from nose.tools import *
import scipy.io
from skfeature.function.similarity_based import trace_ratio
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def test_trace_ratio():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=20, n_informative=5, n_redundant=5, n_classes=2)
    X = X.astype(float)
    n_samples, n_features = X.shape    # number of samples and number of features

    num_fea = 5
   
    #parameters = {
    #    "select_top_k__n_selected_features": [num_fea]
    #}
    assert(trace_ratio.trace_ratio(X, y, n_selected_features=5), True)
    # build pipeline
    #pipeline = []
    #pipeline.append(('select_top_k', SelectKBest(score_func=trace_ratio.trace_ratio, k=num_fea)))
    #pipeline.append(('linear_svm', svm.LinearSVC()))
    #model = Pipeline(pipeline)
    
    #results = cross_val_score(model, X, y, cv=kfold)
    #clf_cv = GridSearchCV(model, parameters, n_jobs=-1)
    #clf_cv.fit(X, y)
    #result = clf_cv.fit(X)
    #print("Accuracy: {}".format(results.mean()))
    #assert_true(results.mean() > 0.1)
