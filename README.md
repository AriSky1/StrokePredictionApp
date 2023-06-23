# StrokePredictionApp

Made with FLASK / XGBOOST


This app uses XGBoost optimized algorithm and a public dataset from the World Health Organization.

An efficient technique of over + undersampling applied to the whole dataset allowed to get around 92% of accuracy, precision, recall, f-1 score.
It is used in cases of very unbalanced data. First, negative stroke cases were reduced, then positive stroke cases were added. 
Using just oversampling didn't give good results with any model.

The selected model is XGBClassifier from sklearn library.


The purpose was to learn how to build an app on flask and train, integrate a Machine Learning classification algorithm.




https://user-images.githubusercontent.com/105871709/225194366-21e4e0ba-f538-4ecd-930b-3b85320c98cc.mp4

