# StrokePredictionApp



This app uses XGBoost optimized algorithm and a public dataset from the World Health Organization.



An efficient technique of over + undersampling applied to the whole dataset allowed to around 92% of accuracy, precision, recall, f-1 score.
It is used in cases of very unbalanced data. First, negative stroke cases were reduced, then positive stroke cases were added. 
Using just oversampling didn't give good results with any model.

The selected model is XGBClassifier from sklearn library.


EDA part ommited in the notebooks for this project - the purpose was to learn how to build an app on flask and train, integrate a Machine Learning classification algorithm.


https://user-images.githubusercontent.com/105871709/225187226-1268e4a4-ba4f-43bf-8c54-4f548718d3be.mp4

