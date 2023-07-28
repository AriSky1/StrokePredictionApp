
# XGB_BEST_PARAMS = {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180}
# xg = XGBClassifier(params=XGB_BEST_PARAMS, random_state=42)
# xg = xg.fit(X_res, y_res)
# cross_val_scores = cross_val_score(xg, X_res, y_res, cv=5)
# print("Cross-Validation Scores:", cross_val_scores)
# print("Mean Cross-Validation Score:", np.mean(cross_val_scores))



# VERIFICATION BY 20 UNSEEN ROWS (10 first P & 10 last N)


print('VERIFICATION BY 20 UNSEEN ROWS (10 first P & 10 last N) :  \n')
input= pd.read_csv('data_test.csv')
# input = input. iloc[:,:-1]
# input=input.drop(columns=input.columns[0], axis=1)
df = pd.read_csv(r'data_strokes_prediction.csv')


# s = pd.Series(input, index=input.keys())
# print('s      \n', s)
# d = pd.DataFrame(s).transpose()
df = df.append(input)
print('df after append', df.tail(10))
# df = df. iloc[:,:-1]
# df=df.drop(columns=df.columns[0], axis=1)
X,y=transformation_pipeline_test(df)

#
# print('df test tail',df.tail(10))

print('Xtail\n',X.tail(20))
# print('input',input)


pipeline2 = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS))])
pipeline2.fit(X_train,y_train)

test_pred = pipeline2.predict(X.tail(20))
print('X.tail(20)\n',X.tail(20))
print('PREDS :' , test_pred)


print(y.shape, test_pred.shape)

# print report
print(classification_report(y.tail(20),test_pred))
print('Accuracy Score: ',accuracy_score(y.tail(20),test_pred))
print('Recall Score: ',recall_score(y.tail(20),test_pred))
print('\nConfusion matrix: \n',confusion_matrix(y.tail(20), test_pred))
print('\n Model :','XGBoost')





xg = XGBClassifier(params=XGB_BEST_PARAMS, random_state=42)


# Fit the training data to the model
xg = xg.fit(X_train, y_train)

# Get cv scores
cross_val_scores = cross_val_score(xg, X_res, y_res, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", np.mean(cross_val_scores))


# Get feature importances
feature_importances = xg.feature_importances_

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': X_res.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# # Plotting feature importances
# plt.figure(figsize=(10, 8))
# sns.barplot(data=importances_df, x='Importance', y='Feature', color='b')
# plt.title('Feature Importance XGBoost')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()
#


# # Hyperparameter tuning using GridSearch
# estimator = XGBClassifier(objective='binary:logistic', nthread=4, seed=random_state)
# parameters = {
#     'max_depth': range(2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05] }
# grid_search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring='roc_auc', n_jobs=10, cv=10, verbose=True)
# grid_search.fit(X_train, y_train)
# # Get the best hyperparameters from GridSearch
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)


# export model
# pkl.dumps(pipeline1, open('best_pipeline1.pkl', 'wb'))
# saved_model = pkl.dumps(pipeline1, 'best_pipeline1.pkl')


# scaler = StandardScaler()
# scale_fit = scaler.fit(X_train, y_train)  # save the mean and std. dev computed for your data.
# dt = scale_fit.transform(dt)  # use the above saved values to scale your single observation or batch observations.