# XGB_BEST_PARAMS = {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180}
# pipeline = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS))])
# pipeline=pipeline.fit(X_train,y_train)
#
#
# print('X_train\n', X_train)
# print('y_train\n', y_train)
#
# xg = XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS)
# xg = xg.fit(X_res, y_res)
# cross_val_scores = cross_val_score(xg, X_res, y_res, cv=5)
# print("Cross-Validation Scores:", cross_val_scores)
# print("Mean Cross-Validation Score:", np.mean(cross_val_scores))
# # Hyperparameter tuning using GridSearch
# estimator = XGBClassifier(learning_rate= 0.1, max_depth=9, n_estimators= 180)
# parameters = {
#     'max_depth': range(2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05] }
# grid_search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring='roc_auc', n_jobs=10, cv=10, verbose=True)
# grid_search.fit(X_train, y_train)
# # Get the best hyperparameters from GridSearch
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
#
# tuned_pred = pipeline.predict(X_test)
#
#
# dt = pd.read_csv('data_test.csv')
# print('dt\n', dt)
#
# X_test_dt, y_test_dt = transformation_pipeline_nores(dt)
#
#
# prediction_label_dt = pipeline.predict(X_test_dt)
# print('prediction_label_dt', prediction_label_dt)
#
#
# print(classification_report(y_test_dt, prediction_label_dt))
# print('Accuracy Score: ', accuracy_score(y_test_dt, prediction_label_dt))
# print('Recall Score: ', recall_score(y_test_dt, prediction_label_dt))
# print('\nConfusion matrix: \n', confusion_matrix(y_test_dt, prediction_label_dt))
# print('\n Model :', 'XGB')








# LGBM_BEST_PARAMS = { }
# XGB_BEST_PARAMS = {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180}
# pipeline = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",LGBMClassifier(random_state=random_state))])
# pipeline=pipeline.fit(X_train,y_train)
#
#
# print('X_train\n', X_train)
# print('y_train\n', y_train)
#
# xg = LGBMClassifier(random_state=random_state)
# xg = xg.fit(X_res, y_res)
# cross_val_scores = cross_val_score(xg, X_res, y_res, cv=5)
# print("Cross-Validation Scores:", cross_val_scores)
# print("Mean Cross-Validation Score:", np.mean(cross_val_scores))
#
#
# tuned_pred = pipeline.predict(X_test)
#
#
# dt = pd.read_csv('data_test.csv')
# print('dt\n', dt)
#
# X_test_dt, y_test_dt = transformation_pipeline_nores(dt)
#
#
# prediction_label_dt = pipeline.predict(X_test_dt)
# print('prediction_label_dt', prediction_label_dt)
#
#
# print(classification_report(y_test_dt, prediction_label_dt))
# print('Accuracy Score: ', accuracy_score(y_test_dt, prediction_label_dt))
# print('Recall Score: ', recall_score(y_test_dt, prediction_label_dt))
# print('\nConfusion matrix: \n', confusion_matrix(y_test_dt, prediction_label_dt))
# print('\n Model :', 'XGB')
