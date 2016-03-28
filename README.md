# Finance


Currenty this is the main repo for my finance scripts. The aim is to build this software bit by bit. Storing each section of the program in diffent repos until it is ready for the next phase of development.



* Machine Learning Playground is for testing Machine learning modles


#machine learning notes

###GBC:

[Parallel(n_jobs=-1)]: Done 1625 out of 1625 | elapsed: 233.0min finished

best score:  0.894301470588

best parameters {'n_estimators': 1000, 'learning_rate': 0.1, 'max_depth': 2}

accuracy_score: (train)  0.997242647059
accuracy_score: (test)  0.897058823529

3.8 seconds per fit

### xgb

{'n_estimators': 1000, 'learning_rate': 0.1, 'max_depth': 2}

accuracy_score: (train)  0.997242647059
accuracy_score: (test)  0.886029411765

train

             precision    recall  f1-score   support

        0.0       1.00      1.00      1.00        10
        1.0       1.00      1.00      1.00         4
        2.0       1.00      1.00      1.00       165
        3.0       1.00      1.00      1.00        22
        4.0       1.00      1.00      1.00        15
        5.0       1.00      1.00      1.00       108
        6.0       1.00      1.00      1.00         5
        7.0       1.00      1.00      1.00        73
        8.0       1.00      1.00      1.00        30
        9.0       1.00      1.00      1.00        22
       10.0       0.98      1.00      0.99       130
       11.0       1.00      0.84      0.91        19
       12.0       1.00      1.00      1.00        33
       13.0       1.00      1.00      1.00       113
       14.0       1.00      1.00      1.00        10
       15.0       1.00      1.00      1.00        31
       16.0       1.00      1.00      1.00         3
       18.0       1.00      1.00      1.00        31
       19.0       1.00      1.00      1.00        16
       20.0       1.00      1.00      1.00        57
       21.0       1.00      1.00      1.00        59
       22.0       1.00      1.00      1.00        10
       23.0       1.00      1.00      1.00        46
       24.0       1.00      1.00      1.00        45
       25.0       1.00      1.00      1.00        31

avg / total       1.00      1.00      1.00      1088


test

             precision    recall  f1-score   support

        0.0       1.00      0.67      0.80         3
        1.0       1.00      0.67      0.80         3
        2.0       0.95      0.97      0.96        40
        3.0       1.00      0.58      0.74        12
        4.0       1.00      0.67      0.80         3
        5.0       0.97      1.00      0.98        28
        6.0       1.00      0.50      0.67         2
        7.0       1.00      1.00      1.00        11
        8.0       0.83      1.00      0.91         5
        9.0       1.00      1.00      1.00         6
       10.0       0.94      1.00      0.97        34
       11.0       1.00      0.71      0.83         7
       12.0       0.71      0.71      0.71         7
       13.0       0.74      0.87      0.80        30
       15.0       0.38      1.00      0.55         3
       16.0       1.00      0.33      0.50         3
       17.0       0.00      0.00      0.00         1
       18.0       0.88      1.00      0.93         7
       19.0       1.00      0.86      0.92         7
       20.0       0.92      0.73      0.81        15
       21.0       0.73      1.00      0.84         8
       22.0       1.00      0.88      0.93         8
       23.0       0.82      0.82      0.82        17
       24.0       0.89      0.89      0.89         9
       25.0       1.00      1.00      1.00         3

avg / total       0.90      0.89      0.88       272







###extratrees

best score:  0.892463235294

best parameters {'n_estimators': 100, 'max_depth': 40}

accuracy_score: (train)  0.998161764706
accuracy_score: (test)  0.852941176471

[Parallel(n_jobs=-1)]: Done 350 out of 350 | elapsed:  3.9min finished


0.7 seconds per fit



###random forest
best score:  0.868566176471

best parameters {'n_estimators': 60, 'max_depth': 40}

accuracy_score: (train)  0.998161764706
accuracy_score: (test)  0.867647058824

[Parallel(n_jobs=-1)]: Done 350 out of 350 | elapsed:  6.1min finished


1.1 seconds per fit
