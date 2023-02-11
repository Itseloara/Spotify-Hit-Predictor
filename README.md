# Spotify-hitpredictor
This project was designed as a machine learning exercise using the spotify "hit predictor" dataset.
Original dataset available [here](https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset).


<img src="https://github.com/samyakmohelay/Spotify-hit-predictor/blob/main/static/img/new1.jpg" width="650" height="400">

 
## Go to the final [Hit or Flop?](https://spotify-hit-predictor-app.herokuapp.com/) page!  


## Contents inside this repository
* Original data
* Models (& variations) tested
* Deployed program
  
  
## Project Scope
The dataset has features for tracks fetched using Spotify's Web API, base on the tracks labeled `hit` or `flop` by the author, which can be used to make a classification model to predict whether any given track would be a 'Hit' or not. With the original Kaggle dataset, I created my own machine learning model to analyze the songs, and then further analyzed the data for a retrospective analysis of features for songs from each decade (1960s - 2010s)..


## Metadata Summary
The original data, retrieved through Spotify's Web API (accesed though the Python library Spotipy), includes 40,000+ songs with release dates ranging from 1960-2019. Each track is identified by the track's name, the artist's name, and a unique resource identifier (uri) for the track.

The dataset includes measurements for the following features, defined and measured by Spotify for each track:

1. danceability
2. energy
3. key
4. loudness
5. mode
6. speechiness
7. acousticness
8. instrumentalness
9. liveness
10. valence
11. tempo
12. duration (in milliseconds)
13. time signature
14. target (a boolean variable describing if the track ever appeared on Billboard's Weekly Hot-100 list)

Two additional features were defined and extracted from the data recieved by the API call for Audio Analysis of each particular track:

15. chorus hit (an estimation of when the chorus of the track would first appear, i.e. "hit")
16. sections (number of sections inside the song)


Details about each of the features and what they measure can be found on [Kaggle](https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset), and through [Spotify's documentation](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/).

  
  
## Methodology
I trained and tested 4 different models to evaluate the dataset, in addition to variations within some of the models.
1. SVM (Support Vector Machine) Model
2. Logistic Regression
3. Neural Network/Deep Learning Model
4. Random Forest Model

> Note: The models were run in local server and the ones with the best results were run through Google's Collaboratory to maximize their score.

----
### 1) SVM with two normalization variations
I ran an SVM model with all 16 features, in two different functions for the normalization: one with Standard Scaler and the other with MinMax Scaler, both for x values and with the following results:

    
    from sklearn.svm import SVC 
    model_SVM = SVC(kernel='linear')
    model_SVM.fit(x_train_scaled, y_train)
    
* Standard Scaler:
``` 
Training Score: 0.7258101138538389
Testing Score: 0.7308553079692517
```
* MinMax Scaler:
```
Training Score: 0.7256479288981154
Testing Score: 0.7306606986474652
```
  
----
### 2) Logistic Regression with two normalization variations
Same as SVM, I ran the model with all 16 features and with the same normalization variations:

    from sklearn.linear_model import LogisticRegression
    model_LR = LogisticRegression()
    model_LR.fit(x_train_scaled, y_train)

* Standard Scaler:
```
Training Data Score: 0.7280158292516786
Testing Data Score: 0.7288119100904933
```
* MinMax Scaler:
```
Training Data Score: 0.7280482662428233
Testing Data Score: 0.7283253867860271
```

> With further analysis of the data, I decided to normalize each feature independently and ran the same models to see if the results improved. 


----
### 1) & 2) with independent normalization for features with values greater than 1 or with negative values.
The features included in this process were: 
* Loudness (also adjusted to positive values by adding 50)
* Duration (modified from microseconds to seconds)
* Key
* Tempo
* Time_signature
* Chorus_hit
* Sections

The results for each model were:
* SVM
```
Training Data Score: 0.765026436147783
Testing Data Score: 0.7634523693684927
```
* Logistic Regression
```
Training Data Score: 0.7280482662428233
Testing Data Score: 0.7286173007687068
```

> None of these adjustments produced improvements high enough where it seemed worth it to develop the models further. Therefore, I decided to test other types of models altogether.
  
----
### 3) Neural Network/Deep Learning
I originally designed the archetecture of the deep learning model to have 4 layers, while measuring the loss and accuracy levels during training and testing of the model. In the process of training and testing the model, I discovered that the addition of the final layer was essentially overtraining the model, leading to decreased accuracy. As such, I eliminated the last layer from the final version of the model. The best variation of the model was with 500 epochs, a batch size of 2000, and MinMax Scaler normalization.

    #Create a sequential model with 3 hidden layers
    from tensorflow.keras.models import Sequential
    model = Sequential() 

    from tensorflow.keras.layers import Dense
    number_inputs = 15
    number_classes = 2

    model.add(Dense(units=14,activation='relu', input_dim=number_inputs))
    model.add(Dense(units=120,activation='relu'))
    model.add(Dense(units=80,activation='relu'))
    model.add(Dense(units=number_classes, activation='softmax')) 

    #Compile the Model
    import tensorflow as tf
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    #Training the Model
    history = model.fit(X_train_scaled, y_train_categorical, epochs=500, batch_size=2000, shuffle=True, verbose=2)

```
Training Score: 0.7951928377151489
Testing Score: 0.7689014077186584
```
  
The initial test of the Neural Network, with all 16 features, performed a bit better than the SVM or Logistic Regression Models. For this reason, I decided to run several variations to see if eliminating any one of the features, or some features in combination, would improve the model's accuracy. The results can be seen in the table below.

|Features Missing|Training Score|Testing Score|
|:---:|:---:|:---:|
|Key|0.7846183776855469|0.773280143737793|
|Key & tempo|0.794771134853363|0.7775615453720093|
|Key, tempo & speechiness|0.7870511412620544|0.7753235101699829|
|Key, tempo, speechiness & chorus_hit|0.7847805619239807|0.7739612460136414|
|Tempo|0.7914950251579285|0.7711394429206848|
|Tempo & speechiness|0.7898731827735901|0.7698744535446167|
|Chorus_hit|0.7880566716194153|0.774739682674408|
|Chorus_hit & speechiness|0.7913652658462524|0.7705556154251099|
|Speechiness|0.784196674823761|0.7706528902053833|

Overall, these tests showed that eliminating any features did not improve accuracy and the best results came from including all 16 features.
  
----
### 4) Random Forest
In this case, the best variation of the model was with Standard Scaler normalization. With this model, I was able to see immediately that no one feature seemed to have a particularly strong weight or influence on the overall results:
|Importance|Feature|
|:---:|:---:|
|0.24652884971769662|instrumentalness|
|0.10647688218070937|danceability|
|0.10021253875816241|loudness|
|0.09385057415725137|acousticness|
|0.08341099244265467|duration_ms|
|0.08323171690224049|energy|
|0.06260221928220147|valence|
|0.046057546266831645|speechiness|
|0.03927362630575717|tempo|
|0.037828883345652195|liveness|
|0.037804710879875365|chorus_hit|
|0.027115992403401484|sections|
|0.023221514930213242|key|
|0.006420602303837413|mode|
|0.0059633501235151045|time_signature|

Individually, each feature was quite weak as a predictor of whether or not a song would be a hit. However, with all the features included, this was actually the best performing model.

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, max_depth=25) 
    model = model.fit(x_train_scaled, y_train)

```  
Training Data Score: 0.9993836971682507
Testing Data Score: 0.7883623625571665
```

I saw that the adjustments I was making resulted in only slight improvements or variations in the results of each model. This led me to believe that any real improvement to the results required taking a closer look at the data I was using to train the model. I theorized that, since music tastes change relatively less from one decade to the next but are much more pronouced over 30-40 years, perhaps limiting the data to a block of twenty years would improve the accuracy. I decided to use the songs from the 2000s since that is the most recent period of 20 years, and thus might more accurately predict what would be considered a hit today. With these adjustments, the accuracy of the model did in fact improve. The final adjustments made to the model, which maximized the results, were number of trees (200), and the max depth (25).

```  
Train score: 0.9967398391653989
Test score: 0.8464797913950456
```
  

## Results
After running the 4 models and the variations described above, I chose the Random Forest model with adjusted settings as the final model, given that it had produced the best results with the highest levels of accuracy.


## Deployment of the Model & Analysis of Historical "hit" data
I designed a public webpage for final publication of the machine learning model, deployed on [Heroku](https://spotify-hit-predictor-app.herokuapp.com/).

Visitors to the site can input a recently released song (or their favorite past song) to see how the model would determine whether it might be a "hit" or not. A second page provides interactives graphs to analyze the preferences in different song features across all decades in the full dataset (1960s-2010s), and provides a bit more background as to how each feature is defined.

With graph analysis, I noticed consistency in some audio features for hits throughout the decades, meaning that hits have higher quantities of these features and they seems to retain the same influence as to whether a song is popular.  These feature include: danceability, energy, loudness and valence. Other features have increasingly fluctuated in their importance from one decade to the next, characteristics like chorus hit, duration, liveness, mode, sections and speechiness.  That is to say that maybe in the 60's these features were more prominent in hits of that era, but in recent decades songs with higher levels of these features are more likely to be flops.


<img src="https://github.com/samyakmohelay/Spotify-hit-predictor/blob/main/static/img/new2.jpg" width="650" height="400">
