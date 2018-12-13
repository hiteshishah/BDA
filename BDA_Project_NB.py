import csv
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

def main():
    stars = []
    reviews = []

    with open('Mexican_Restaurant_Reviews.csv', 'r', encoding='utf8') as csvfile:
        data = csv.reader(csvfile)
        next(data)
        for row in data:
            stars.append(int(row[3]))
            reviews.append(re.sub('[^A-Za-z0-9]+', ' ', row[4]))

    new_stars = []

    for star in stars:
        if star == 1 or star == 2:
            new_stars.append("neg")
        elif star == 3:
            new_stars.append("neutral")
        else:
            new_stars.append("pos")

    reviews_train, reviews_test, stars_train, stars_test = train_test_split(reviews, new_stars, test_size=0.3,
                                                                            random_state=42)

    vectorizer = CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))

    train_features = vectorizer.fit_transform(reviews_train)

    test_features = vectorizer.transform(reviews_test)

    # Naive
    nb = SGDClassifier()
    nb.fit(train_features, stars_train)
    predictions_naive = nb.predict(test_features)

    print(classification_report(stars_test, predictions_naive))

    # SVM
    # svm = SGDClassifier()
    # svm.fit_transform(train_tfidf, stars_train)
    # predictions_svm = svm.predict(test_features)
    # countFor5ClassPrediction = 0
    # countFor3ClassPrediction = 0
    # n = len(predictions_svm)
    # for i in range(0, n):
    #     if predictions_svm[i] == stars_test[i]:
    #         countFor5ClassPrediction += 1
    #     if review_type[predictions_svm[i]] == review_type[stars_test[i]]:
    #         # print(predictions_naive[i], stars_test[i])
    #         countFor3ClassPrediction += 1
    #
    # print("Accuracy obtained for 5 class(Star rating from 1-5) prediction using SVM: ",
    #       countFor5ClassPrediction / n * 100)
    # print("Accuracy obtained for 3 class(Positive, Neutral, Negative) prediction using SVM: ",
    #       countFor3ClassPrediction / n * 100)
    #
    # # GRID SEARCH
    # # parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
    # # nb_classification = Pipeline(
    # #     [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])
    # # nb_classification = nb_classification.fit(reviews_train, stars_train)
    # # grid_Naive = GridSearchCV(nb_classification, parameters, n_jobs=-1)
    # # grid_Naive = grid_Naive.fit(reviews_train, stars_train)
    # # print(grid_Naive.best_score_)
    # # print(grid_Naive.best_params_)


main()
