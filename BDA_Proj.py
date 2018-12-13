import csv
import re
from nltk.corpus import stopwords

def main():
    stars = []
    reviews = []

    with open('Mexican_Restaurant_Reviews.csv', 'r', encoding='utf8') as csvfile:
        data = csv.reader(csvfile)
        next(data)
        for row in data:
            stars.append(int(row[3]))
            reviews.append(row[4].lower())

    dict = {}
    for i in range(0, len(reviews)):
        for word in re.split('\s', reviews[i]):
            try:
                dict[word][0] += 1
            except KeyError:
                dict[word] =[1, stars[i]]

    dict = sorted(dict.items(), key=lambda x: x[1][0], reverse=True)

    print(dict)

main()
