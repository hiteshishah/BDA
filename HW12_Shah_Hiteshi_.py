"""
hw12_shah_hiteshi.py
author: Hiteshi Shah
date: 12/3/2017
description: To perform sequential analysis using bigrams and trigrams
"""

def main():

    # getting the corpus into a list of words from the txt file
    with open("words1.txt", encoding="utf-8") as file:
        words = file.read().split()

    # retaining the original list of words
    OGwords = words

    # adding '^' to the start of each word and '$' to the end of each word in the list
    words = ["^" + word + "$" for word in words]

    # creating dictionaries for all bigrams and trigrams and their respective counts
    bigrams = {}
    trigrams = {}
    for word in words:
        for i in range(1, len(word)):
            gram = (word[i - 1] + word[i]).lower()
            if gram in bigrams.keys():
                bigrams[gram] += 1
            else:
                bigrams[gram] = 1

        for j in range(2, len(word)):
            gram = (word[j - 2] + word[j - 1] + word[j]).lower()
            if gram in trigrams.keys():
                trigrams[gram] += 1
            else:
                trigrams[gram] = 1

    # creating dictionaries for different counts
    wordStart = {}
    wordEnd = {}

    vowelAfterT = {}

    letterAfterQ = {}

    for key, value in bigrams.items():
        if key[0] == "^":
            wordStart[key[1]] = value

        if key[1] == "$":
            wordEnd[key[0]] = value

        if key[0] == "t":
            if key[1] == "a":
                vowelAfterT['a'] = value
            if key[1] == "e":
                vowelAfterT['e'] = value
            if key[1] == "i":
                vowelAfterT['i'] = value
            if key[1] == "o":
                vowelAfterT['o'] = value
            if key[1] == "u":
                vowelAfterT['u'] = value

        if key[0] == "q":
            if key[1] != "u":
                letterAfterQ[key[1]] = value

    print("'nn' count: " + str(bigrams['nn']))
    print("'ss' count: " + str(bigrams['ss']) + "\n")

    print("'ll' count: " + str(bigrams['ll']))
    print("'cc' count: " + str(bigrams['cc']) + "\n")

    print("'tt' count: " + str(bigrams['tt']))
    print("'rr' count: " + str(bigrams['rr']) + "\n")

    print("Most common starting letter: " + max(wordStart, key=lambda k: wordStart[k]))
    print("Most common ending letter: " + max(wordEnd, key=lambda k: wordEnd[k]) + "\n")

    print("Most common vowel after 't': " + max(vowelAfterT, key=lambda k: vowelAfterT[k]) + "\n")

    print("'ie' count: " + str(bigrams['ie']))
    print("'ei' count: " + str(bigrams['ei']))
    print("'ly' count: " + str(bigrams['ly']))
    print("'li' count: " + str(bigrams['li']) + "\n")

    print("'er' ending words count: " + str(trigrams['er$']))
    print("'ed' ending words count: " + str(trigrams.get('ed$', 0)) + "\n")

    print("'ant' count: " + str(trigrams['ant']))
    print("'ent' count: " + str(trigrams['ent']) + "\n")

    print("Most common letter after 'q': " + max(letterAfterQ, key=lambda k: letterAfterQ[k]) + "\n")

    print("'tio' count: " + str(trigrams['tio']))
    print("'ion' count: " + str(trigrams['ion']) + "\n")

    # counting the number of trigrams that only occur once
    count = 0

    for key, value in trigrams.items():
        if not("^" in key or "$" in key):
            if value == 1:
                count += 1

    print("Number of trigrams occurring only once: " + str(count) + "\n")

    # calculating the chances for each of the words
    chances = chance(OGwords, bigrams)

    chances = sorted(chances.items(), key=lambda x: x[1])

    print("20 words with the lowest chance: ")
    for i in range(0, 20):
        print(chances[i])


def chance(words, bigrams):
    '''
    function to calculate the chances (probability of its occurrence) of the given list of words
    :param words: list of words
    :param bigrams: list of bigrams
    :return: a dictionary of words and their corresponding chances
    '''
    chances = {}

    for word in words:
        transitions = 0
        chance = 0
        if len(word) > 1:
            for i in range(1, len(word)):
                gram = (word[i - 1] + word[i]).lower()
                transitions += 1
                chance += bigrams[gram]
            chance /= sum(bigrams.values())
            chances[word] = chance / transitions

    return chances

main()