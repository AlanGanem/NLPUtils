# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:29:48 2020

@author: User Ambev
"""

import nltk
from nltk.collocations import *
import pandas as pd
from matplotlib import pyplot as plt
from random import shuffle


class BiTriGramsFinder():
    def __init__(self):
        return

    def fit(
            self,
            corpora,
            bi_threshold=7,
            tri_threshold=7,
            quad_threshold=7,
            bi_freq_filter=10,
            tri_freq_filter=10,
            quad_freq_filter=10,
            bi_metric='pmi',
            tri_metric='pmi',
            quad_metric='pmi',
            analyse=False,
            metric=None,
            threshold=None,
            freq_filter=None,
            **analyse_args

    ):
        # assert metric in [
        #     'chi_sq',
        #     'dice',
        #      'fisher',
        #      'jaccard',
        #      'likelihood_ratio',
        #      'mi_like',
        #      'phi_sq',
        #      'pmi',
        #      'poisson_stirling',
        #      'raw_freq',
        #      'student_t'
        # ]

        # set global ngram params
        if metric:
            bi_metric = metric
            tri_metric = metric
            quad_metric = metric
        if threshold:
            bi_threshold = threshold
            tri_threshold = threshold
            quad_threshold = threshold
        if freq_filter:
            bi_freq_filter = freq_filter
            tri_freq_filter = freq_filter
            quad_freq_filter = freq_filter

        self.thresholds = [bi_threshold, tri_threshold, quad_threshold]
        # self.metrics =
        # self.freq_filters =
        ######

        bigram_measures = nltk.collocations.BigramAssocMeasures()
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        quadgram_measures = nltk.collocations.QuadgramAssocMeasures()

        # change this to read in your data
        shuffle(corpora)
        titls_set = list(set(corpora))
        titls_set = [word for words in titls_set for word in words.split(' ')]
        while '' in titls_set:
            titls_set.remove('')
        # titls_set = ' '.join(titls_set)

        quadgram_finder = QuadgramCollocationFinder.from_words(titls_set)
        trigram_finder = TrigramCollocationFinder.from_words(titls_set)
        bigram_finder = BigramCollocationFinder.from_words(titls_set)

        # only n_grams that appear n+ times
        quadgram_finder.apply_freq_filter(quad_freq_filter)
        trigram_finder.apply_freq_filter(tri_freq_filter)
        bigram_finder.apply_freq_filter(bi_freq_filter)
        # return n_grams above threshold
        self.quadgrams = quadgrams = list(
            quadgram_finder.above_score(getattr(quadgram_measures, quad_metric), quad_threshold))
        self.trigrams = trigrams = list(
            trigram_finder.above_score(getattr(trigram_measures, tri_metric), tri_threshold))
        self.bigrams = bigrams = list(bigram_finder.above_score(getattr(bigram_measures, bi_metric), bi_threshold))

        self.quadgram_dict = {' '.join(i): '_'.join(i) for i in quadgrams}
        self.trigram_dict = {' '.join(i): '_'.join(i) for i in trigrams}
        self.bigram_dict = {' '.join(i): '_'.join(i) for i in bigrams}
        if analyse:
            self.fit_analysis(
                quadgram_finder=quadgram_finder,
                trigram_finder=trigram_finder,
                bigram_finder=bigram_finder,
                quad_metric=quad_metric,
                tri_metric=tri_metric,
                bi_metric=bi_metric,
                **analyse_args
            )

        return self

    def transform(self, corpora):

        new_corpora = []
        for title in tqdm.tqdm(corpora):
            for quad in self.quadgram_dict:
                if quad in title:
                    title = title.replace(quad, self.quadgram_dict[quad])
            for tri in self.trigram_dict:
                if tri in title:
                    title = title.replace(tri, self.trigram_dict[tri])
            for bi in self.bigram_dict:
                if bi in title:
                    title = title.replace(bi, self.bigram_dict[bi])
                    print(title)
            new_corpora.append(title)

        return new_corpora

    def fit_analysis(self, quadgram_finder, trigram_finder, bigram_finder, quad_metric, tri_metric, bi_metric,
                     **analyse_args):

        bigram_measures = nltk.collocations.BigramAssocMeasures()
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        quadgram_measures = nltk.collocations.QuadgramAssocMeasures()

        quadgram_scores = quadgram_finder.score_ngrams(getattr(quadgram_measures, quad_metric))
        trigram_scores = trigram_finder.score_ngrams(getattr(trigram_measures, tri_metric))
        bigram_scores = bigram_finder.score_ngrams(getattr(bigram_measures, bi_metric))

        quadgram_values = [round(quadgram_scores[i][1], 1) for i in range(len(quadgram_scores))]
        trigram_values = [round(trigram_scores[i][1], 1) for i in range(len(trigram_scores))]
        bigram_values = [round(bigram_scores[i][1], 1) for i in range(len(bigram_scores))]

        plt.clf()
        pd.Series(quadgram_values).plot(kind='hist', label='quadgram', color='orange', **analyse_args)
        pd.Series(trigram_values).plot(kind='hist', label='trigram', color='green', **analyse_args)
        pd.Series(bigram_values).plot(kind='hist', label='bigram', color='blue', **analyse_args)
        plt.xlabel('bigrams: {}\ntrigrams: {}\nquadgrams: {}'.format(bi_metric, tri_metric, quad_metric))

        plt.axvline(x=self.thresholds[0], color='blue')
        plt.axvline(x=self.thresholds[1], color='green')
        plt.axvline(x=self.thresholds[2], color='orange')

        plt.legend()

