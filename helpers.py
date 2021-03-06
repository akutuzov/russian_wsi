#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.manifold import TSNE
import pylab as plot


def fingerprint(text, model, weights=False):
    """
    :param text: list of words
    :param model: word2vec model in Gensim format
    :param weights: whether to use word weights based on frequencies (word2vec model should contain frequency data)
    :return: average vector of words in text
    """
    # Creating list of all words in the document, which are present in the model
    words = [w for w in text if w in model]
    lexicon = list(set(words))
    lw = len(lexicon)
    if lw < 1:
        print('Empty lexicon in', text, file=sys.stderr)
        return np.zeros(model.vector_size)
    vectors = np.zeros((lw, model.vector_size))  # Creating empty matrix of vectors for words
    for i in list(range(lw)):  # Iterate over words in the text
        word = lexicon[i]
        if weights:
            weight = wordweight(word, model)
        else:
            weight = 1.0
        vectors[i, :] = model[word] * weight  # Adding word and its vector to matrix
    semantic_fingerprint = np.sum(vectors, axis=0)  # Computing sum of all vectors in the document
    semantic_fingerprint = np.divide(semantic_fingerprint, lw)  # Computing average vector
    return semantic_fingerprint


def wordweight(word, model, a=10 ** -3, wcount=250000000):
    """
    :param word: word token
    :param model: word2vec model in Gensim format
    :param a: smoothing coefficient
    :param wcount: number of words in the training corpus (the default value corresponds to the RNC)
    :return: word weight (rare words get higher weights)
    """
    prob = model.wv.vocab[word].count / wcount
    weight = a / (a + prob)
    return weight


def save(df, corpus):
    """
    :param df: Data Frame with predictions
    :param corpus: dataset name
    :return: path to the saved file
    """
    output_fpath = corpus + "_predictions.csv"
    df.to_csv(output_fpath, sep="\t", encoding="utf-8", index=False)
    print("Generated dataset: {}".format(output_fpath))
    return output_fpath


def visualize(contexts, matrix, clusters, name, goldclusters=None):
    """
    :param contexts: point labels
    :param matrix: array of vectors for data points
    :param clusters: cluster labels
    :param name: the query word
    :param goldclusters: gold cluster labels
    :return: matrix projected to 2 dimensions
    """
    perplexity = int(len(contexts) ** 0.5)
    print(f"Perplexity: {perplexity}")
    if perplexity >= len(clusters):
        print('Error! Perplexity more than the number of points. Exiting...')
        exit()
    if goldclusters is None:
        goldclusters = [0] * len(clusters)
    embedding = TSNE(n_components=2, perplexity=perplexity, metric='cosine', n_iter=500, init='pca')
    y = embedding.fit_transform(matrix)

    xpositions = y[:, 0]
    ypositions = y[:, 1]
    plot.clf()
    colors = ['black', 'red', 'cyan', 'lime', 'brown', 'yellow', 'magenta', 'goldenrod', 'navy', 'purple', 'silver']
    markers = ['.', 'o', '*', '+', 'x', 'D']
    for context, x, y, cluster, goldcluster in zip(contexts, xpositions, ypositions, clusters, goldclusters):
        plot.scatter(x, y, 40, marker=markers[int(float(goldcluster))], color=colors[cluster])
        plot.annotate(context, xy=(x, y), size='x-small', color=colors[cluster])

    plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # plot.legend(loc='best')
    # plot.show()  # Uncomment if you want to show the plots immediately.
    plot.savefig(name + '.png', dpi=300, bbox_inches="tight")
    plot.close()
    plot.clf()
    return y
