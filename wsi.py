#!/usr/bin/python3

from pandas import read_csv
from evaluate import evaluate
import argparse
import sys
import numpy as np
import gensim
import logging
from sklearn.manifold import TSNE
import pylab as plot
from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def visualize(contexts, matrix, clusters, name, goldclusters=None):
    perplexity = 10.0
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
        plot.scatter(x, y, 10, marker=markers[int(float(goldcluster))], color=colors[cluster])
        plot.annotate(context, xy=(x, y), size='x-small', color=colors[cluster])

    plot.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plot.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plot.legend(loc='best')
    # plot.show()  # Uncomment if you want to show the plots immediately.
    plot.savefig(name + '.png', dpi=300)
    plot.close()
    plot.clf()
    return y


def wordweight(word, model, a=10 ** -3, wcount=250000000):
    prob = model.wv.vocab[word].count / wcount
    weight = a / (a + prob)
    return weight


def fingerprint(text, model):
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
        weight = wordweight(word, model)
        vectors[i, :] = model[word] * weight  # Adding word and its vector to matrix
    semantic_fingerprint = np.sum(vectors, axis=0)  # Computing sum of all vectors in the document
    semantic_fingerprint = np.divide(semantic_fingerprint, lw)  # Computing average vector
    return semantic_fingerprint


def save(df, name, corpus):
    output_fpath = corpus + "_train.{}.csv".format(name)
    df.to_csv(output_fpath, sep="\t", encoding="utf-8")
    print("Generated {} baseline dataset: {}".format(name, output_fpath))
    return output_fpath


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', help='Path to input file with contexts')
    arg('--output', help='Path to output file with predictions')
    arg('--model', help='Path to word2vec model')
    arg('--test', dest='testing', action='store_true', help='Make predictions?')
    parser.set_defaults(testing=False)
    args = parser.parse_args()
    modelfile = args.model
    model = gensim.models.Word2Vec.load(modelfile)
    model.init_sims(replace=True)
    if args.testing:
        dataset = 'RUSSE_data/' + args.input + '/test_tagged.csv'
    else:
        dataset = 'RUSSE_data/' + args.input + '/train_tagged.csv'

    # This combination of the Affinity Propagation parameters was best in our experiments.
    # But in your task they can be different!
    damping = 0.7
    preference = -0.7

    df = read_csv(dataset, sep="\t", encoding="utf-8")
    predicted = []
    goldsenses = []
    for query in df.word.unique():
        print('Now analyzing', query, '...', file=sys.stderr)
        subset = df[df.word == query]
        if not args.testing:
            goldsenses.append(len(subset.gold_sense_id.unique()))
        contexts = []
        matrix = np.empty((subset.shape[0], model.vector_size))
        counter = 0
        lengths = []
        for line in subset.iterrows():
            con = line[1].tagged_context
            identifier = line[1].context_id
            label = query + str(identifier)
            contexts.append(label)
            if type(con) == float:
                print('Empty context at', label, file=sys.stderr)
                fp = np.zeros(model.vector_size)
            else:
                bow = con.split()
                bow = [w for w in bow if len(w.split('_')[0]) > 1]
                bow = [b for b in bow if b.split('_')[0] != query.split('_')[0]]
                fp = fingerprint(bow, model)
                lengths.append(len(bow))
            matrix[counter, :] = fp
            counter += 1
        clustering = AffinityPropagation(preference=preference, damping=damping).fit(matrix)
        cur_predicted = clustering.labels_.tolist()
        predicted += cur_predicted
        if not args.testing:
            gold = subset.gold_sense_id
            print('Gold clusters:', len(set(gold)), file=sys.stderr)
        print('Predicted clusters:', len(set(cur_predicted)), file=sys.stderr)
        if not args.testing:
            if len(set(gold)) < 6 and len(set(cur_predicted)) < 12:
                visualize(contexts, matrix, cur_predicted, query, gold)
            else:
                print('Too many clusters, not visualizing', file=sys.stderr)
        else:
            if len(set(cur_predicted)) < 12:
                visualize(contexts, matrix, cur_predicted, query)
    df.predict_sense_id = predicted
    if args.testing:
        save(df, "predict", args.input)
        exit()
    else:
        res = evaluate(save(df, "predict-train", args.input))
        print('ARI:', res)
        print('Average number of senses:', np.average(goldsenses))
        print('Variation of the number of senses:', np.std(goldsenses))
        print('Minimum number of senses:', np.min(goldsenses))
        print('Maximum number of senses:', np.max(goldsenses))


if __name__ == '__main__':
    main()
