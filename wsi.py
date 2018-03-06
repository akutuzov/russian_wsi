#!/usr/bin/python3

from os import path
from pandas import read_csv
from evaluate import evaluate
import argparse
import sys
import numpy as np
import gensim
import logging
from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from helpers import visualize, fingerprint, save

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
    dataset = args.input

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
        if args.testing:
            if len(set(cur_predicted)) < 12:
                visualize(contexts, matrix, cur_predicted, query)
        else:
            if len(set(gold)) < 6 and len(set(cur_predicted)) < 12:
                visualize(contexts, matrix, cur_predicted, query, gold)
            else:
                print('Too many clusters, not visualizing', file=sys.stderr)

    df.predict_sense_id = predicted
    fname = path.splitext(path.basename(args.input))[0]
    if args.testing:
        save(df, fname)
    else:
        res = evaluate(save(df, fname))
        print('ARI:', res)
        print('Average number of senses:', np.average(goldsenses))
        print('Variation of the number of senses:', np.std(goldsenses))
        print('Minimum number of senses:', np.min(goldsenses))
        print('Maximum number of senses:', np.max(goldsenses))


if __name__ == '__main__':
    main()
