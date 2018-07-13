# Word Sense Induction for Russian
This is an implementation of Russian Word Sense induction system designed for the [RUSSE'18 shared task](http://russe.nlpub.org/2018/wsi/). This system [ranked 2nd](https://competitions.codalab.org/competitions/17810#results) among 19 participants for the **wiki-wiki** task. The [paper](https://arxiv.org/abs/1805.02258) describing it is accepted to the 24rd International Conference on Computational Linguistics and Intellectual Technologies (Dialogue’2018).

The approach is based on clustering averaged word vectors ("semantic fingerprints") of contexts for ambiguous words. Word vectors can be extracted from any word embedding model.

Input data must correspond to the RUSSE'18 data format, as exemplified by the files in the `RUSSE_data` subdirectory in this repository. 
Note that the words in the `word` and `context` columns of the input dataset should match entries in your word embedding model:
for example, regarding lemmatization and PoS-tags. 
We provide each RUSSE'18 dataset in 3 versions: raw text, lemmas, lemmas + PoS tags.

Word embedding models for Russian can be downloaded, for example, from [RusVectōrēs](http://rusvectores.org/models/) web service.

## Usage:
```
python3 wsi.py [-h] --input INPUT --model MODEL [--weights] [--2stage] [--test]
  -h, --help     show this help message and exit
  --input INPUT  Path to input file with contexts
  --model MODEL  Path to word2vec model
  --weights      Use word weights?
  --2stage       2-stage clustering?
  --test         Make predictions for test file with no gold labels?
```
For example, 

`python3 wsi.py --input RUSSE_data/wiki-wiki/train_tagged.csv --model ruscorpora_upos_skipgram_300_5_2018.vec.gz`

will use the word embedding model `ruscorpora_upos_skipgram_300_5_2018.vec.gz` to induce senses for all ambiguous words from `train_tagged.csv`, cluster the contexts according to these senses, and evaluate the resulting clustering against gold labels in the dataset. 
The new dataset containing cluster assignments will be saved to `train_tagged_predictions.csv`.

The `--test` argument is used when gold labels are not available (like in `test_tagged.csv` files). 
In this case, the script will make predictions, but not evaluation. Off by default.

The `--weights` argument will additionally weight words by their frequencies when creating semantic fingerprints of context utterances. 
We highly recommend this setting, as it greatly improves the performance. However, for this to work, you need a word embedding model saved in native Gensim format (to store word frequencies in the original training corpus). That's why it is off by default.

The `--2stage` argument tells the script to perform clustering in 2 stages: first, induce the number of clusters using *Affinity Propagation* and then do actual clustering with the induced number using *Spectral Clustering*. 
This increased performance for some datasets, but not all, therefore it is off by default.
