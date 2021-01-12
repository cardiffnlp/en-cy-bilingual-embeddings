from nltk.cluster import KMeansClusterer
import scipy.stats
import math
import pandas as pd
import numpy as np
import nltk

class Evaluator:
    def __init__(self, model, language):
        self.model = model
        self.language = language
        self.analogies_data = '../data/evaluation_datasets/analogies_%s.txt' % self.language
        self.clustering_data = '../data/evaluation_datasets/AP_Classifiers_%s.csv' % self.language
        self.simlex999_data = '../data/evaluation_datasets/SimLex-999_%s.csv' % self.language
        self.wordsimilarity353_data = '../data/evaluation_datasets/word_pairs_%s.csv' % self.language
        self.synonyms_data = '../data/evaluation_datasets/synonyms_%s.csv' % self.language
        self.summary = {}

    def rand_index(self, distance_measure):
        words = self.clustering_evaluation_dataset['Word']
        full_words = self.clustering_evaluation_dataset.set_index('Word')
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for word1 in words:
            for word2 in words:
                if word1 != word2:
                    if full_words.loc[word1, 'Cat'] == full_words.loc[word2, 'Cat']:
                        if full_words.loc[word1, distance_measure] == full_words.loc[word2, distance_measure]:
                            true_positive += 1
                        else:
                            false_negative += 1
                    else:
                        if full_words.loc[word1, distance_measure] == full_words.loc[word2, distance_measure]:
                            false_positive += 1
                        else:
                            true_negative += 1
        rand_index = (true_positive + true_negative) / sum([true_positive, false_positive, true_negative, false_negative])
        return rand_index

    def entropy(self, distance_measure):
        cats = list(set(self.clustering_evaluation_dataset['Cat']))
        ncats = len(cats)
        total_entropy = 0
        n = len(self.clustering_evaluation_dataset)
        for cluster in range(ncats):
            n_r = self.clustering_evaluation_dataset[distance_measure].value_counts()[cluster]
            E = 0
            for category in cats:
                n_ir = self.clustering_evaluation_dataset.groupby('Cat')[distance_measure].value_counts()[category].get(cluster, 0)
                if n_ir != 0:
                    E += ((n_ir / n_r) * math.log(n_ir / n_r))
            E *= math.log(ncats)
            total_entropy -= ((n_r / n) * E)
        return total_entropy

    def purity(self, distance_measure, ncats):
        return sum([self.clustering_evaluation_dataset[self.clustering_evaluation_dataset[distance_measure] == i]['Cat'].value_counts().max() for i in range(ncats)]) / len(self.clustering_evaluation_dataset)


    def evaluate_analogies(self):
        self.analogy_scores = self.model.wv.evaluate_word_analogies(self.analogies_data)
        self.summary['Analogy Score'] = self.analogy_scores[0]

    def evaluate_clustering(self):
        data = pd.read_csv(self.clustering_data, names=['Word', 'Cat'])
        full = pd.concat([data, pd.DataFrame({'v' + str(i): [self.model.wv[word][i] for word in data['Word']] for i in range(self.model.vector_size)})], axis=1)
        ncats = len(set(full['Cat']))
        kclusterer = KMeansClusterer(ncats, distance=nltk.cluster.util.euclidean_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(np.array(full[['v' + str(i) for i in range(self.model.vector_size)]]), assign_clusters=True)
        data['Euclidean'] = assigned_clusters
        kclusterer = KMeansClusterer(ncats, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(np.array(full[['v' + str(i) for i in range(self.model.vector_size)]]), assign_clusters=True)
        data['Cosine'] = assigned_clusters
        self.clustering_evaluation_dataset = data
        self.summary['Clustering Purity (Euclidean)'] = self.purity('Euclidean', ncats)
        self.summary['Clustering Purity (Cosine)'] = self.purity('Cosine', ncats)
        self.summary['Clustering Entropy (Euclidean)'] = self.entropy('Euclidean')
        self.summary['Clustering Entropy (Cosine)'] = self.entropy('Cosine')
        self.summary['Clustering Rand Index (Euclidean)'] = self.rand_index('Euclidean')
        self.summary['Clustering Rand Index (Cosine)'] = self.rand_index('Cosine')

    def evaluate_simlex999(self):
        self.simlex999_evaluation_dataset = pd.read_csv(self.simlex999_data, names=['Word 1', 'Word 2', 'POS', 'SimLex999', 'conc(w1)', 'conc(w2)', 'concQ', 'Assoc(USF)', 'SimAssoc333', 'SD(SimLex)'], skiprows=1)
        self.simlex999_evaluation_dataset['Model'] = self.simlex999_evaluation_dataset.apply(lambda row: self.model.wv.n_similarity(row['Word 1'], row['Word 2']), axis=1)
        self.summary['SimLex999 Correlation (SimLex999)'] = scipy.stats.spearmanr(self.simlex999_evaluation_dataset['SimLex999'], self.simlex999_evaluation_dataset['Model'], nan_policy='omit').correlation
        self.summary['SimLex999 Correlation (Assoc(USF))'] = scipy.stats.spearmanr(self.simlex999_evaluation_dataset['Assoc(USF)'], self.simlex999_evaluation_dataset['Model'], nan_policy='omit').correlation

    def evaluate_wordsimilarity353(self):
        self.wordsimilarity353_evaluation_dataset = pd.read_csv(self.wordsimilarity353_data, names=['Word 1', 'Word 2', 'Human Score'], skiprows=1)
        self.wordsimilarity353_evaluation_dataset['Model'] = self.wordsimilarity353_evaluation_dataset.apply(lambda row: self.model.wv.n_similarity(row['Word 1'], row['Word 2']), axis=1)
        self.summary['WordSimilarity353 Correlation'] = scipy.stats.spearmanr(self.wordsimilarity353_evaluation_dataset['Human Score'], self.wordsimilarity353_evaluation_dataset['Model'], nan_policy='omit').correlation

    def evaluate_synonyms(self):
        self.synonyms_evaluation_dataset = pd.read_csv(self.synonyms_data, names=['Word', 'Synonym', 'Other 1', 'Other 2', 'Other 3'])
        synonym_similarities = []
        other1_similarities = []
        other2_similarities = []
        other3_similarities = []
        for _, row in self.synonyms_evaluation_dataset.iterrows():
            synonym_similarities.append(self.model.wv.n_similarity(row['Word'], row['Synonym']))
            other1_similarities.append(self.model.wv.n_similarity(row['Word'], row['Other 1']))
            other2_similarities.append(self.model.wv.n_similarity(row['Word'], row['Other 2']))
            other3_similarities.append(self.model.wv.n_similarity(row['Word'], row['Other 3']))
        self.synonyms_evaluation_dataset['Synonym Similarity'] = synonym_similarities
        self.synonyms_evaluation_dataset['Other 1 Similarity'] = other1_similarities
        self.synonyms_evaluation_dataset['Other 2 Similarity'] = other2_similarities
        self.synonyms_evaluation_dataset['Other 3 Similarity'] = other3_similarities
        self.synonyms_evaluation_dataset['Max'] = self.synonyms_evaluation_dataset[['Synonym Similarity', 'Other 1 Similarity', 'Other 3 Similarity', 'Other 3 Similarity']].transpose().max()
        self.synonyms_evaluation_dataset['Correct'] = self.synonyms_evaluation_dataset['Max'] == self.synonyms_evaluation_dataset['Synonym Similarity']
        self.summary['Synonuyms Percentage Correct'] = self.synonyms_evaluation_dataset['Correct'].mean()

