# Learning English-Welsh bilingual embeddings and applications in text categorisation

This project focuses on creating cross-lingual representations of words in a joint embedding space for Welsh and English. This project is funded by the Welsh Government. This repository provides three main resources:

- Monolingual and bilingual (English and Welsh) **corpora** and **pre-trained models** (bilingual embeddings and a sentiment analysis system for Welsh).
- **Code** to train custom word embedding models
- **Code** to train custom text classification system for Welsh

# Corpora and pre-trained models

We list below English and Welsh corpora that we were used for training our bilingual embeddings.

## Welsh Corpora

(Note: full dataset = 144,976,542 words after tokenisation)

Sources:

+ wiki_cy.tx
  + SOURCE: https://cy.wikipedia.org/wiki/Hafan
  + DESCRIPTION: All Welsh language wikipedia pages.
  + EXTRACTION: Scraped using `urllib` and `beautifulsoup` in Python, obtaining all text within a `<p>` tag. Citations and mathematics removed.
  + NUMBER OF WORDS: 21,233,177

+ uagtpnaw_cy.txt
  + SOURCE: http://xixona.dlsi.ua.es/corpora/UAGT-PNAW/
  + CITE: Jones, D. and Eisele, A., 2006. Phrase-based statistical machine translation between English and Welsh. In Proceedings of the 5th SALTMIL Workshop on Minority Languages at the 5th International Conference on Language Resources and Evaluation (pp. 75-77).
  + DESCRIPTION: Proceeedings of the Welsh Assembly 1999-2006
  + EXTRACTION: Available as plain text file.
  + NUMBER OF WORDS: 11,527,963

+ kynulliad3_cy.txt
  + SOURCE: http://cymraeg.org.uk/kynulliad3/
  + CITE: Kevin Donnelly (2013). "Kynulliad3: a corpus of 350,000 aligned Welsh and English sentences from the Third Assembly (2007-2011) of the National Assembly for Wales." http://cymraeg.org.uk/kynulliad3.
  + DESCRIPTION: Proceedings of the Welsh Assembly 2007 - 2011
  + EXTRACTION: Queried SQL database.
  + NUMBER OF WORDS: 8,884,870
  
+ bible_cy.txt
  + SOURCE: http://www.beibl.net/
  + DESCRIPTION: All books of the bible in modern Welsh.
  + EXTRACTION: Scraped using `urllib` and `beautifulsoup` in Python, obtaining all text within a `<p>` tag.
  + NUMBER OF WORDS: 749,573

+ cOrPUS_cy.txt
  + SOURCE: http://opus.nlpl.eu/
  + DESCRIPTION: OPUS is a collection of translated texts on the web. For en-cy we have the folling corpera: Tatoeba v20190709, GNOME v1, JW300 v1, KDE4 v2, QED v2.0, Ubuntu v14.10, EUbookshop v2.
  + EXTRACTION: Available as plain text files.
  + NUMBER OF WORDS: 1,224,956

+ translation_memories_cy.txt
  + SOURCE: https://gov.wales/bydtermcymru/translation-memories
  + DESCRIPTION: This collection is of translation memory files containing published bilingual versions of documents and other materials by the Welsh Government (August 2019 - May 2020).
  + EXTRACTION: .tmx files extracted with Python's `translate-toolkit` package.
  + NUMBER OF WORDS: 1,857,267

+ proceedings_cy.txt
  + SOURCE: https://record.senedd.wales/XMLExport
  + DESCRIPTION: Record of proceedsings of the Welsh Assembly. Plenary information from the start of the Fifth Assembly (May 2016) and Committee information from November 2017 onwards.
  + EXTRACTION: Download .xml files, extract text with `beautifulsoup` in Python.
  + NUMBER OF WORDS: 17,177,715

+ cronfa-electronig-cymraeg.txt
  + SOURCE: https://www.bangor.ac.uk/canolfanbedwyr/ceg.php.en
  + CITE: Ellis, N. C., O'Dochartaigh, C., Hicks, W., Morgan, M., & Laporte, N.  (2001). Cronfa Electroneg o Gymraeg (CEG): A 1 million word lexical database and frequency count for Welsh
  + DESCRIPTION: Curated collection of Welsh texts including novels, short stories, religious writings and administrative documents.
  + EXTRACTION: Available as ASCII files. Special characters then encoded to unicode.
  + NUMBER OF WORDS: 1,046,800

+ ancrubadan.txt
  + SOURCE: http://crubadan.org/ (Emailed Kevin Scannell for original data)
  + CITE: Scannell, K.P., 2007, September. The Crúbadán Project: Corpus building for under-resourced languages. In Building and Exploring Web Corpora: Proceedings of the 3rd Web as Corpus Workshop (Vol. 4, pp. 5-15).
  + DESCRIPTION: Crawl of many websites, blog posts and twitter from the An Crúbadán project.
  + EXTRACTION: Plain text files provided for each source. Html table provided including meta-data all data sources. Removed all wikipedia data (as we have more up to date data in another data set).
  + NUMBER OF WORDS: 22,572,066

+ deche.txt
  + SOURCE: https://llyfrgell.porth.ac.uk/Default.aspx?search=deche&page=1&fp=0
  + CITE: Prys, D., Jones, D. and Roberts, M., 2014, August. DECHE and the Welsh National Corpus Portal. In Proceedings of the First Celtic Language Technology Workshop (pp. 71-75).
  + DESCRIPTION: The Digitization, E-publishing and Electronic Corpus DEChE. Digitised textbooks publically available.
  + EXTRACTION: Manually downloaded all books in .epub format. Converted to plain text using `epub_conversion.utils` and `beautifulsoup` in Python.
  + NUMBER OF WORDS: 2,126,153

+ corpuscrawler.txt
  + SOURCE: https://github.com/google/corpuscrawler
  + DESCRIPTION: A corpus crawler built by google for a number of languages. The Welsh language crawler extracts the Universal Declaration of Human Rights, and all articles from BBC Cymru Fyw from 2011 until now (17/10/2019).
  + EXTRACTION: Crawler built in Python. Crawled text is processed to remove metadata extracted by the crawler.
  + NUMBER OF WORDS: 14,791,835

+ gwerddon.txt
  + SOURCE: http://www.gwerddon.cymru/cy/hafan/
  + DESCRIPTION: All (29) editions of the Welsh language academic journal Gwerddon.
  + EXTRACTION: Downloaded and extracted using R and the `pdftools` package. Some manual post-formitting done to correct for footnotes and so on.
  + NUMBER OF WORDS: 767,677

+ wefannau.txt
  + SOURCE: https://golwg360.cymru/, https://pedwargwynt.cymru/, https://barn.cymru/, https://poblcaerdydd.com/
  + DESCRIPTION: All articles from the news websites Golwg360 and PoblCaerdydd, and all free articles from O'r Pedwar Gwynt and Barn magazines.
  + EXTRACTION: Web crawled using `wget`. Then text extracted form html with `beautifulsoup` in Python, obtaining all text within a `<p>` tag.
  + NUMBER OF WORDS: 7,388,917

+ corcencc_full.txt
  + SOURCE: http://www.corcencc.org/ (Recieved original data privately - using an early version of the corpus) Full copy of CorCenCC v1.0 is available from: https://research.cardiff.ac.uk/converis/portal/detail/Dataset/119878310?auxfun=&lang=en_GB
  + DESCRIPTION: The full CorCenCC in preprocessed form.
  + CITE: Knight, D., Morris, S., Fitzpatrick, T., Rayson, P., Spasić, I., Thomas, E-M., Lovell, A., Morris, J., Evas, J., Stonelake, M., Arman, L., Davies, J., Ezeani, I., Neale, S., Needs, J., Piao, S., Rees, M., Watkins, G., Williams, L., Muralidaran, V., Tovey-Walsh, B., Anthony, L., Cobb, T., Deuchar, M., Donnelly, K., McCarthy, M. and Scannell, K. (2020). CorCenCC: Corpws Cenedlaethol Cymraeg Cyfoes – the National Corpus of Contemporary Welsh.Cardiff University. http://doi.org/10.17035/d.2020.0119878310
  + EXTRACTION: Identified html and extracted all text in `<p>` tags using `beatifulsoup` in Python. All non html files left as is.
  + NUMBER OF WORDS: 10,630,657

+ s4c.txt
  + SOURCE: Received original data privately (i.e. not publicly available) 
  + DESCRIPTION: Video Text Track (.vtt) files of recent bilingual subtitles of S4C progammes.
  + EXTRACTION: Text manipulation to strip away formatting.
  + NUMBER OF WORDS: 26,931,013

## English Corpora 

+ UMBC
  + SOURCE: http://ebiquity.umbc.edu/blogger/2013/05/01/umbc-webbase-corpus-of-3b-english-words
  + DESCRIPTION: Data from the web, which includes blog posts, news stories, wikipedia, etc.
  + EXTRACTION: Cleand, tokenized and pre-processed.
  + VERSIONS: [Only tokenized](https://drive.google.com/file/d/1NIx5lbqg1_PYb53XLkCrnh_4hXQyr9bN/view?usp=sharing); [only tokenized and chunked (no POS tags)](https://drive.google.com/file/d/191S3GjlkNsLge3aPQdO7iKsCFRjLiP3s/view?usp=sharing);  [tokenized, tagged and chunked](https://drive.google.com/file/d/1cLXr0iyY-UiSJfxrkvVgE2URmIYV7yfD/view?usp=sharing).
  
+ WIKIPEDIA
  + Source: www.wikipedia.org
  + DESCRIPTION: Encyclopedia.
  + EXTRACTION: Removed wiki markup and pre-processed.
  + VERSIONS: [One sentence per line, tokenized, lemmatized, chunked, lower-cased and pos-tagged](https://drive.google.com/file/d/1jw7ly2IIY4UowgaBXslozYTa5kBBx9UC/view?usp=sharing) (`the_D centreville_N amusement_park_N or_C centreville_N theme_park_N be_V a_D child_N 's_P amusement_N ...`).

## Other resources relevant to the project

- **Translation service used for sentiment analysis**.
  - [Google translate api](https://pypi.org/project/googletrans/)
- **Bilingual English - Welsh dictionary**.
  - [Dictionary](https://github.com/cardiffnlp/en-cy-bilingual-embeddings/blob/master/data/resources/dictionaries/dictionary.tsv)
  - Copyright 2016 Prifysgol Bangor University. Licensed under the Apache License, Version 2.0.

## Pre-trained models

Here we make available the embeddings produced in the project. All embeddings are trained using two algorithms: `word2vec` (CBOW variant) and `fasttext`, and for each, we release a range of models for various vector sizes, and minimum frequency and context window values. Unless otherwise specified, we apply the supervised variant of `VecMap` for the bilingual mappings. We also provide all the required files to perform Welsh sentiment analysis:

- [Cross-lingual English-Welsh embeddings] (https://drive.google.com/drive/folders/1iGqzFlZifSeHzPhnz3qM68a37Ul7S7Xq?usp=sharing)
- [Cross-lingual English-Welsh sentiment analysis files] (https://drive.google.com/drive/folders/1VArQ4_bTzz8IJj8h7o969UrI12l1Cl9z?usp=sharing) - See below for customization and training details.
_____________________

Below we describe how to train your own bilingual (English and Welsh) embeddings and sentiment analysis models.

# Training your own bilingual word embeddings

The first step is to train monolingual word embeddings. For this, you need a pre-processed corpus, in one-sentence-per-line format, which is the format that `train_embeddings.py` expects.

To train monolingual embeddings, launch the following command:

```bash
python3 -i src/train_embeddings.py --corpus YOUR-CORPUS --model YOUR-MODEL --output-directory YOUR-OUTPUT-DIR
```
where `YOUR-MODEL` can be either `fasttext` or `word2vec`. The resulting vectors will be saved in the specified output directory with the suffix `_vectors.vec`.

## Create training and test data from a bilingual dictionary

In order to train a bilingual mapping, you need a _bilingual dictionary_. However, a first step is to ensure that, for any dictionary, there is sufficient overlap between the bilingual dictionary entries and the vocabulary in the word embeddings trained in the previous step. We provide a set of scripts to collect word embedding vocabularies, first, and then create a training/test split from the dictionary with the filtered vocabulary.

### Retrieve vocabularies from monolingual word embeddings

Launch this command:

```bash
python3 src/get_vocab_from_vectors.py --dict YOUR-DICT --source-vectors-folder FOLDER-WITH-SOURCE-EMBEDDINGS --target-vectors-folder FOLDER-WITH-TARGET-EMBEDDINGS --output-folder OUTPUT-FOLDER
```
This script scans `FOLDER-WITH-SOURCE-EMBEDDINGS` and `FOLDER-WITH-TARGET-EMBEDDINGS` and produces two vocabulary files `source_vocab.txt` and `target_vocab.txt` and stores them in `OUTPUT-FOLDER`.

### Split dictionary into train/test

To create your own train and test dictionaries, you need source and target vocabularies as well as a bilingual dictionary. Run the following command.

```bash
python3 -i src/split-dictionary.py --dict DICTIONARY --source-vocab SOURCE-VOCAB-FILE --target-vocab TARGET-VOCAB-FILE --output-folder OUTPUT-DIR
```

This script will save two files: `train_dict.csv` and `test_dict.csv` into the `OUTPUT-DIR` folder.

# Experiment 1 - Bilingual English - Welsh embeddings

Detailed instructions available in the [vecmap repo](https://github.com/artetxem/vecmap). 

## Launch batch VecMap mappings

For convenience, we provide a method for evaluating batches of English-Welsh vectors where we evaluate two embeddings trained with the same hyperparameters (see file naming requirements for properly aligning embedding pairs).

The below command will first scan `MAPPED-ENG-FOLDER` and `MAPPED-WEL-FOLDER` for embeddings of the same config (model name - word2vec or fasttext -, and the `mc`, `s` and `w` parameters), then apply the supervised variant of VecMap using the `--traindict` dictionary as supervision.

```bash
python3 src/vecmap_launcher.py --traindict data/resources/dictionaries/train_dict_freqsplit.csv --source-vectors-folder MAPPED-ENG-FOLDER --target-vectors-folder MAPPED-WEL-FOLDER
```
This step creates a bilingual mapping for each of the input models with the suffix `_mapped.vec`.

## Launch batch dictionary induction evaluation

The below command will first scan the folders `MAPPED-ENG-FOLDER` and `MAPPED-WEL-FOLDER` for _mapped_ embeddings of the same config (model name, and `mc`, `s` and `w` parameters), then evaluate them on the `TEST-DICT` dictionary. It will save the results for different retrieval methods into the `RESULTS-FOLDER` directory.

```bash
python3 src/vecmap_eval_launcher.py --testdict TEST-DICT --source-vectors-folder MAPPED-ENG-FOLDER --target-vectors-folder MAPPED-WEL-FOLDER --results-folder RESULTS-FOLDER
```
Note that there are several methods for retrieving similar words in bilingual spaces. Please refer to the original VecMap codes for all the variants. Here we use as default `nearest neighbours by cosine similarity`, which is the fastest method, although `csls` works slighty better.

### Results

| coverage | accuracy | model    | size | mc | window | retrieval |
|----------|----------|----------|------|----|--------|-----------|
| 93.93    | 15.43    | word2vec | 500  | 6  | 4      | nn        |
| 93.93    | 15.1     | word2vec | 300  | 6  | 6      | nn        |
| 93.93    | 15       | word2vec | 500  | 6  | 6      | nn        |
| 93.93    | 14.98    | word2vec | 500  | 6  | 8      | nn        |
| 93.93    | 14.85    | word2vec | 300  | 6  | 4      | nn        |
| 93.93    | 14.44    | word2vec | 300  | 6  | 8      | nn        |
| 100      | 14.06    | word2vec | 500  | 3  | 4      | nn        |
| 100      | 14.06    | word2vec | 300  | 3  | 6      | nn        |
| 100      | 13.98    | word2vec | 500  | 3  | 6      | nn        |
| 100      | 13.98    | word2vec | 500  | 3  | 8      | nn        |
| 100      | 13.79    | word2vec | 300  | 3  | 4      | nn        |
| 100      | 13.67    | word2vec | 300  | 3  | 8      | nn        |
| 93.93    | 11.98    | fasttext | 500  | 6  | 4      | nn        |
| 93.93    | 11.82    | fasttext | 500  | 6  | 6      | nn        |
| 93.93    | 11.59    | fasttext | 500  | 6  | 8      | nn        |
| 93.93    | 11.14    | word2vec | 100  | 6  | 4      | nn        |
| 93.93    | 10.98    | fasttext | 300  | 6  | 6      | nn        |
| 93.93    | 10.98    | word2vec | 100  | 6  | 6      | nn        |
| 93.93    | 10.91    | word2vec | 100  | 6  | 8      | nn        |
| 93.93    | 10.76    | fasttext | 300  | 6  | 4      | nn        |
| 100      | 10.68    | fasttext | 500  | 3  | 4      | nn        |
| 93.93    | 10.5     | fasttext | 300  | 6  | 8      | nn        |
| 100      | 10.31    | fasttext | 500  | 3  | 8      | nn        |
| 100      | 10.28    | word2vec | 100  | 3  | 8      | nn        |
| 100      | 10.28    | fasttext | 500  | 3  | 6      | nn        |
| 100      | 10.22    | word2vec | 100  | 3  | 6      | nn        |
| 100      | 9.66     | word2vec | 100  | 3  | 4      | nn        |
| 100      | 9.6      | fasttext | 300  | 3  | 4      | nn        |
| 100      | 9.46     | fasttext | 300  | 3  | 6      | nn        |
| 100      | 9.38     | fasttext | 300  | 3  | 8      | nn        |
| 93.93    | 6.53     | fasttext | 100  | 6  | 6      | nn        |
| 93.93    | 6.51     | fasttext | 100  | 6  | 4      | nn        |
| 93.93    | 6.44     | fasttext | 100  | 6  | 8      | nn        |
| 100      | 5.43     | fasttext | 100  | 3  | 6      | nn        |
| 100      | 5.29     | fasttext | 100  | 3  | 8      | nn        |
| 100      | 5.26     | fasttext | 100  | 3  | 4      | nn        |

# Experiment 2: Cross-lingual sentiment analysis

In this section we explain the steps to be taken for training and evaluating a Welsh sentiment analysis system trained only on English data. On an automatically generated test set, we obtain a **63% Accuracy with a zero-shot model** (i.e., a model evaluated on Welsh data but only trained on English). Note that these numbers can increase with proper hyperparameter tuning of our neural network. The reported results are based on the following pipeline:

- Since we were not able to find a sentiment analysis dataset and pre-trained model for the Welsh language, we propose to train a cross-lingual model on automatically translated English data. 
- We used the [IMDB reviews sentiment dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), which contains 50,000 positive/negative movie reviews (25k for training, 25k for evaluation).
- We used the Google translate API for generating **IMDB-CY** (the Welsh verison fo IMDB reviews), which can be accessed [here](https://github.com/luisespinosaanke/wel-eng-embeddings/tree/master/data/corpora/welsh/imdb-cy).
- We release code for training a sentiment analysis model based on a stack of Convolutional Neural Networks and Long Short-Term Memory Networks (adapted from [here](https://www.aclweb.org/anthology/N18-2061/)). This model, when trained with our cross-lingual embeddings, can then be applied indistinctly to English and Welsh data.

## Classifier training

This script produces three files: a classifier, a tokenizer and a file with a id-to-label mapping (e.g., 1=positive and 0=negative). 

```bash
python3 src/train_crosslingual_classifier.py --en-word-vectors ENGLISH-MAPPED-VECTORS --wel-word-vectors WELSH-MAPPED-VECTORS --dataset data/corpora/english/imdb/train/ --output-model PATH-TO-CLASSIFIER --output-tokenizer PATH-TO-TOKENIZER --output-labelmap PATH-TO-LABEL-MAP
```

## Evaluate crosslingual classifier

This script will print to screen the classification report on the provided `TEST-DATASET`.

```bash
python3 src/test_crosslingual_classifier.py --input-model PATH-TO-CLASSIFIER --input-tokenizer PATH-TO-TOKENIZER --input-labelmap PATH-TO-LABEL-MAP --dataset TEST-DATASET
```
