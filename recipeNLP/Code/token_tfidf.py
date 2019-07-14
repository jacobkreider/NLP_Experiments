import math
import os
import pandas as pd
from pathlib import Path


token_folder = Path('/home/jacob/DataScience/NLP_Experiments/recipeNLP/Token Sheets/')
os.chdir('/home/jacob/DataScience/NLP_Experiments/recipeNLP/Token Sheets/')

# Read in data from tokenization exercise
tf_corpus = pd.read_csv('equiv_data_agg.csv')
equiv_class_docs = pd.read_csv('equiv_class_doc_counts.csv')
equiv_class_source = pd.read_csv('token_data_pivot_ec.csv')
equiv_class_source = equiv_class_source[['Term', 'equivalence_class']]

# Add IDF column to above dataframe
doc_count = 24.0
equiv_class_docs['IDF'] = (doc_count/(equiv_class_docs['total_docs']))
equiv_class_docs['IDF'] = equiv_class_docs['IDF'].apply(math.log)

idf_scores = equiv_class_docs[['equivalence_class', 'IDF']]


# Finding TF for each term in each document
files = os.listdir(token_folder)
files_xlsx = [f for f in files if f[-4:] == 'xlsx']

tf_data = pd.DataFrame()

for f in files_xlsx:
    data = pd.read_excel(f, 'Sheet1')
    data_with_equiv_class = pd.merge(data, equiv_class_source, on='Term')
    data_with_equiv_class = data_with_equiv_class.groupby(
        'equivalence_class').sum().reset_index()
    data_with_equiv_class['tf'] = data_with_equiv_class['Occurrence']/\
                                  len(data_with_equiv_class['Occurrence'])
    data_with_equiv_class['Document'] = os.path.basename(f).replace(
        r'Tokens.xlsx', '')
    tf_data = tf_data.append(data_with_equiv_class)

# Finding TF the correct way
tf_corpus['tf'] = tf_corpus['Occurrence']/len(
    tf_corpus['equivalence_class'])

# Compute TF-IDF from word count and IDF columns using in-doc tf-idf
# First, merge our IDF scores for each term into our tf_data
tf_idf_data_in_doc = pd.merge(tf_data, idf_scores, on='equivalence_class')
tf_idf_data_in_doc['tf-idf'] = tf_idf_data_in_doc['tf'] * tf_idf_data_in_doc['IDF']

#print(tf_idf_data.sort_values('tf-idf'))

tf_idf_data_in_doc.to_csv('tf-idf_experiment.csv')

# Compute 'traditional' tf-idf

tf_idf_data = pd.merge(tf_corpus, idf_scores, on='equivalence_class')
tf_idf_data['tf-idf'] = tf_idf_data['tf'] * tf_idf_data['IDF']

tf_idf_data.to_csv('tf-idf.csv')

