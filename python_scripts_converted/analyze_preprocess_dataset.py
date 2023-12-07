# %%
import os
import pandas as pd
import gc

# %%
from multiprocessing.pool import Pool
from contextlib import closing
from tqdm import tqdm


def multiprocess_func(main_input, func, additional_inputs=None,
                      gather_func=None, to_split=True, gather_func_args=None,
                      chunk_size=100, n_processes=8):
    if not gather_func_args:
        gather_func_args = []
    if not additional_inputs:
        additional_inputs = []
    if not gather_func:
        gather_func = lambda x: [z for i in x for z in i]
    if to_split:
        splitted = [(main_input[i:i + chunk_size], *additional_inputs) if additional_inputs else main_input[i:i + chunk_size]\
                    for i in range(0, len(main_input), chunk_size)]
    else:
        splitted = [(i, *additional_inputs) if additional_inputs else i for i in main_input]
    with closing(Pool(n_processes)) as p:
        result = list(tqdm(p.imap(func, splitted),
                           total=len(splitted)))
    return gather_func(result, *gather_func_args)


# %%
path = '../../../data_reviews/'

# %%
! wc -l $path/*

# %%
"""
# Merging the data
"""

# %%
df1 = pd.read_csv(os.path.join(path, 'rozetka_ukr.csv'), encoding='windows-1251', 
           sep=';')
df1.shape

# %%
df1['entity_name'] = df1['prod_link'].apply(lambda x: x.split('/')[-3])

# %%
df1 = df1[['comment','translate', 'rating', 'entity_name']]\
.rename(columns={'comment':'review', 'translate':'review_translate'})
df1['dataset_name'] = 'rozetka'

# %%
df2 = pd.read_csv(os.path.join(path, 'rozetka_ru.csv'), encoding='windows-1251', 
           sep=';')
df2.shape

# %%
df2 = df2[~df2['prod_link'].isna()]

# %%
df2['entity_name'] = df2['prod_link'].apply(lambda x: x.split('/')[-3])

# %%
df2 = df2[['comment','translate', 'rating', 'entity_name']]\
.rename(columns={'comment':'review', 'translate':'review_translate'})
df2['dataset_name'] = 'rozetka'

# %%
df3 = pd.read_csv(os.path.join(path, 'hotels_final.csv'), encoding='windows-1251', 
           sep=';')
df3.shape

# %%
df3 = df3.rename(columns={'hotel_name':'entity_name'})

# %%
df3 = df3[['review', 'translate', 'overall_rating', 'entity_name']]\
.rename(columns={'overall_rating' : 'rating', 'translate':'review_translate'})
df3['dataset_name'] = 'tripadvisor_hotels_ukraine'

# %%
df4 = pd.read_csv(os.path.join(path, 'restaurants_review_final.csv'), encoding='windows-1251', 
           sep=';')
df4.shape

# %%
df4 = df4.rename(columns={'name':'entity_name'})

# %%
df4 = df4.rename(columns={'overall_rating' : 'rating'})[['review', 'title_translate', 'review_translate', 'rating',
                                                        'entity_name']]
df4['dataset_name'] = 'tripadvisor_restaurants_ukraine'

# %%
df = pd.concat([df1, df2, df3, df4], axis=0)

# %%
df.head()

# %%
del df1, df2, df3, df4;
gc.collect();

# %%
df = df[~df['rating'].isna()]

# %%
df['title_translate'] = df['title_translate'].fillna('')

# %%
df = df[~df['review'].isna()]

# %%
df['translated'] = df['review']!=df['review_translate']

# %%
df.isna().sum()

# %%
df['translated'].value_counts()

# %%
df.shape

# %%
df[df['rating']==2].sample(1)[['review', 'review_translate']].values

# %%
df['entity_name'].nunique()

# %%
"""
# Basic data analysis
"""

# %%
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
sns.set()

# %%
from nltk.tokenize import sent_tokenize

# %%
"""
## Characters number 
"""

# %%
print('Max number of characters in translated review : {}'.format(df['review_translate'].apply(len).max()))
print('Min number of characters in translated review : {}'.format(df['review_translate'].apply(len).min()))
print('Mean number of characters in translated review : {}'.format(df['review_translate'].apply(len).mean()))
print('Median number of characters in translated review : {}'.format(df['review_translate'].apply(len).median()))


# %%
sns.distplot(np.log10(df['review_translate'].apply(len)))

# %%
np.percentile(df['review_translate'].apply(len), q=0.2)

# %%
"""
### filter out those reviews which char len is an outlier
"""

# %%
df = df[df['review_translate'].apply(len)>np.percentile(df['review_translate'].apply(len), q=0.2)]

# %%
"""
### find those reviews which have a lot less characters that real text
"""

# %%
df['diff_len'] = df['review'].apply(len)-df['review_translate'].apply(len)

# %%
df = df[df['review_translate']!='#ERROR!']

# %%
df['diff_len'] = df['diff_len'].apply(abs)

# %%
sns.distplot(np.log1p(df['diff_len']))

# %%
df = df[df['diff_len']<200]

# %%
df[df['translated']==True]['diff_len'].max()

# %%
df = df.drop(columns=['diff_len'])

# %%
"""
### deleting empty symbols
"""

# %%
df['review_translate'] = df['review_translate'].str.strip()
df = df[df['review_translate'].apply(lambda x: True if x else False)]

# %%
"""
### remove \n char
"""

# %%
df['translated'].value_counts()

# %%
import re

# %%
def remove_multy_spaces(text):
    try:
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as ex:
        return None

# %%
df['review_translate'] = df['review_translate'].str.replace('\n', '').str.strip()

# %%
def spacing_between_chars_text(text):
    text = list(text)
    new_text = []
    for idx_char in range(len(text)):
        if not text[idx_char].isalnum() and text[idx_char]!="'" and text[idx_char]!=' ':
            new_text.append(' ')
            new_text.append(text[idx_char])
            new_text.append(' ')
        else:
            new_text.append(text[idx_char])

    return ''.join(new_text).strip()

# %%
df['review_translate'] = multiprocess_func(df['review_translate'].values, 
                  func=spacing_between_chars_text,
                  gather_func=lambda x: x,
                  to_split=False)

# %%
df['review_translate'] = multiprocess_func(df['review_translate'].values, 
                  func=remove_multy_spaces,
                  gather_func=lambda x: x,
                  to_split=False)

# %%
df['review_translate'].values[0]

# %%
"""
## Sentence number
"""

# %%
sent_tokenized = multiprocess_func(df['review_translate'].values, 
                  func=sent_tokenize,
                  gather_func=lambda x: x,
                  to_split=False)

# %%
sns.distplot(np.log10([len(i) for i in sent_tokenized]))

# %%
df['review_translate_sentences'] = sent_tokenized

# %%
"""
# Delete those which are partially translated
"""

# %%
import fasttext
from itertools import chain

# %%
model = fasttext.load_model('../../../lid.176.bin')

# %%
def detect_lang_sentences(batched_texts, model):
    result = []
    for texts in tqdm(batched_texts):
        lengths = [len(i) for i in texts]
        sentences = list(chain(*texts))
        predicted_langs, _ = model.predict(sentences)
        predicted_langs = list(map(lambda x: x[0].split('__')[-1], predicted_langs))
        assert sum(lengths)==len(sentences)
        assert len(predicted_langs)==len(sentences)
        batched_langs = []
        start = 0
        end = lengths[0]
        for i in lengths[1:]:
            to_add = predicted_langs[start:end]
            if not to_add:
                break
            batched_langs.append(to_add)
            start = end
            end = end+i
            
        if predicted_langs[start:end]:
                batched_langs.append(predicted_langs[start:end])
        assert [len(i) for i in batched_langs]==lengths
        result.extend(batched_langs)

    return result

# %%
def detect_lang(batched_texts, model):
    result = []
    for texts in tqdm(batched_texts):
        predicted_langs, _ = model.predict(list(texts))
        result.extend(list(map(lambda x: x[0].split('__')[-1], predicted_langs)))

    return result

# %%
batch_size=100
to_detect_lang = df.loc[df['translated']==True, 'review_translate_sentences'].values
batches = [to_detect_lang[i:i+batch_size] for i in range(0, len(to_detect_lang), batch_size)]

# %%
sum([len(i) for i in batches])

# %%
result = detect_lang_sentences(batches, model)

# %%
batch_size=100
to_detect_lang = df.loc[df['translated']==True, 'review_translate'].values
batches = [to_detect_lang[i:i+batch_size] for i in range(0, len(to_detect_lang), batch_size)]

# %%
result = detect_lang(batches, model)

# %%
df['language_translated'] = 'uk'
df.loc[df['translated']==True, 'language_translated'] = result

# %%
df = df[df['language_translated']=='uk']

# %%
df.drop(columns='language_translated', inplace=True)

# %%
"""
# Tokenize texts
"""

# %%
from nltk.tokenize import regexp_tokenize

# %%
def NLTK_special_chars_excluded_tokenizer(input_text):
    overall_pattern = r"[\w'-]+|[^\w\s'-]+"
    return regexp_tokenize(input_text, pattern=overall_pattern, gaps=False, discard_empty=True)

# %%
def tokenize_sentence_tokens(sentences):
    tokens = []
    for sent in sentences:
        tokens.append(NLTK_special_chars_excluded_tokenizer(sent))
    return tokens

# %%
df['review_translate_sentences_tokens'] = multiprocess_func(df['review_translate_sentences'].values, 
                  func=tokenize_sentence_tokens,
                  gather_func=lambda x: x,
                  to_split=False)

# %%
"""
# Add spaces between chars 
"""

# %%
from functools import partial

# %%
def apply_func_sent(sentences, func):
    result = []
    for sent in sentences:
        result.append(func(sent))
    return result

# %%
def spacing_between_chars_tokens(tokens):
    tokens = list(np.hstack([spacing_between_chars(i) for i in tokens]))
    return [i for i in tokens if i]

# %%
def spacing_between_chars(text):
    text = list(text)
    new_text = []
    for idx_char in range(len(text)):
        if not text[idx_char].isalnum() and text[idx_char]!="'":
            new_text.append(' ')
            new_text.append(text[idx_char])
            new_text.append(' ')
        else:
            new_text.append(text[idx_char])

    return ''.join(new_text).strip().split(' ')

# %%
spacing_between_chars_sentences = partial(apply_func_sent, func=spacing_between_chars_tokens)

# %%
df['review_translate_sentences_tokens'] = multiprocess_func(df['review_translate_sentences_tokens'].values, 
                  func=spacing_between_chars_sentences,
                  gather_func=lambda x: x,
                  to_split=False)

# %%
"""
# Find pos tags
"""

# %%
import pymorphy2

# %%
morph = pymorphy2.MorphAnalyzer(lang='uk')

# %%
def pos_tagging(ent):
    batch, morph = ent
    tags_batch = []
    for sentences in batch:
        tags_sentences = []
        for sentence in sentences:
            tags_sentences.append([morph.parse(word)[0].tag._POS for word in sentence])
        tags_batch.append(tags_sentences)
    return tags_batch

# %%
df['review_translate_sentences_pos'] = multiprocess_func(df['review_translate_sentences_tokens'].values, 
                  func=pos_tagging,
                  gather_func=None,
                  to_split=True,
                  chunk_size=100,
                  n_processes=12,
                  additional_inputs=[morph])

# %%
"""
# Find lemmas
"""

# %%
def lemmatizing(ent):
    batch, morph = ent
    tags_batch = []
    for sentences in batch:
        tags_sentences = []
        for sentence in sentences:
            tags_sentences.append([morph.parse(word)[0].normal_form for word in sentence])
        tags_batch.append(tags_sentences)
    return tags_batch

# %%
df['review_translate_sentences_lemma'] = multiprocess_func(df['review_translate_sentences_tokens'].values, 
                  func=lemmatizing,
                  gather_func=None,
                  to_split=True,
                  chunk_size=100,
                  n_processes=12,
                  additional_inputs=[morph])

# %%
df.head(1)

# %%
"""
# Delete plain questions
"""

# %%
def is_question_sentences(ent):
    sentences, tags = ent
    is_question_vector = []
    for i in range(len(sentences)):
        is_question_vector.append(is_question(sentences[i], tags[i]))
    return is_question_vector

# %%
def is_question(words, tags):
    tags = [tag for word, tag in list(zip(words, tags))\
            if not word in ['.', ',', '!', '?']]
    
    # Check if the last character of the sentence is a question mark
    if words[-1] == "?" and len(tags)>1:
        # Check if the sentence ends with a verb or an auxiliary verb
        if tags[-1] in ["VERB", "INFN"] or (tags[-1] == "GRND" and tags[-2] in ["VERB", "INFN"]):
            return True
        # Check if the sentence starts with an auxiliary verb and ends with a verb
        elif tags[0] == "PRCL" and tags[-1] in ["VERB", "INFN"]:
            return True
        else:
            return False
    elif words[-1]=='?' and len(tags)==1:
        return True
    else:
        return False


# %%
to_input = list(zip(df['review_translate_sentences_tokens'].values.tolist(), 
            df['review_translate_sentences_pos'].values.tolist()))

# %%
questions_mask = multiprocess_func(to_input, 
                  func=is_question_sentences,
                  gather_func=lambda x: x,
                  to_split=False,
                  n_processes=12,
                  )

# %%
df['is_question'] = questions_mask

# %%
df = df[~df['is_question'].apply(lambda x: all(x))]

# %%
df.to_csv('processed_data.csv', index=False)

# %%
