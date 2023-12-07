# %%
import bs4 
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import ElementClickInterceptedException, StaleElementReferenceException
import os
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import pickle
import urllib
sns.set()

# %%
from multiprocessing.pool import Pool
from contextlib import closing

# %%
from functools import partial

# %%
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
def process_buble(x):
    return float('.'.join(x))

# %%
def bs4_parse_reviews(input_tuple):
    page_to_parse, hotel_name = input_tuple
    records = []
    try:
        for review_page in bs4.BeautifulSoup(page_to_parse).find_all('div', {'class':'WAllg _T'}):

            record = {}
            record['overall_rating'] = process_buble(review_page.find('div',{'data-test-target':'review-rating'})\
                                           .span['class'][-1].split('_')[-1])
            per_type_bubble = review_page.find_all('div', {'class':'hemdC S2 H2 WWOoy'})

            if per_type_bubble:
                for j in per_type_bubble:
                    record[j.text+'_rating'] = process_buble(j.span.span['class'][-1].split('_')[-1])


            record['review'] = review_page.find('div',{'class':'fIrGe _T'}).text
            record['hotel_name'] = hotel_name
            
            records.append(record)
    except Exception as ex:
        print(ex)
        
    return records

# %%
def read_file(path):
    with open(path, 'r') as f:
        return f.read()

# %%
def parse_reviews_multiproc(name, abs_path):
    path = os.path.join(abs_path,name)
    records = []
    for path_page in os.listdir(path):
        page = read_file(os.path.join(path,path_page))
        records.extend(bs4_parse_reviews((page, name)))
    return records

# %%
ABS_PATH = 'trip_advisor_data_hotels'


# %%
hotels_df = pd.read_csv('hotels_links.csv')

# %%
hotels_to_load = hotels_df[hotels_df['parsed']==True]['title'].values.tolist()

# %%
partial_parse_reviews_multiproc = partial(parse_reviews_multiproc, abs_path=ABS_PATH)


# %%
reviews = multiprocess_func([i for i in os.listdir(ABS_PATH) if not i.startswith('.')],
                  func=partial_parse_reviews_multiproc,
                  to_split=False,
                 n_processes=8)

# %%
reviews = pd.DataFrame(reviews)

# %%
reviews = reviews.drop_duplicates(['review', 'hotel_name'])

# %%
reviews.head()

# %%
reviews.shape

# %%
reviews['overall_rating'].value_counts().plot.bar()

# %%
reviews['overall_rating'].value_counts()

# %%
reviews.isna().sum()

# %%
reviews[reviews['overall_rating']==3.0].sample()['review'].values[0]

# %%
reviews.to_csv('hotel_reviews.csv', index=False)

# %%
