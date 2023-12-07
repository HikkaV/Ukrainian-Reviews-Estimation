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
"""
# First level parsing
"""

# %%
software_names = [SoftwareName.CHROME.value]
operating_systems = [OperatingSystem.LINUX.value, OperatingSystem.WINDOWS.value, OperatingSystem.MACOS.value]
user_agent_rotator = UserAgent(software_names=software_names,
                              operating_systems=operating_systems,
                              limit=100)
main_link = 'https://www.tripadvisor.ru/Hotels-g294473-Ukraine-Hotels.html#LEAF_GEO_LIST'

# %%
main_url = 'https://www.tripadvisor.ru/'

# %%
def parse_sites(main_link, user_agent_rotator, max_ex=100):
    first_button_xpath = '//*[@id="component_7"]/div/button'
    next_page_xpath = '//*[@id="taplc_main_pagination_bar_hotels_less_links_v2_0"]/div/div/div/span[2]'
    
    user = user_agent_rotator.get_random_user_agent()
    custom_options = webdriver.ChromeOptions()
    custom_options.add_argument(f'user_agent={user}')
    
    driver = webdriver.Chrome(options=custom_options)
    driver.get(main_link)
    WebDriverWait(driver, 90).until(EC.presence_of_element_located((By.XPATH, first_button_xpath)))
    driver.find_element(by=By.XPATH, value=first_button_xpath).click()
    
    pages = [driver.page_source]
    ex_counter=0
    while True:
        try:
            WebDriverWait(driver, 90).until(EC.presence_of_element_located((By.XPATH, next_page_xpath)))
            driver.find_element(by=By.XPATH, value=next_page_xpath).click()
            ex_counter = 0
        except Exception as ex:
            if not isinstance(ex, (StaleElementReferenceException, ElementClickInterceptedException)):
                print(ex)
                break
            else:
                ex_counter+=1
                
            if ex_counter>=max_ex:
                break
        time.sleep(15)
        pages.append(driver.page_source)
    driver.quit()
    return pages

# %%
def parse_first_lvl(page):
    soup = bs4.BeautifulSoup(page)
    to_save = []
    for ui_column in soup.find_all('div', {'class':'ui_column is-8 main_col allowEllipsis'}):
        try:
            bubble_rating_parsed = ui_column.find('a', {'data-clicksource':'BubbleRating'})

            to_save.append((bubble_rating_parsed.get('alt'), bubble_rating_parsed.get('href'),
                        ui_column.find('div', {'class':'listing_title'}).text))
        except:
            pass
    return to_save

# %%
pages = parse_sites(main_link, user_agent_rotator)

# %%
hotels_df = pd.DataFrame(multiprocess_func(pages, parse_first_lvl,
                      gather_func=None, to_split=False,
                      n_processes=8), columns=['rating', 'link', 'title'])

# %%
hotels_df = hotels_df.drop_duplicates()

# %%
hotels_df['link'] = hotels_df['link'].apply(lambda x: urllib.parse.urljoin(main_url, x))
hotels_df['title'] = hotels_df['title'].apply(lambda x: '.'.join(x.split('.')[1:]).strip())
hotels_df['rating'] = hotels_df['rating'].apply(lambda x: float(x.split('of')[0].strip().replace(',','.')))

# %%
hotels_df['title'] = hotels_df['title'].apply(lambda x: x.replace('/','\\'))

# %%
hotels_df['parsed'] = False

# %%
hotels_df.to_csv('hotels_links.csv', index=False)

# %%
