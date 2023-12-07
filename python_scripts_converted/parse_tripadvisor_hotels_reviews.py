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
from functools import partial
from selenium.webdriver.common.action_chains import ActionChains
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import OperatingSystem, SoftwareName
import pyautogui
import threading
import multiprocessing
from selenium.webdriver.common.proxy import Proxy, ProxyType
sns.set()

# %%
def augment_link(link, num):
    before_link, after_link = link.split('Reviews')
    return before_link+'Reviews-'+f'or{num*5}'+after_link

# %%
import stem

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
# Second level parsing with translate + selenium
"""

# %%
hotels_df = pd.read_csv('hotels_links.csv')

# %%
software_names = [SoftwareName.CHROME.value]
operating_systems = [OperatingSystem.LINUX, OperatingSystem.MACOS.value,
                    OperatingSystem.WINDOWS]

# %%
user_agent_rotator = UserAgent(software_names=software_names,
                               operating_systems=operating_systems, limit=hotels_df.shape[0]*2)

# %%
def save_html(file, path):
    with open(path+'.html', 'w') as f:
        f.write(file)

# %%
def get_driver(user_agent, run_headless=False):
    custom_options = webdriver.ChromeOptions()
    prox = "socks5://localhost:9050"
    custom_options.add_argument('--proxy-server=%s' % prox)
    
    if run_headless:
        custom_options.add_argument('headless')
    custom_options.add_argument("lang=uk")
    custom_options.add_argument('--ignore-certificate-errors')
    custom_options.add_argument('--disable-dev-shm-usage')
    custom_options.add_argument(f'user-agent={user_agent}')
    driver = webdriver.Chrome(options=custom_options)
    return driver

# %%
def check_ip_proxy(address):
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('headless')

    prox = "socks5://localhost:9050"
    options.add_argument('--proxy-server=%s' % prox)
    
    driver = webdriver.Chrome(options=options)
    driver.get('https://api.ipify.org/')
    ip_address = driver.find_element(By.TAG_NAME, "body").text
    driver.quit()
    
    return ip_address

# %%
def check_change_ip(address, default_ip_address, debug=False):
    try:
        ip_address = check_ip_proxy(address)
    except:
        ip_address = None
    
    if debug:
        print(f'Old ip: {default_ip_address}, new ip : {ip_address}')
        
    if default_ip_address!=ip_address and ip_address:
        if debug:
            print('IPs are different')
        return True
    return False

# %%
def access_denied_check_with_address(address, url):
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('headless')
    prox = "socks5://localhost:9050"
    options.add_argument('--proxy-server=%s' % prox)
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)
    except:
        driver.quit()
        return False
    
    html = driver.page_source
    driver.quit()
    try:
        return bs4.BeautifulSoup(html).find('head').title.text!='Access Denied'
    except:
        return True

# %%
def access_denied_check_with_page(html):
    try:
        return bs4.BeautifulSoup(html).find('head').title.text!='Access Denied'
    except:
        return True

# %%
def parse_free_proxies():
    ips = []
    url = 'https://free-proxy-list.net/'
    soup = bs4.BeautifulSoup(requests.get(url).text)
    for i in soup.find('table', {'class':'table table-striped table-bordered'}).find_all('tr'):
        found = i.find_all('td')[:2]
        if found:
            ip, port = found
            ips.append(ip.text+':'+port.text)
    return ips

# %%
def wait_and_click_by(driver, value, by, time_sleep=15):
    WebDriverWait(driver, time_sleep).until(EC.presence_of_element_located((by, value)))
    driver.find_element(by=by, value=value).click()

# %%
"""
## chek proxy
"""

# %%
from collections import Counter
from stem import Signal
from stem.control import Controller

# %%
default_ip = check_ip_proxy('')

# %%
default_ip

# %%
"""
## parsing itself
"""

# %%
import os
import queue

# %%
ABS_PATH = 'trip_advisor_data_hotels'
if not os.path.exists(ABS_PATH):
    os.mkdir(ABS_PATH)
    
for i in hotels_df['title']:
    dir_path = os.path.join(ABS_PATH,i)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


# %%
def parse_reviews(link, path, abs_path, user_agent,
                  parts_scroll=8, sleep_time_list=None, run_headless=True,
                 max_errors=50):
    
    
    # exception handling 
    passed = {'got_initial_link': False,
              'see_all_languages': False}
    passed['link'] = link
    passed['hotel_name'] = path
    
    caught_ex = None
        
    # overall path
    path_to_save = os.path.join(abs_path, path)
    
    #check if there are already parsed pages
    n_already_parsed = len(os.listdir(path_to_save))
    if n_already_parsed:
        link = augment_link(link, n_already_parsed)
    
    
    # get driver
    try:
        driver = get_driver(user_agent, run_headless)
    except Exception as ex:
        caught_ex = ex
    
    if caught_ex:
        passed['got_initial_link'] = False
        passed['num_overall'] = 9999
        passed['num_parsed'] = 0
        passed['exception'] = caught_ex
        return passed


    # initial link getting
    try:
        driver.get(link)
        time.sleep(5)
    except Exception as ex:
        caught_ex = ex
        

    if caught_ex:
        passed['got_initial_link'] = False
        passed['num_overall'] = 9999
        passed['num_parsed'] = 0
        passed['exception'] = caught_ex
        return passed
    else:
        passed['got_initial_link'] = True

    # check if access denied
    if not access_denied_check_with_page(driver.page_source):
        caught_ex = 'Access dnied'
        
    if caught_ex:
        passed['got_initial_link'] = False
        passed['num_overall'] = 9999
        passed['num_parsed'] = 0
        passed['exception'] = caught_ex
        return passed
    
    # see all languages
    try:
        wait_and_click_by(driver, 'Qukvo', By.CLASS_NAME, 30)
        passed['see_all_languages'] = True
        time.sleep(5)
    except:
        passed['see_all_languages'] = False
    

    c = 0
    errors = 0
    first_page = None

    while True:
        passed['show_more'] = False
        passed['saved_file'] = False
        passed['next_page'] = False


        try:
            # show more 
            wait_and_click_by(driver, 'Ignyf', By.CLASS_NAME, 30)
            time.sleep(2)
            passed['show_more'] = True
            # if first page, then save it
            if c == 0:
                first_page = driver.page_source

            # save to txt
            save_html(driver.page_source, os.path.join(path_to_save, f'page_{str(n_already_parsed+c)}'))
            time.sleep(1)
            passed['saved_file'] = True            
            c += 1
            
            # next page
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, 'ui_button.nav.next')))
            button_el = driver.find_element(by=By.CLASS_NAME, value='ui_button.nav.next')
            if button_el.is_enabled() and button_el.is_displayed():
                button_el.click()
            else:
                break
            passed['next_page'] = True   
            errors = 0
            
        except Exception as ex:
            if not isinstance(ex, (StaleElementReferenceException, ElementClickInterceptedException)):
                caught_ex = ex
                break
            else:
                errors+=1
            if errors>=max_errors:
                break  
                
        finally:
            time.sleep(np.random.choice(sleep_time_list))

            
    driver.quit()
    
    if not caught_ex:
        passed = dict([(k, True) for k in passed.keys()])

    try:
        passed['num_overall'] = int(bs4.BeautifulSoup(first_page) \
                                    .find_all('span', {'data-test-target': 'CC_TAB_Reviews_LABEL'})[0] \
                                    .find('span', {'class': 'iypZC Mc _R b'}).text)
        passed['got_overall_num'] = True
    except:
        passed['got_overall_num'] = False
        passed['num_overall'] = 0

    passed['num_parsed'] = 5 * (n_already_parsed+c)
    passed['exception'] = caught_ex

    return passed

# %%
n_threads = 8
headless = True
sleep_time_list = list(range(3,15))

# %%
parse_reviews_partial = partial(parse_reviews,
                                run_headless=headless,
                                sleep_time_list=sleep_time_list,
                               abs_path=ABS_PATH)

# %%
user_agents = [user_agent_rotator.get_random_user_agent() for i in range(hotels_df.shape[0])]

# %%
sub_df = hotels_df[hotels_df['parsed']==False]
input_tuples = list(zip(sub_df['link'].values.tolist(), sub_df['title'].values.tolist(), user_agents))

# %%
batch_size = 100
sleep_between_batches_time = [120, 180, 300, 600]

# %%
batched_input_tuples = [input_tuples[i:i+batch_size] for i in range(0, len(input_tuples)+batch_size, batch_size)]

# %%
def parse_reviews_multiprocessing(input_tuple):
    link, path, user_agent = input_tuple
    passed_dict = parse_reviews_partial(link, path=path, user_agent=user_agent)
    return passed_dict

# %%
for batch in batched_input_tuples:
    with closing(ThreadPool(n_threads)) as p:
        results = list(tqdm(p.imap(parse_reviews_multiprocessing, batch), total=len(batch)))

    mask_passed = dict([(i['link'], i['num_parsed']/(i['num_overall']+1)>0.8) for i in results])
    hotels_df.loc[hotels_df['parsed']==False,'parsed'] = hotels_df.loc[hotels_df['parsed']==False,'link']\
    .apply(lambda x: mask_passed.get(x, False))
    time.sleep(np.random.choice(sleep_between_batches_time))

# %%
hotels_df['parsed'].value_counts()

# %%
hotels_df.to_csv('hotels_links.csv', index=False)

# %%
