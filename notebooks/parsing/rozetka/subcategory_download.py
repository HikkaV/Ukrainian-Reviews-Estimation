"""
This is the first script to be run for rozetka parsing. It collects subcategory links in the website, so we can later iterate over them in the next script (product_links_download)

"""


import random

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.relative_locator import locate_with
import time
from selenium.common.exceptions import NoSuchElementException



test_driver = webdriver.Firefox()
test_driver.set_window_size(2048, 1200)


test_driver.get("https://rozetka.com.ua/ua/")

time.sleep(5)
xpath_categories_list = "/html/body/app-root/div/div/rz-main-page/div/aside/rz-main-page-sidebar/div[1]/rz-sidebar-fat-menu/div/ul"
categories_list = test_driver.find_element(By.XPATH, xpath_categories_list)



xpath_category_link_template = "/html/body/app-root/div/div/rz-main-page/div/aside/rz-main-page-sidebar/div[1]/rz-sidebar-fat-menu/div/ul/li[<ind>]/a"
category_ind = 1


# downloading categories links

more_categories_exist = True

loaded_categories_lst = []

while(more_categories_exist):
    try:
        xpath_category_link = xpath_category_link_template.replace('<ind>', str(category_ind))
        category_link = test_driver.find_element(By.XPATH, xpath_category_link)
        cur_category_link_href = category_link.get_attribute('href')
        cur_category_name = cur_category_link_href.split('/')[-3]
        loaded_categories_lst.append({"category_name": cur_category_name, "category_link": cur_category_link_href})
        category_ind += 1
    except NoSuchElementException:
        more_categories_exist = False


print("loaded_categories_lst")
print(loaded_categories_lst)


loaded_subcategories_lst = []

for cat in loaded_categories_lst:
    test_driver.get(cat["category_link"])
    time.sleep(2)

    row_ind = 1
    col_ind = 1
    rows_available = True
    while (rows_available):


        columns_available = True
        while (columns_available):
            try:
                if row_ind > 50:
                    rows_available = False
                    break
                xpath_subcategory_link_template = "/html/body/app-root/div/div/rz-super-portal/div/main/section/div[2]/rz-dynamic-widgets/rz-widget-list[<row_ind>]/section/ul/li[<col_ind>]/rz-list-tile/div/a[1]"
                xpath_subcategory_link = xpath_subcategory_link_template.replace('<row_ind>', str(row_ind)).replace(
                    '<col_ind>', str(col_ind))
                subcategory_link = test_driver.find_element(By.XPATH, xpath_subcategory_link)
                cur_subcategory_link_href = subcategory_link.get_attribute('href')
                cur_subcategory_name = cur_subcategory_link_href.split('/')[-3]

                entry = cat.copy()
                entry["subcategory_name"] = cur_subcategory_name
                entry["subcategory_link"] = cur_subcategory_link_href

                loaded_subcategories_lst.append(entry)
                print(entry)

                col_ind += 1
            except NoSuchElementException:
                columns_available = False
                col_ind = 1
                row_ind += 1
                print("NEW ROW EXPECTED")
                if row_ind > 50:
                    rows_available = False
                    break

            if row_ind > 50:
                rows_available = False
                break


    test_driver.back()
    time.sleep(2)





pd.to_pickle(loaded_subcategories_lst, "loaded_subcategories_lst.pkl", protocol=2)


