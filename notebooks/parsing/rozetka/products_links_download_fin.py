"""
This is the second script to run for parsing rozetka. It collects data about links to products and stores them in a file. Basically it is link to product + its category, subcategory
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


loaded_subcategories_lst = pd.read_pickle("loaded_subcategories_lst.pkl")


test_driver.get("https://rozetka.com.ua/ua/")


time.sleep(5)

prodlinks_data = []

num_processed_subcats = 0

for subcat_data in loaded_subcategories_lst:

    cur_subcategory_link_href = subcat_data["subcategory_link"]
    cur_subcategory_name = subcat_data["subcategory_name"]

    print("num_processed_subcats")
    print(num_processed_subcats)
    print("cur_subcategory_name")
    print(cur_subcategory_name)

    try:

        #subcategory_link = test_driver.find_element(By.XPATH, xpath_subcategory_link)


        test_driver.get(cur_subcategory_link_href)
        time.sleep(3)

        with open("product_links.txt", "a") as f:
            f.write("- CATEGORY NAME:\n")
            f.write("- " + subcat_data["category_name"] + "\n")

            f.write("-- SUBCATEGORY NAME:\n")
            f.write("-- " + cur_subcategory_name + "\n")

        #viewing list of products

        prod_ind = 1

        xpath_prod_link_template = "/html/body/app-root/div/div/rz-category/div/main/rz-catalog/div/div/section/rz-grid/ul/li[<prod_ind>]/rz-catalog-tile/app-goods-tile-default/div/div[2]/a[1]"

        more_prods_exist=True
        links_parsed_in_subcat = 0

        while(more_prods_exist and (links_parsed_in_subcat < 1000)):
            print("Products while")

            try:
                xpath_prod_link = xpath_prod_link_template.replace("<prod_ind>", str(prod_ind))
                prod_button = test_driver.find_element(By.XPATH, xpath_prod_link)
                prod_link = prod_button.get_attribute("href")
                print(prod_link)

                prodlinks_data.append({"category_name": subcat_data["category_name"], "subcategory_name": subcat_data["subcategory_name"], "prod_link": prod_link})

                with open("product_links2.txt", "a") as f:
                    f.write(prod_link + "\n")

                print("adding to prod ind")
                prod_ind += 1
                links_parsed_in_subcat += 1

            except NoSuchElementException:
                print("Didn't find product")
                more_products_xpath = "/html/body/app-root/div/div/rz-category/div/main/rz-catalog/div/div/section/rz-catalog-paginator/rz-load-more/a/span"
                try:
                    more_products_button = test_driver.find_element(By.XPATH, more_products_xpath)
                    more_products_button.click()
                    time.sleep(3)
                except NoSuchElementException:
                    more_prods_exist = False
                    break

        num_processed_subcats += 1
    except:
        print("TRYING SUBCATEGORY AGAIN:")
        print(cur_subcategory_name)
        test_driver.get(cur_subcategory_link_href)
        time.sleep(3)




pd.to_pickle(prodlinks_data, "prodlinks_data.pkl", protocol=2)
