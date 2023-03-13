"""
This is the third script to parse the comments for rozetka themselves.
You might need to run script_to_keep_parsing before it

Also each time parsing fails you MUST change rerun_id variable (increase it) so that new
files where parsed data is stored have different name 
"""

from tqdm import tqdm
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.relative_locator import locate_with
import time
import random
from selenium.common.exceptions import NoSuchElementException
from multiprocessing import Process


import multiprocessing, logging
#mpl = multiprocessing.log_to_stderr()
#mpl.setLevel(logging.INFO)


#NOTE: notdone_data is generated with script_to_keep_parsing in case 
#parsing fails and we need to continue

prodlinks_data = pd.read_pickle("notdone_data.pkl")

random.shuffle(prodlinks_data)



comms_data = []


N_threads = 8


def chunks_gen(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def task(chunk, chunk_id):

    rerun_id = 0  # NOTE: increase manually after each code restart when continue parsing


    #mpl.info("Process number: " + str(chunk_id))

    try:
        test_driver = webdriver.Firefox()
        test_driver.set_window_size(2048, 1200)

        with open("comments_files/parsed_data_" + str(rerun_id) + "_" + str(chunk_id) + ".txt", "w") as f:
            f.write("\n")

        for prod in tqdm(chunk):

            try:
                #print(prod)
                test_driver.get(prod["prod_link"])
                time.sleep(4)

                xpath_all_comments_page = "/html/body/app-root/div/div/rz-product/div/rz-product-tab-main/div[1]/div[2]/div[2]/rz-product-comments-tab-main/section/a"

                more_comments_was_clicked = False
                try:
                    all_comments_button = test_driver.find_element(By.XPATH, xpath_all_comments_page)
                    all_comments_button.click()
                    time.sleep(3)
                    more_comments_was_clicked = True
                except NoSuchElementException:
                    pass

                comment_ind = 1
                xpath_comment_template = "/html/body/app-root/div/div/rz-product/div/rz-product-tab-feedback/div/div/main/rz-product-comments/section/section/rz-comment-list/ul/li[<comment_ind>]/rz-product-comment/rz-comment/div/div/div[2]"

                more_comments_exists = True
                while (more_comments_exists):
                    try:
                        xpath_comment = xpath_comment_template.replace('<comment_ind>', str(comment_ind))
                        comment_element = test_driver.find_element(By.XPATH, xpath_comment)
                        #mpl.info("comment_element")
                        num_stars = -1
                        general_comment_text = ""
                        extra_comment_text = ""

                        general_comment_element_xpath = xpath_comment + '/p'
                        #mpl.info(general_comment_element_xpath)
                        
                        if len(test_driver.find_elements(By.XPATH, general_comment_element_xpath)) != 0:
                            #mpl.info("general_comment_element_xpath")
                            general_comment_element = comment_element.find_element(By.XPATH, general_comment_element_xpath)
                            general_comment_text = general_comment_element.text
                            #mpl.info(general_comment_text)

                        #mpl.info("mid 1")


                        extra_comment_element_xpath = xpath_comment + '/dl'
                        if len(test_driver.find_elements(By.XPATH, extra_comment_element_xpath)) != 0:
                            #mpl.info("extra_comment_element_xpath")
                            extra_comment_element = comment_element.find_element(By.XPATH, extra_comment_element_xpath)
                            extra_comment_text = extra_comment_element.text
                            #mpl.info(extra_comment_text)

                        #mpl.info("mid 2")

                        total_comment_text = general_comment_text + "\n" + extra_comment_text

                        #mpl.info(total_comment_text)


                        stars_element_xpath_v1 = xpath_comment + '/rz-comment-rating/div/ul/li[1]'
                        stars_element_xpath_v2 = xpath_comment + '/rz-comment-rating/div'

                        if len(test_driver.find_elements(By.XPATH, stars_element_xpath_v1)) != 0:
                            #mpl.info("stars_element_xpath_v1")
                            stars_element = comment_element.find_element(By.XPATH, stars_element_xpath_v1)
                            stars_element_html = stars_element.get_attribute("innerHTML")
                            num_stars = stars_element_html.count('#ffa900')

                        elif len(test_driver.find_elements(By.XPATH, stars_element_xpath_v2)) != 0:
                            #mpl.info("stars_element_xpath_v2")
                            stars_element = comment_element.find_element(By.XPATH, stars_element_xpath_v2)
                            stars_element_html = stars_element.get_attribute("innerHTML")
                            num_stars = stars_element_html.count('#ffa900')


                        #full_html = comment_element.get_attribute("innerHTML")

                        #num_stars = full_html.count('#ffa900')

                        # print("NUM STARS: " + str(num_stars))
                        # print(comment_element.text)

                        #mpl.info("NUM STARS: " + str(num_stars))
                        #mpl.info(total_comment_text)

                        doc = prod.copy()
                        doc["comment"] = total_comment_text
                        doc["rating"] = str(num_stars)

                        #print(doc)

                        with open("comments_files/parsed_data_" + str(rerun_id) + "_" + str(chunk_id) + ".txt", "a") as f:
                            f.write(str(doc) + "\n")

                        comment_ind += 1

                    except NoSuchElementException:
                        try:
                            xpath_more_comments = "/html/body/app-root/div/div/rz-product/div/rz-product-tab-feedback/div/div/main/rz-product-comments/section/section/button"
                            more_comments_button = test_driver.find_element(By.XPATH, xpath_more_comments)
                            more_comments_button.click()
                        except NoSuchElementException:
                            more_comments_exists = False
                            break
            except:
                pass
    except:
        print("Chunk " + chunk_id + " is broken")


prodlinks_chunks = list(chunks_gen(prodlinks_data, len(prodlinks_data) // N_threads))

print("prodlinks_chunks")
print(len(prodlinks_chunks))
print(prodlinks_chunks[0][:10])
time.sleep(10)

for ind, c in enumerate(prodlinks_chunks):
    print("Length of chunk " + str(ind))
    print(len(c))


processes = []
for ind, chunk in enumerate(prodlinks_chunks):
    t = Process(target=task, args=(chunk, ind))
    processes.append(t)
    t.start()


print("Join called")
for t in processes:
    t.join()

print("After join")

