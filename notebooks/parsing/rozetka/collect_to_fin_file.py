"""
This is the final script that collects all data parsed and stored in comments_files/ to one csv
"""

import json
import pandas as pd
import ast
import os

done_data = set()

for fpath in os.listdir("comments_files/"):
    with open("comments_files/" + fpath) as f:
        for l in f.readlines():
            if l.strip():
                d = ast.literal_eval(l)
                done_data.add(str(d))

done_data = [ast.literal_eval(x) for x in done_data]
print(len(done_data))

pd.DataFrame(done_data).to_csv("parsed_rozetka_reviews.csv")

