"""
This is the script to be run before comments_parser_threads
"""

import json
import pandas as pd
import ast
import os

prodlinks_data = pd.read_pickle("prodlinks_data.pkl")


done_data = set()
done_data_lst = []

for fpath in os.listdir("comments_files/"):
    with open("comments_files/" + fpath) as f:
        for l in f.readlines():
            if l.strip():
                #l = l.strip().replace('\'', '\"')
                #print(l)
                d = ast.literal_eval(l)
                for k in ["comment", "rating"]:
                    del d[k]
                done_data.add(str(d))
                done_data_lst.append(str(d))

prodlinks_data = set([str(x) for x in prodlinks_data])

notdone_data = prodlinks_data - done_data



print("ALL data length")
print(len(prodlinks_data))
print(list(prodlinks_data)[:4])
print("Done data length")
print(len(done_data))
print(len(done_data_lst))
print(list(done_data)[:4])
print("Not done data length")
print(len(notdone_data))
print(list(notdone_data)[:4])

notdone_data = [ast.literal_eval(x) for x in notdone_data]

print("Not done fin")
print(notdone_data[:4])
pd.to_pickle(notdone_data, "notdone_data.pkl", protocol=2)
