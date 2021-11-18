import os
import re
from os import error

DOMAINS = ["books", "dvd", "kitchen_&_housewares", "electronics"]

def review_file_to_dict(dataset_f):

    with open(dataset_f) as f:
        lines = f.readlines()
        review_dict = {}
        review_idx = 0
        final_lst = []

        # Split each Review in a dictionary of Reviews, we'll obtain something like this:
        # {0: [__rewiew1_content__], 1: [__rewiew2_content__], 2: [__rewiew3_content__] ...}
        for line in lines:
            try:
                "</review>" not in line and review_dict[review_idx].append(line.rstrip())

                if "</review>" in line:
                    review_idx += 1
            except KeyError:
                review_dict[review_idx] = []
            except Exception as err:
                error("An exception occurred: ", err)
        
        # This part will return a list of dictionnaries, dicts which will contain each reviews.
        # [{unique_id: __id__, asin: __asin__, ...}, {unique_id: __id__, asin: __asin__, ...}, {unique_id: __id__, asin: __asin__, ...}]

        for _idx, review in review_dict.items():
            tmp = {}
            category = None
            for line in review:
                if re.match(r'^<', line) and not re.match(r'^</', line):
                    category = line
                    tmp[category] = []
                tmp[category].append(line)
            final_lst.append(dict(map(lambda x: (x[0], ''.join(x[1][1:-1])), tmp.items())))

        # Add negative or positive key in the dictionnary, depending on the parsed file
        [review.update({"<polarity>": -1 if "negative" in dataset_f else 1}) for review in final_lst]
        [review.update({"<domain>": domain}) if domain in dataset_f else None for review in final_lst for domain in DOMAINS]

    return final_lst

def review_file_to_list(folder_name):
    parsed_list = []

    for name in os.listdir(folder_name):
        for review_file in os.listdir(folder_name + '/' + name):
            if "unlabeled" not in review_file:
                parsed_list +=  review_file_to_dict(folder_name + '/' + name + '/' + review_file)
    return parsed_list
