"""
Data cleaning(pre-processing):
    1. Remove all reviews contains irregular special characters, like …, § ✦ ♬ ☐ ☑ ☼ ☣ ۞ emojis and other interesting staff commonly contained in gaming reviews. Except for "♥". We found people love to type ♥, and ♥ is a strong indicator of positive reviews.

    2.Calculate statistics about ♥ and divide the data to train, dev, and test.
"""

import re
import random
import sys

input_file = sys.argv[1]

""" convert csv to tsv, and separate year and month into two columns.
      store all special_chars apperrs, happens to be non-english chars and emojis. """
TOTAL = 0
special_chars = set()
out = open("review.tsv", "w")
special = open("special-chars.txt", "w")
with open(sys.argv[1], "r") as f:
    for line in f:
        TOTAL += 1
        rec = line.strip().split(",")
        rec[8] = rec[8][1:]
        rec[9] = rec[9][:-2]
        for x in rec[-1]:
            if ord(x) >= 128:
                special_chars.add(x)
        out.write("\t".join(rec) + "\n")
out.close()
special.write(" ".join(special_chars))
special.close()
print("Total number of raw review:", TOTAL)


""" separate reviews with/withOUT special chars """
review_with_special = open("review-with-special", "w")
review_with_special.write("\t".join(["label", "review"]) + "\n")
WITH_SPECIAL = 0  # number of (invalid) data: reviews with special chars

review_without_special = open("review-ascii-only", "w")
review_without_special.write("\t".join(["label", "review"]) + "\n")
N = 0  # total number of (valid) data: reviews without special chars except for ♥

def features(rec, num_class=2):
    # field is zero-based indexing
    if num_class == 2:
        return [rec[field] for field in (6, -1)]

# special_chars.remove("♥")
SPECIAL_REGEX = re.compile("[" + "".join(special_chars) + "]")
with open("review.tsv", "r") as f:
    pos_heart, heart = 0, 0  # calculate the proportion of positive reviews in reviews contain ♥
    for line in f:
        rec = line.strip().split("\t")
        if re.search(SPECIAL_REGEX, rec[-1]):
            review_with_special.write("\t".join(features(rec, num_class=2)) + "\n")
            WITH_SPECIAL += 1
        else:
            review_without_special.write("\t".join(features(rec, num_class=2)) + "\n")
            N += 1
            # if "♥" in rec[-1]:
            #     heart += 1
            #     if rec[6] == "1":
            #         pos_heart += 1

review_with_special.close()
review_without_special.close()

print(f"    - {WITH_SPECIAL} reviews contain speial characters.")
print(f"    - {TOTAL - WITH_SPECIAL}({round((TOTAL - WITH_SPECIAL) / TOTAL * 100, 2)}%) valid (reviews without special characters)")
# print(f"Out of {heart} reviews contain ♥, {pos_heart}({round(pos_heart / heart * 100, 2)}%) of them are positive")





def dev_test_split(file, s=.8):
    dev = open(file + ".dev", "w")
    test = open(file + ".test", "w")
    for _ in (dev, test):
        _.write("label\treview\n")
    with open(file, "r") as f:  # input review files
        f.readline()
        for line in f:
            where = random.uniform(0, 1)
            dev.write(line) if where < s else test.write(line)
    for _ in (dev, test):
        _.close()

dev_test_split("review-ascii-only")


# # 现在是不要❤️
