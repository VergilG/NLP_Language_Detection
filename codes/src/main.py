from preprocessing import preprocess_corpus
import os
from collections import Counter

# 预处理，只保留汉字和常见汉字标点符号
path = '/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/data/raw/backup/raw_chat_corpus/weibo-400w/weibo.txt'

import thulac

thu = thulac.thulac()
a = thu.cut('以，为，曰，也，有，而，其，一')
print(a)
