import re

import emoji
import string

def tweet_cleaner(sentence: str) -> str:
    sentence = re.sub(r'@[A-Za-z0-9_]+','',sentence)
    sentence = re.sub(r'#','',sentence)
    sentence = re.sub(r'RT : ','',sentence)
    sentence = re.sub(r'\n','',sentence)
    # to remove emojis
    sentence = re.sub(emoji.get_emoji_regexp(), r"", sentence)
    sentence = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',sentence)
    sentence = re.sub(r"https?://\S+|www\.\S+","",sentence)
    sentence = re.sub(r"<.*?>","",sentence)
    return sentence

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
