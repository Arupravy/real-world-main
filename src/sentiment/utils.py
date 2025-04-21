# sentiment/utils.py
import re

def clean_text(text):
    return re.sub(r"http\S+|www\S+|[^a-zA-Z0-9\s]", '', text)
