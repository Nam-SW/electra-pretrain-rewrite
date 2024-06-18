import re

from mecab import MeCab

mecab = MeCab()


def text_processing(text):
    result = re.sub(r"[^.,:\(\)\[\]%\- 가-힣A-Za-z0-9]", " ", str(text))
    result = result.strip()

    mecab_tokenized = [t[0] for t in mecab.pos(result)]
    return " ".join(mecab_tokenized)
