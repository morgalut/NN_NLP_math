# nlp_analysis.py

import spacy

class ArithmeticNLP:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_sentence(self, sentence):
        doc = self.nlp(sentence)
        for token in doc:
            print(f"{token.text}: {token.dep_} (head: {token.head.text})")
