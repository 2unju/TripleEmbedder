import os
import re
import pandas as pd


class Mapping:
    def __init__(self):
        relation_path = os.path.join(os.path.dirname(__file__), "rule", "relations.tsv")
        relations = pd.read_csv(relation_path, sep="\t", names=["rel", "nl"], header=None).values.tolist()
        self.relations = dict()
        for rel in relations:
            self.relations[rel[0]] = rel[1]
        print("All Relations Loaded")

    def has_final_consonant(self, word):
        return not (ord(word[-1]) - 44032) % 28 == 0

    def set_entity(self, entity, tok: str, sentence):
        if self.has_final_consonant(entity):
            sentence = re.sub(tok + "와\s", tok + "과 ", sentence)
            sentence = re.sub(tok + "가\s", tok + "이 ", sentence)
            sentence = re.sub(tok + "는\s", tok + "은 ", sentence)
            sentence = re.sub(tok + "를\s", tok + "을 ", sentence)
            sentence = re.sub(tok + "로\s", tok + "으로 ", sentence)
            sentence = re.sub(tok + "로서\s", tok + "으로서 ", sentence)
            sentence = re.sub(tok + "로써\s", tok + "으로써 ", sentence)
        else:
            sentence = re.sub(tok + "과\s", tok + "와 ", sentence)
            sentence = re.sub(tok + "이\s", tok + "가 ", sentence)
            sentence = re.sub(tok + "은\s", tok + "는 ", sentence)
            sentence = re.sub(tok + "을\s", tok + "를 ", sentence)
            sentence = re.sub(tok + "으로\s", tok + "로 ", sentence)
            sentence = re.sub(tok + "으로서\s", tok + "로서 ", sentence)
            sentence = re.sub(tok + "으로써\s", tok + "로써 ", sentence)

        return re.sub(tok, entity, sentence)

    def mapping_to_nls(self, triples: list):
        res = []
        for triple in triples:
            try:
                nl_format = self.relations[triple[1]]
            except:
                print("relation {} dosen't has rule".format(triple[1]))
                print(f"{triple[0]} {triple[1]} {triple[2]}")
                print(triple)
                exit()
            try:
                nl = self.set_entity(triple[0], "<s>", nl_format)
                nl = self.set_entity(triple[2], "<o>", nl)
            except:
                print(nl_format)
                print(triple)
                exit()
            res.append(nl)

        return res

    def mapping_to_nl(self, triple):
        try:
            nl_format = self.relations[triple[1]]
        except:
            print("relation {} dosen't has rule".format(triple[1]))
            print(f"{triple[0]} {triple[1]} {triple[2]}")
            exit()
        try:
            nl = self.set_entity(triple[0], "<s>", nl_format)
            nl = self.set_entity(triple[2], "<o>", nl)
        except:
            print(nl_format)
            print(triple)
            exit()
        return nl