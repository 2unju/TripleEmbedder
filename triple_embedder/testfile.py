from transformers import AutoTokenizer

def test():
    tok = AutoTokenizer.from_pretrained("klue/bert-base")
    def tokenize(q, r=None, o=None):
        if not q:
            return None
        if r and o:
            q = q.strip() + "[SEP]" + r.strip() + "[SEP]" + o.strip()
        return tok(q, add_special_tokens=True, max_length=32, padding='max_length', return_tensors="pt", truncation=True)["input_ids"]

    tri = ["장미", "isA", "꽃"]
    print(tokenize(tri))
    nl = "장미는 꽃이다."
    print(tokenize(nl))
    print("check")


test()