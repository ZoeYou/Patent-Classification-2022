import gensim
from gensim.models.wrappers import FastText

# for depxml embedding
with open("data/all2.depxml.200.wmin20.glove.w2c", "r", encoding="latin-1") as in_f:
    lines = in_f.read().splitlines()
    nb_lines = lines[0]
    nb_dim = lines[1]

    lines = lines[2:]
    keys = [l.split(" ")[0] for l in lines]
    vals = [" ".join(l.split(" ")[1:]) for l in lines]

    cleaned_keys = []
    for k in keys:
        cleaned_keys.append(k.split(":")[0])

with open("data/depGlove.200.txt", "w", encoding="utf-8") as out_f:
    out_f.write(nb_lines + " " + nb_dim + "\n")
    out_f.write("\n".join([k+" "+v for k,v in zip(cleaned_keys, vals)]))


# for stantard embeddings
model = gensim.models.KeyedVectors.load_word2vec_format("data/depGlove.200.txt")
word_vectors = model.wv
print(word_vectors)

word_vectors.save('data/depGlove.200.gensim')
