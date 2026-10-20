import tiktoken

gpt2 = tiktoken.get_encoding('gpt2')
print(gpt2.n_vocab)

sentence = gpt2.encode('Hello my name is Nihar')
print(sentence)
for token in sentence:
    print(gpt2.decode([token]))

with open('data/input.txt', 'r') as file:
    TEXT = file.read()

text = gpt2.encode(TEXT)
print(len(TEXT)/len(text))