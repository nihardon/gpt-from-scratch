import numpy as np
from tokenizer import encode, EOT 


DEBUG = True

def prepare(t: str):
    valid_inputs = {
        'valid' : 'val',
        'val' : 'val',
        'validation' : 'val',
        'train' : 'train'
    }
    t = valid_inputs[t]

    path = 'data/TinyStoriesV2-GPT4-train.txt'
    if t == 'val':
        path = 'data/TinyStoriesV2-GPT4-valid.txt'
    
    with open(path, 'r', encoding='utf-8') as file:
        raw = file.read()
    
    split = text.split('<|endoftext|>')
    text = []
    for story in split:
        story = story.strip()
        if story:
            text.append(story)

    num_stories = len(text)
    num_tokens = 0
    with open(f'data/{t}.bin', 'wb') as out:
        for story in text:
            encoding = encode(story)
            encoding.append(EOT)
            num_tokens += len(encoding)
            arr = np.array(encoding, dtype=np.uint16)
            arr.tofile(out)
    
    if DEBUG:
        print(t + f' stories: {num_stories}')
        print(t + f' tokens: {num_tokens}')
        print(t + f' chars/token: {len(raw)/num_tokens}')

if __name__ == '__main__':
    prepare('train')
    prepare('val')