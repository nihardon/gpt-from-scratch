import tiktoken
import numpy as np

DEBUG = True

encoder = tiktoken.get_encoding('gpt2')

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
        text = file.read()
    
    raw = text.split('<|endoftext|>')
    text = []
    for story in raw:
        story = story.strip()
        if story:
            text.append(story)

    num_stories = len(text)
    num_tokens = 0
    with open(f'data/{t}.bin', 'wb') as out:
        for story in text:
            encoding = encoder.encode_ordinary(story)
            encoding.append(encoder.eot_token)
            num_tokens += len(encoding)
            arr = np.array(encoding, dtype=np.uint16)
            arr.tofile(out)
    
    if DEBUG:
        print(t + f' stories: {num_stories}')
        print(t + f' tokens: {num_tokens}')
        print(t + f' chars/token: {len(TEXT)/num_tokens}')

if __name__ == '__main__':
    prepare('train')
    prepare('val')