import numpy as np
from tokenizer import encode, EOT
from config import TRAIN_TXT, VAL_TXT, TRAIN, VAL


def prepare(src: str, dst: str):
    with open(src, 'r', encoding='utf-8') as file:
        raw = file.read()

    stories = []
    for story in raw.split('<|endoftext|>'):
        story = story.strip()
        if story:
            stories.append(story)

    num_tokens = 0
    with open(dst, 'wb') as out:
        for story in stories:
            encoding = encode(story)
            encoding.append(EOT)
            num_tokens += len(encoding)
            np.array(encoding, dtype=np.uint16).tofile(out)

    print(f'{dst}: {len(stories)} stories, {num_tokens} tokens, {len(raw)/num_tokens:.2f} chars/token')


if __name__ == '__main__':
    prepare(TRAIN_TXT, TRAIN)
    prepare(VAL_TXT, VAL)
