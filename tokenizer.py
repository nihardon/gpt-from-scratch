import tiktoken

encoder = tiktoken.get_encoding('gpt2')

VOCAB_SIZE = encoder.n_vocab
EOT = encoder.eot_token

encode = encoder.encode_ordinary
decode = encoder.decode

