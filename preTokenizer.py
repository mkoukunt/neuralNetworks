from tokenizers.pre_tokenizers import WhitespaceSplit

pre_tokenizer = WhitespaceSplit()
print(pre_tokenizer.pre_tokenize_str("Hello, world! How_are you?"))
# Output keeps punctuation attached to words:
# [('Hello,', (0, 6)), ('world!', (7, 13)), ('How', (14, 17)), ('are', (18, 21)), ('you?', (22, 26))]
