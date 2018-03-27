# from ekphrasis.classes.preprocessor import TextPreProcessor
# from ekphrasis.classes.tokenizer import SocialTokenizer
# from ekphrasis.dicts.emoticons import emoticons

# text_processor = TextPreProcessor(
#     normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
#         'time', 'url', 'date', 'number'],
#     annotate={"hashtag", "allcaps", "elongated", "repeated",
#         'emphasis', 'censored'},
#     fix_html=True,  # fix HTML tokens
#     segmenter="twitter", 
#     corrector="twitter", 
#     unpack_hashtags=True,  # perform word segmentation on hashtags
#     unpack_contractions=True,  # Unpack contractions (can't -> can not)
#     spell_correct_elong=False,  # spell correction for elongated words
#     tokenizer=SocialTokenizer(lowercase=True).tokenize,
#     dicts=[emoticons]
# )

# f = open('tweets.txt','r')
# f1 = open('tokenized_tweets1.txt','w')
# for line in f:
# 	a = line.split()
# 	b = a[-2]
# 	c = a[-1]
# 	a.pop()
# 	a.pop()
# 	a = ' '.join(a)
# 	a = ' '.join(text_processor.pre_process_doc(a))
# 	# print(a,c)
# 	f1.write(a+' '+c+'\n')


from itertools import izip
c = 0
with open("tokenized_tweets.txt") as textfile1, open("tokenized_tweets1.txt") as textfile2: 
    for x, y in izip(textfile1, textfile2):
        x = x.strip()
        y = y.strip()
        if x!=y:
            print '====================================='
            print x
            print y
            c=c+1
print c
