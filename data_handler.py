import codecs

def get_data():
	tweets = []
	with codecs.open('./'+'tweets.txt', 'r', encoding='utf-8') as f:
		data = f.readlines()
	for line in data:
		a = line.split()
		b = a[-2]
		c = a[-1]
		a.pop()
		a.pop()
		a = ' '.join(a)
		tweets.append({
                'text': a.lower(),
                'label': c,
                'user': b
                })
	return tweets
