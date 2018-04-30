import twitter
import time
import sys
import requests
from requests_oauthlib import OAuth1
reload(sys)
sys.setdefaultencoding('utf8')

# api = twitter.Api(consumer_key='W3V5kR3IlmkAOSCURflUC6FCv',
#   consumer_secret='0Z1YTzrr3lkxLu20bDJrEG8Nd0j1uLwmBKWqAnax8gNCvNfKgt',
#   access_token_key='982756759-SLdDdTeF574x5N8ierKubo6YHLycIecwBMahK4bX',
#   access_token_secret='6T3ZSN39paZZ6Pi23Vshxj6oT5h9eBapmsgcfggxjDtQ1')

f = open('NAACL_SRW_2016.csv','r')
f1 = open('full_tweets.txt','w')
auth = OAuth1('W3V5kR3IlmkAOSCURflUC6FCv','0Z1YTzrr3lkxLu20bDJrEG8Nd0j1uLwmBKWqAnax8gNCvNfKgt','982756759-SLdDdTeF574x5N8ierKubo6YHLycIecwBMahK4bX','6T3ZSN39paZZ6Pi23Vshxj6oT5h9eBapmsgcfggxjDtQ1')
c = 0
calls = 0
for line in f:
	try:
		l = line.split(',')
		url = 'https://api.twitter.com/1.1/statuses/show/'+l[0]+'.json'
		tweet = requests.get(url, auth=auth)
		tweet = tweet.json()
		if 'errors' in tweet:
			print 'Error for id ---> ',l[0]
			print tweet['errors']
		else:
			# print tweet['in_reply_to_status_id'],tweet['']
			# text = tweet['text'].replace('\n',' ')
			# user = tweet['user']['id']
			# print text , user
			f1.write(str(tweet)+'\n')
		time.sleep(1)
	except Exception as e:
		print e , l[0]
		c = c+1
print c

