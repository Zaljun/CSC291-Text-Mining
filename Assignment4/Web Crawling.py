from bs4 import BeautifulSoup
import requests

response = requests.get('https://www.tensorflow.org/get_started/mnist/beginners')
soup = BeautifulSoup(response.text)
#print(soup.prettify())

# extract links from the web-page
links_data = soup.find_all('a')  # finds all tags with 'a'
title = soup.find('title').get_text()  
text  = soup.get_text()
dic   = {}
dic[title] = text

#for each link in root page, get response and extract its title and text
for link in links_data:
    print(link.get('href'))
    r     = requests.get(link.get('href'))
    title = r.find('title').get_text()
    text  = r.get_text()
    dic[title] = text

'''
Problem: some of the links using "get('href')" don't have 'http' or 'https' 
so that url is invalid when requesting
'''

'''
10.8
Can count clicks, higher rank pages tend to have greater number.
Can record time spent in that page. Usually people spend more time 
on their desired page. However, it could be opposite when users
searching some fact.

10.10
If we add a new page to the web, what happens to other existing PageRank
scores? 
If the new page is isolated, then the scores of others will decrease because
the new page sparse the probability of randomly clicking pages.

10.12
A query where a high-scoring authority page could be a desired document:
    a title of some essay or book
A query where a high-scoring hub page could be a desired document:
    a forum name such as reddit
'''