# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:47:04 2021

@author: natha
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import time
import random
import pandas as pd

# This script will find tweets relating to a search query 
# It is dynamic - it will dynamically scroll.

# https://twitter.com/search?q=ied%20until%3A2021-12-01%20since%3A2021-09-01%20-filter%3Areplies&src=typed_query&f=live
#url = "https://twitter.com/search?q=ied"
url = "https://twitter.com/search?q=ied%20until%3A2021-11-30%20since%3A2021-09-01%20-filter%3Areplies&src=typed_query&f=live"

# Define user agent list
user_agent_list  = ['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',]

# Set up scraper options
options = webdriver.ChromeOptions()
options.add_argument("--enable-javascript")

# Initialise hold list
content_hold = []
url_hold = []

# Randomise  user agent
user_agent = random.choice(user_agent_list)
options.add_argument(f'user-agent={user_agent}')

# Execute browser
driver = webdriver.Chrome(chrome_options=options)
driver.get(url)

# Allow page to open
time.sleep(3)

# Set dynamic scrolling parameters 
scroll_pause_time = 1 
screen_height = driver.execute_script("return window.screen.height;")   # get the screen height of the page
i = 1
scroll_count = 1

while True:
    
    # Parse page HTML data
    soup = BeautifulSoup(driver.page_source, 'lxml')

    tweets = []
    for li in soup.find_all("li", class_='js-stream-item'):
 
        # If our li doesn't have a tweet-id, we skip it as it's not going to be a tweet.
        if 'data-item-id' not in li.attrs:
            continue
 
        else:
            tweet = {
                'tweet_id': li['data-item-id'],
                'text': None,
                'user_id': None,
                'user_screen_name': None,
                'user_name': None,
                'created_at': None,
                'retweets': 0,
                'likes': 0,
                'replies': 0
            }
 
            # Tweet Text
            text_p = li.find("p", class_="tweet-text")
            if text_p is not None:
                tweet['text'] = text_p.get_text()
 
            # Tweet User ID, User Screen Name, User Name
            user_details_div = li.find("div", class_="tweet")
            if user_details_div is not None:
                tweet['user_id'] = user_details_div['data-user-id']
                tweet['user_screen_name'] = user_details_div['data-screen-name']
                tweet['user_name'] = user_details_div['data-name']
 
            # Tweet date
            date_span = li.find("span", class_="_timestamp")
            if date_span is not None:
                tweet['created_at'] = float(date_span['data-time-ms'])
 
            # Tweet Retweets
            retweet_span = li.select("span.ProfileTweet-action--retweet > span.ProfileTweet-actionCount")
            if retweet_span is not None and len(retweet_span) > 0:
                tweet['retweets'] = int(retweet_span[0]['data-tweet-stat-count'])
 
            # Tweet Likes
            like_span = li.select("span.ProfileTweet-action--favorite > span.ProfileTweet-actionCount")
            if like_span is not None and len(like_span) > 0:
                tweet['likes'] = int(like_span[0]['data-tweet-stat-count'])
 
            # Tweet Replies
            reply_span = li.select("span.ProfileTweet-action--reply > span.ProfileTweet-actionCount")
            if reply_span is not None and len(reply_span) > 0:
                tweet['replies'] = int(reply_span[0]['data-tweet-stat-count'])
 
            tweets.append(tweet)
    
    # scroll one screen height each time
    driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))  
    i += 1
    scroll_count += 1
    time.sleep(scroll_pause_time)
    # update scroll height each time after scrolled, as the scroll height can change after we scrolled the page
    scroll_height = driver.execute_script("return document.body.scrollHeight;")  
    
    # Break the loop when the height we need to scroll to is larger than the total scroll height
    if (screen_height) * i > scroll_height:
        break
    
content_df = pd.DataFrame(content_hold)    
#content_df.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\content_df.csv")    
