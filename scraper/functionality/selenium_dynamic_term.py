# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:47:04 2021

@author: natha
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import time
import random

# This script will find tweets relating to a search query 
# It is dynamic - it will dynamically scroll.

url = "https://twitter.com/search?q=ied"

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
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    #To get the content of the tweet in the URL
    title = soup.get_text()
    get_url = driver.current_url
    content_hold.append(title)
    url_hold.append(get_url)
    
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
    if (scroll_count) > 500000:
        break
    
    
    
