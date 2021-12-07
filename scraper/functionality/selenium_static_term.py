# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:01:16 2021

@author: natha
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import time
import random

# This script will find tweets relating to a search query 
# It is static - it will not dynamically scroll.

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

# Allow page to load
time.sleep(5)

# Parse page HTML data
soup = BeautifulSoup(driver.page_source, 'html.parser')

#To get the content of the tweet in the URL
title = soup.get_text()
get_url = driver.current_url
content_hold.append(title)
url_hold.append(get_url)