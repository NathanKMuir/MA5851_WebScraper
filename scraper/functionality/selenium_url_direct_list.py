# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:03:07 2021

@author: natha
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import time

# This script takes a given list of URLs and performs a Selenium scrape on it.
urls = ["https://twitter.com/ASBMilitary/status/1466331358339469319",
        "https://twitter.com/DeGreenTribe/status/1468157962111881220",
        "https://twitter.com/pakhead/status/1468156397200101376"]

# Define user agent


# Set up scraper options
options = webdriver.ChromeOptions()
options.add_argument("--enable-javascript")
driver = webdriver.Chrome(chrome_options=options)

# Initialise hold list
hold = []

# Create loop
for url in urls:
    
    # Execute browser
    driver.get(url)
    
    # Allow page to load
    time.sleep(5)
    
    # Parse page HTML data
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    #To get the content of the tweet in the URL
    title = soup.get_text()
    hold.append(title)
