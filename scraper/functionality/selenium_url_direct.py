# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 19:39:26 2021

@author: natha
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# This script takes a given URL and performs a Selenium scrape on it.
url = "https://twitter.com/ASBMilitary/status/1466331358339469319"

# Set up scraper
chrome_options = Options()
chrome_options.add_argument('--enable-javascript')
# or? chrome_options.add_argument('javascript.enabled', True)
driver = webdriver.Chrome(options=chrome_options)
driver.get(url)
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Allow page to load
time.sleep(1)

#To get the content of the tweet in the URL
hold = []
title = soup.get_text()
hold.append(title)