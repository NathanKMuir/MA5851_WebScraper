# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:36:20 2021

@author: natha
"""

from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome()

driver.get('https://twitter.com/IntelTweet/status/1467625759699259392')

soup = BeautifulSoup(driver.page_source, 'html.parser')

#To get the title of the page
title = soup.get_text()
hold = []
hold.append(title)
print(re.findall(r'"(.*?)"', text1))