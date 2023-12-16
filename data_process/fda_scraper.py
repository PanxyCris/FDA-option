import json
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By

def scrape_fda():
    """
    scrape DFA pharmacy and ticker data to json file
    """
    dic = {}
    chrome_options = webdriver.ChromeOptions()
    with webdriver.Chrome(options=chrome_options) as driver:
        driver.get(
            "https://calendar.google.com/calendar/u/0/embed?showTitle=0&height=600&wkst=1&bgcolor=%23FFFFFF&src=evgohovm2m3tuvqakdf4hfeq84@group.calendar.google.com&color=%23711616&src=5dso8589486irtj53sdkr4h6ek@group.calendar.google.com&color=%23182C57&ctz=America/Los_Angeles")
        driver.find_element(By.ID, 'tab-controller-container-agenda').click()
        sleep(3)
        driver.execute_script("window.scrollBy(0, -500)")
        sleep(60)
        children = driver.find_element(By.ID, 'eventContainer1').find_elements(By.CLASS_NAME, "day")
        for child in children:
            date_str = child.find_element(By.CLASS_NAME, 'date-label').text
            for event in child.find_elements(By.CLASS_NAME, 'event'):
                content = event.find_element(By.CLASS_NAME, 'event-title')
                ticker = content.text.split()[0]
                if ticker not in dic:
                    dic[ticker] = []
                dic[ticker].append(date_str)
    with open('files/datasource/pharm.json', 'w') as jsonfile:
        json.dump(dic, jsonfile)