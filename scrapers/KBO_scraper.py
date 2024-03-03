from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
import requests
from typing import List
import sys
import numpy as np
import os
from datetime import datetime
from selenium.common.exceptions import NoSuchElementException

def choose_location(locations):
    print(f"There are multiple locations for the specified location:")
    for i, loc in enumerate(locations, start=1):
        print(f"{i}: {loc}")

    while True:
        try:
            choice = int(input("Enter the number corresponding to the correct location: "))
            if 1 <= choice <= len(locations):
                return locations[choice - 1]
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
def next_page_exists(driver):
    try:
        driver.find_element(By.LINK_TEXT, "Volgende")
        return True
    except NoSuchElementException:
        return False
    
def download_pdf(pdf_url, filename, driver):
    try:
        response = requests.get(pdf_url)
        with open(filename, 'wb') as file:
            file.write(response.content)
    except Exception as e:
        print(f"no pdf file found: {e}")
        driver.close()
        print("driver is closed")
    
def handle_location_input(driver, option, municipality_input):
    if option != 3:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.autocomplete ul li.selected')))
        location_options = driver.find_elements(By.CSS_SELECTOR, '.autocomplete ul li')
        if len(location_options) > 1:
            selected_location = choose_location([option.text for option in location_options])
            municipality_input.clear()
            municipality_input.send_keys(selected_location)   

def scrape_data(nace_code, location, temp_pdf_filename, option, driver): 

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'nacecodes')))

        nace_input = driver.find_element(By.ID, 'nacecodes')
        
        if not option:
            municipality_selector = ('gem', 'gemeente1')
            print('city only')
        elif option:
            municipality_selector = ('gemb', 'gemeente0')
            print('city and neighbours')
        elif option == 3:
            municipality_selector = ('post', 'postnummer1')

        municipality_radio, municipality_input = [driver.find_element(By.ID, id) for id in municipality_selector]
        search_button = driver.find_element(By.NAME, 'actionLu')

        nace_input.send_keys(nace_code)
        municipality_radio.click()
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, municipality_input.get_attribute('id'))))
        municipality_input.send_keys(location)

        handle_location_input(driver, option, municipality_input)

        search_button.click()
        pdf_url = driver.find_element(By.XPATH, '//a[contains(@href, "activiteitenlijst.pdf")]').get_attribute('href')
        download_pdf(pdf_url, temp_pdf_filename, driver)

        ondernemingsnummers = extract_ondernemingsnummers(driver)
        company_data_list = []
        for num in ondernemingsnummers:
            company_data = check_activities(num, nace_code)
            if company_data:
                company_data_list.append(company_data)

        company_data_array = np.array(company_data_list)
        return company_data_array

    except TimeoutException:
        print(f"No results found for location: {location} or NACE-code: {nace_code} is incorrect.")
        return None

def extract_ondernemingsnummers(driver):
    try: 
        ondernemingsnummers = []
        while True:
            rows = driver.find_elements(By.XPATH, '//tbody/tr')
            for row in rows:
                ondernemingsnummer_element = row.find_element(By.XPATH, './td[3]/a')
                ondernemingsnummer = ondernemingsnummer_element.text.strip()
                ondernemingsnummers.append(ondernemingsnummer)

            if not next_page_exists(driver):
                break

            driver.find_element(By.LINK_TEXT, "Volgende").click()
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//tbody/tr')))
        
        return ondernemingsnummers
    except Exception as e:
        print(f"error when retrieving 'ondernemersnummers': {e}")
        driver.close()
        print("driver is closed")

def check_activities(ondernemingsnummer, nace_code):
    
    # The URL of the API endpoint
    base_url = "https://kbo.party/api/v1/enterprise/{}?key=NZD7E8OAEjnsoqHa"

    # The headers to be sent with the request
    headers = {
        "Accept": "application/json"
    }
    response = requests.get(base_url.format(ondernemingsnummer), headers=headers)

    if response.status_code == 200:
        data = response.json()
        for activity in data.get('activities', []):
            if activity.get('nace') == int(nace_code):
                for address in data.get("addresses", []):
                    company_name = data.get('name')
                    company_nr = ondernemingsnummer
                    company_addr = f"{address.get('street')} {address.get('housenumber')}, {address.get('zip')} {address.get('municipality')}, Belgium"
                    return [company_name, company_addr]
        return []            

def kbo_scraper(locations: List[str], nace_codes: List[str], option: bool, driver):
    start_url = "https://kbopub.economie.fgov.be/kbopub/zoekactiviteitform.html"
    all_company_data = []
    for location in locations:
        for nace_code in nace_codes:
            try:
                driver.get(start_url)
                try:
                    temp_pdf_filename = f'temp_{nace_code}_{location}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pdf'
                    company_data_array = scrape_data(nace_code, location, temp_pdf_filename, option, driver)
                    all_company_data.extend(company_data_array)
                finally:
                    os.remove(temp_pdf_filename)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
    return all_company_data

def print_usage_and_exit():
    print("Usage: python KBO_scraper.py <arg1> <arg2> <arg3>")
    print("1: search for 'gemeente'")
    print("2: search for 'gemeente & buurgemeenten'")
    print("3: search for 'postcode'")
    print("<arg2>: path of the NACE code text file")
    print("<arg3>: path of the location text file")
    sys.exit(1)

def setup_chrome_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--profile-directory=Default")
    chrome_options.add_argument("--disable-plugins-discovery")
    chrome_options.add_argument("--incognito")
    chrome_options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2,})
    driver = webdriver.Chrome(chrome_options)
    driver.delete_all_cookies()
    return driver


def main(locations, option, nace_codes):
    driver = setup_chrome_driver() 
    try:
        all_company_data = kbo_scraper(locations, nace_codes, option, driver)
        return all_company_data
    except Exception as e:
        print(f"Critical error, stopping the scraper: {e}")
    finally:
        # driver.quit()
        driver.close()
        print("driver is closed")


