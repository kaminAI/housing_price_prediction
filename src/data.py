import logging
from fake_headers import Headers
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
from category_encoders.one_hot import OneHotEncoder
import joblib


class StatusCodeError(Exception):
    """HTTP status code error"""

def get_sold_item_hrefs(url, try_num=1):
    h = Headers(headers=False)

    try:
        page = requests.get(url=url, headers=h.generate())
        if page.status_code != 200:
            raise StatusCodeError

    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout, StatusCodeError):
        logging.info(f'try {try_num} failed for {url}')
        try_num = try_num + 1
        time.sleep(5 * 3) # server request timeout of 5 sec, multipy by n jobs + 1 
 
        if try_num <= 3:
            return get_sold_item_hrefs(url, try_num)

        else:
            logging.info(f'ABORTING, try {try_num} failed for {url}')
            return []

    else:
        soup = BeautifulSoup(page.content, "lxml")
        item_list = soup.find('div', attrs={'class':'_2m6km uC2y2 _3oDFL'}).find_all('a')
        item_hrefs = [item['href'] for item in item_list]
    
        return item_hrefs

def get_property_details(href, try_num=1):
    url = f'https://www.booli.se{href}'
    h = Headers(headers=False)

    try:
        page = requests.get(url=url, headers=h.generate())
        if page.status_code != 200:
            raise StatusCodeError

    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout, StatusCodeError):
        logging.info(f'try {try_num} failed for {url}')
        try_num = try_num + 1
        time.sleep(5 * 3) # server request timeout of 5 sec, multipy by n jobs + 1 

        if try_num <= 3:
            return get_property_details(href, try_num)

        else:
            logging.info(f'ABORTING, try {try_num} failed for {url}')
            return {}

    else:
        soup = BeautifulSoup(page.content, "lxml")

        try: 
            marker = soup.find('span', attrs={'class': 'FmduO _1JxpX'}).text
        except AttributeError:
            marker = 'undefined'

        if marker == 'Slutpris':
            property_details_div = soup.find_all('div', attrs={'class': 'DfWRI _1Pdm1 _2zXIc sVQc-'})

            d={}
            for div in property_details_div:
                item = div.find_all('div')
                key, value = item[0].text, item[1].text
                d[key] = value

            property_details_h1 = soup.find('h1', attrs={'class': 'lzFZY _10w08'})
            d['adress'] = property_details_h1.text

            property_details_h4 = soup.find_all('h4', attrs={'class': '_1544W _10w08'})
            try: # not always available
                d['area_and_n_rooms'] = property_details_h4[0].text # its not so good having this joined, manual cleaning needed when e.g area not existing
                d['location'] = property_details_h4[1].text 

            except IndexError:
                pass

            d['url'] = url
            
            return d

        else:
            return {}

def post_process(df):
    
    columns_mapping = {
            "Utropspris": "price_ask",
            "Prisutveckling": "price_gain",
            "Såld eller borttagen": "sell_date", # only used to sort
            "Avgift": "montly_fee_cost",
            "Driftskostnad": "montly_operating_cost",
            "Våning": "floor",
            "Byggår": "construction_year",
            "area_and_n_rooms": "area_and_n_rooms",
            "location": "location"
        }

    df = df.rename(
        columns=columns_mapping
    )

    # a lot of random columns can follow with scraper, remove these
    columns_to_keep = columns_mapping.values()
    df = df[columns_to_keep]
    
    # vectorized cleaning
    df['price_ask'] = df['price_ask'].str.strip(' kr').str.replace(' ', '').astype(float)
    df['price_gain'] = df['price_gain'].str.split('(').str[-1].str.strip('+/%)').astype(float)
    df['sell_date'] = pd.to_datetime(df['sell_date'])
    df['montly_fee_cost'] = df['montly_fee_cost'].str.strip(' kr/mån').str.replace(' ', '').astype(float)
    df['montly_operating_cost'] = df['montly_operating_cost'].str.strip(' kr/mån').str.replace(' ', '').astype(float)
    df['floor'] = df['floor'].str.strip(' tr').str.replace('½','.5').str.replace('BV','0').astype(float)
    df['construction_year'] = df ['construction_year'].astype(float)
    
    df[['area', 'n_rooms']] = df['area_and_n_rooms'].str.split(',', expand=True)
    df.drop(columns=['area_and_n_rooms'], inplace=True)
    df['area'] = df['area'].str.strip(' m²').str.replace('½','.5').astype(float)
    df['n_rooms'] = df['n_rooms'].str.strip(' rum').str.replace('½','.5').astype(float)

    df['list_price_per_m2'] = df.price_ask/df.area

    # handle categorical features
    df['location'] = df['location'].str.replace('Lägenhet, ', '').str.lower() # don't use strip here, will remove e.g. gärd[et]
    top_X_location = df['location'].value_counts()[:5].index
    joblib.dump(top_X_location, '../data/top_X_location.pkl') # save mapper
    df.loc[~df['location'].isin(top_X_location), 'location'] = 'other'
    

    ohe = OneHotEncoder(use_cat_names=True, )
    dummies = ohe.fit_transform(df[['location']])
    joblib.dump(ohe, '../data/location_one_hot_encoder.pkl') # save transformer

    df = df.drop('location', axis = 1)
    df = df.join(dummies)

    # sort by sell date old to recent
    df.sort_values(by='sell_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
     
    return df
