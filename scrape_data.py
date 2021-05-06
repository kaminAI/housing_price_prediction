from joblib import Parallel, delayed
from tqdm import tqdm
import time
from random import randint
from datetime import datetime as dt
import pandas as pd
import logging
from src.data import get_sold_item_hrefs, get_property_details

def scrape_booli_paralell(page, L):
    url = f'https://www.booli.se/slutpriser/stockholms+innerstad/143?objectType=L%C3%A4genhet&sort=soldDate&page={page}'
    item_hrefs = get_sold_item_hrefs(url)
    
    for href in tqdm(item_hrefs, desc=f'items page {page}', leave=False):
        d = get_property_details(href)
        
        if bool(d): # only append non empty dict
            L.append(d)
        time.sleep(3)

if __name__ == "__main__":
    
    today = dt.now().date().strftime('%Y%m%d')
    n_pages = 1000 # note this might vary over time, max is 1000
    file_name = f"data/booli_{n_pages}_pages_raw_{today}"
    
    logging.basicConfig(
        filename=file_name + '.log',
        filemode='w',
        format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
    )

    L = []
    n_jobs = 2 # if changing this change sleep in scraper functions due to server timout 
    with Parallel(n_jobs=n_jobs, backend='threading') as p:

        # note tqdm will fire when process starts with joblib,
        # hence this progressbar is sligly missleading at start / end
        p(delayed(scrape_booli_paralell)(page, L) for page in tqdm(range(1, n_pages +1), desc='pages'))
        
    pd.DataFrame(L).to_csv(file_name + '.csv', index=False)
