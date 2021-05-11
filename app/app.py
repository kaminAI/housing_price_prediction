import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import os; import sys; from pathlib import Path
import torch
root = Path(__file__).parents[1]
sys.path.append(str(root))

from src.data import preprocess_inference
from src.model import Regressor

# web scraping booli
@st.cache
def load_data(url):

    page = requests.get(url=url)
    soup = BeautifulSoup(page.content, "lxml")

    property_details_div = soup.find_all('div', attrs={'class': 'DfWRI _1Pdm1 _2zXIc sVQc-'})

    d={}
    for div in property_details_div:
        item = div.find_all('div')
        key, value = item[0].text, item[1].text
        d[key] = value

    property_details_h1 = soup.find('h1', attrs={'class': 'lzFZY _10w08'})
    d['adress'] = property_details_h1.text

    property_details_h2 = soup.find('h2', attrs={'class': 'lzFZY _10w08'})
    d['Utropspris'] = property_details_h2.text

    

    property_details_h4 = soup.find_all('h4', attrs={'class': '_1544W _10w08'})
    
    try: # not always available
        d['area_and_n_rooms'] = property_details_h4[0].text 
        d['location'] = property_details_h4[1].text 

    except IndexError:
        pass

    d['url'] = url

    for key in ["Utropspris", "Avgift", "Driftskostnad", "Våning","Byggår","area_and_n_rooms", "location"]:
        if key not in d.keys():
            d[key]=None

    return d


st.title('Realestate closing price predictions')

st.markdown("""
This app predicts the closing price (after the bidding process) of Swedish appartements and houses.

Usage:

* **Step 1:** find the ad for the appartement on https://www.booli.se/ and copy the url
* **Step 2:** paste the url in the box below 
* **Step 3:** press calculate the get and estimate of the winning bid
""")



url = st.text_input(label='The URL link', value='https://www.booli.se', help='paste link to the ad from www.booli.se')

valid_url  = 'booli.se/annons/' in url or 'booli.se/bostad/' in url

if valid_url:


    r = requests.get(url)
    text = r.text
    index = index = text.find('"og:image" content=') + len('"og:image" content=') +1
    image_url =  text[index:index+100].split('"')[0]
    st.image(image_url, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    data  = load_data(url)

    df = pd.DataFrame(data, index=[0])

    st.dataframe(df)

    df_inference  = preprocess_inference(df)

    st.dataframe(df_inference)


    model_loaded = Regressor().load_from_checkpoint(root / 'model_checkpoints' / 'epoch=143-avg_rmse_val=7.32.ckpt')
    model_loaded.eval()

    with torch.no_grad():
        pred = model_loaded(torch.tensor(df_inference.values).float())

    st.dataframe(pred.numpy())








