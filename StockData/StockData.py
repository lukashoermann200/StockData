#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 21:36:35 2023

@author: lukas
"""

#this script is designed to scrape its financial statements
#yahoo finance only contains the recent 5 year
#macrotrends can trace back to 2005 if applicable
import numpy as np
import re
import json
import pandas as pd
import requests
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import time
import yfinance as yf
import cpi
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import ImageFont
from bs4 import BeautifulSoup


ticker_dict = {'VLPNY' : 'voestalpine-ag',
               'EADSY' : 'airbus-group',
               'GLNCY' : 'glencore',
               'BAYRY' : 'bayer',
               'WCH' : 'WCH',
               'SHEL' : 'shell',
               'AMD' : 'amd',
               'NVDA' : 'nvidia',
               'PYPL' : 'paypal-holdings',
               'V' : 'visa',
               'LIN' : 'linde',
               'PFE' : 'pfizer',
               'JNJ' : 'johnson-johnson',
               'LLY' : 'eli-lilly',
               'NET' : 'cloudflare',
               'RIO' : 'rio-tinto',
               'CAT' : 'caterpillar',
               }
other_tickers = {'VLPNY' : ['VLPNY', 'VOE.VI'],
                 'EADSY' : ['AIR.PA', 'EADSY'],
                 'GLNCY' : ['GLNCY'],
                 'BAYRY' : ['BAYRY'],
                 'WCH' : ['WCH'],
                 'SHEL' : ['shell'],
                 'AMD' : ['amd'],}


#simply scrape
def scrape(url, **kwargs):
    
    session = requests.Session()
    session.headers.update(
            {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'})
    
    response=session.get(url, **kwargs)

    return response


#create dataframe
def etl(response):

    #regex to find the data
    num = re.findall('(?<=div\>\"\,)[0-9\.\"\:\-\, ]*', response.text)
    text = re.findall('(?<=s\: \')\S+(?=\'\, freq)', response.text)
    
    #print(num)
    #print(text)

    #convert text to dict via json
    dicts=[json.loads('{'+i+'}') for i in num]

    #create dataframe
    df=pd.DataFrame()
    for ind, val in enumerate(text):
        df[val]=dicts[ind].values()    
        df.index=dicts[ind].keys()
    
    return df


def getUrl(ticker):
    if ticker in ticker_dict:
        url = 'https://www.macrotrends.net/'
        return_url = url+"stocks/charts/"+ticker+"/"+ticker_dict[ticker]+"/"
    
    else:
        return_url = _getUrl(ticker)
    
    print(return_url)
    
    return return_url


def _getUrl(ticker):
    service = Service(executable_path='geckodriver')
    options = webdriver.FirefoxOptions()
    #options.headless = True
    #options.add_argument('--headless')
    driver = webdriver.Firefox(service=service, options=options)
    
    #driver = webdriver.Firefox(options=opts, executable_path='/home/lukas/Dokumente/Stocks/geckodriver')
    
    #driver = webdriver.Firefox(executable_path='/home/lukas/Downloads/geckodriver')
    url = 'https://www.macrotrends.net/'
    driver.get(url)
    box = driver.find_element(By.CSS_SELECTOR, ".js-typeahead")
    box.send_keys(ticker)
    time.sleep(5)
    box.send_keys(Keys.DOWN, Keys.ENTER)
    time.sleep(5)
    geturl = driver.current_url
    time.sleep(5)
    driver.quit()
    
    return_url = ''
    
    #check if the ticker is available in MacroTrends
    if "stocks" in geturl:
        geturlsp = geturl.split("/", 10)
        return_url = url+"stocks/charts/"+geturlsp[5]+"/"+geturlsp[6]+"/"
    
    print(return_url)
    
    return return_url


def getStockPriceHistory(ticker, frequency='a'):
    
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    
    data = yf.Ticker(ticker)
    #todays_data = stock.history(period='1d')
    
    stock_price_history = data.history(start="2010-01-01",  end=now)
    
    #print(data.financials)
    
    return stock_price_history


def getFinancials(ticker, stat_name, frequency='a'):
    if frequency == 'q':
        frequency_string = '?freq=Q'
    else:
        frequency_string = '?freq=A'
    
    url_base = getUrl(ticker)
    url = url_base + stat_name + frequency_string
    response = scrape(url)
    df = etl(response)
    
    return df


def getFinancialStatements(ticker, frequency='a'):
    return getFinancials(ticker, 'financial-statements', frequency=frequency)


def getBalanceSheet(ticker, frequency='a'):
    return getFinancials(ticker, 'balance-sheet', frequency=frequency)


def getCashFlowStatement(ticker, frequency='a'):
    return getFinancials(ticker, 'cash-flow-statement', frequency=frequency)


def getFinancialRatios(ticker, frequency='a'):
    return getFinancials(ticker, 'financial-ratios', frequency=frequency)


def getDescription(ticker):
    
    url_base = getUrl(ticker)
    url = url_base + 'stock-price-history'
    response = scrape(url)
    text = re.findall('\>(.*?)\<', response.text)
    
    max_len = 0
    return_text = ''
    for t in text:
        if len(t) > max_len:
            max_len = len(t)
            return_text = t
    
    return return_text


def getRatio(ticker, ratio, frequency='a'):
    if frequency == 'q':
        frequency_string = '?freq=Q'
    else:
        frequency_string = '?freq=A'
    
    table_text = None
    
    for i in range(15):
        url_base = getUrl(ticker)
        url = url_base + ratio
        
        print(url)
        
        response = scrape(url)
        
        input_text = response.text
        input_text = input_text.replace("\n","")
        
        table_text = getTables(input_text)
        if len(table_text) > 0:
            break
    
    #print(table_text)
    
    table_text = table_text[0]
    
    numbers = getNumbers(table_text)
    columns = findPattern(table_text, '">', '(.*?)', '</th>')[2:]
    
    indices = []
    data = []
    
    if isinstance(numbers[0], str):
        for n in numbers:
            if isinstance(n, str):
                indices.append(n)
            else:
                data.append(n)
    
    data = np.array(data).reshape((len(indices), len(columns)))
    
    df = pd.DataFrame(data, indices, columns)
    
    return df


def getAnalysis(ticker):
    """
    (?:user\/|id=)\K\d+

    (?:         # Start of a non-capturing group
    user\/      # Match `user/`
    |           # Or
    id=         # `id=`
    )           # End of non-capturing group
    \K\d+       # Forget matched strings then match digits

    """
    url_base = 'https://finance.yahoo.com/quote/'
    url = url_base + ticker + '/analysis' + '?p=' + ticker
    response = scrape(url)
    
    tables = getTables(response.text)
    
    analysis_dict = {}
    
    if not len(tables) == 0:
        for t in tables[:-1]:
            #headers = getTableHeader(t, '<span>', '</span>')
            numbers = getNumbers(t)
            
            columns_0 = findPattern(t, '<th', '(.*?)', '</th>')
            
            columns_1 = []
            for c in columns_0:
                columns_1.append( findPattern(c, '<span>', '[0-9a-zA-Z\.\s?\(\)\/]*', '</span>')[0] )
            
            table_name = columns_1[0]
            columns = columns_1[1:]
            
            indices = findPattern(t, '<span>', '[0-9a-zA-Z\.\s?\(\)\/\%]*', '</span></td>')
            indices = removePattern(indices, 'N/A')
            data = np.array(numbers).reshape((len(indices), len(columns)))
            
            df = pd.DataFrame(data, indices, columns)
            analysis_dict[table_name] = df
    
    return analysis_dict


def getTables(input_text):
    # get tables
    sub1 = '<table'
    sub2 = '</table>'
    pattern = '(.*?)' # '(.*)' -> all strings, ? shortest strings possible
    
    return findPattern(input_text, sub1, pattern, sub2)


def findPattern(input_text, sub1, pattern, sub2):
    s = '(?<='+str(re.escape(sub1))+')'
    e = '(?='+str(re.escape(sub2))+')'
    
    pattern = s+pattern+e
    
    return re.findall(pattern, input_text)
    
    
def getTableHeader(input_text, sub1, sub2):
    #pattern = s+'\S+'+e
    #pattern = s+'[0-9a-zA-Z\.\s?\(\)\/]*'+e
    pattern = '(.*?)'
    
    text = findPattern(input_text, sub1, pattern, sub1)
    
    text_new = []
    for t in text:
        if not t == 'N/A':
            text_new.append(t)
    
    return text_new


def removePattern(text, pattern):
    text_new = []
    for t in text:
        if not t == pattern:
            text_new.append(t)
    
    return text_new


def getNumbers(input_text):
    #sub1 = '<td class="Ta(end)">'
    #sub2 = '<td class="Ta(end) Py(10px)">'
    sub3 = '</td>'
    
    s = '(?<='+str(re.escape('">'))+')'
    e = '(?='+str(re.escape(sub3))+')'
    
    p1 = '[0-9A-Z\.\"\:\-\%\,\$ ]*'
    p2 = str(re.escape('<span>N/A</span>'))
    p3 = str(re.escape('N/A'))
    p4 = str(re.escape('nan'))
    p5 = str(re.escape('inf'))
    
    #num = re.findall(s+"(.*)"+e, response.text)
    
    #pattern = s+'[0-9\.\"\:\-\%\, ]*'+e
    pattern = s+'('+p1+'|'+p2+'|'+p3+'|'+p4+'|'+p5+')'+e
    
    num = re.findall(pattern, input_text)
    
    num_new = []
    for n in num:
        if len(n) == 0:
            num_new.append( 0.0 )
        elif n[-1] == 'M':
            num_new.append( np.float64(n[:-1]) * 1e6 )
        elif n[-1] == 'B':
            num_new.append( np.float64(n[:-1]) * 1e9 )
        elif n[-1] == '%':
            num_new.append( np.float64(n[:-1]) )
        elif n[0] == '$':
            num_new.append( np.float64(n[1:]) )
        elif 'N/A' in n:
            num_new.append( 0.0 )
        elif 'inf' in n:
            num_new.append( 0.0 )
        else:
            try:
                num_new.append( np.float64(n) )
            except Exception:
                num_new.append( n )
    
    return num_new


def main():
    
    #url='https://www.macrotrends.net/stocks/charts/AAPL/apple/financial-statements'
    url='https://www.macrotrends.net/stocks/charts/AAPL/apple/stock-price-history'
    response=scrape(url)
    
    print(response.text)
    
    df=etl(response)
    df.to_csv('aapl financial statements.csv')
    
    return


def saveResponse():
    
    #url='https://www.macrotrends.net/stocks/charts/AAPL/apple/financial-statements'
    url='https://www.macrotrends.net/stocks/charts/AAPL/apple/stock-price-history'
    response=scrape(url)
    
    file = open('response.txt', 'w')
    file.write(response.text)
    file.close()

    #text = re.findall('/^[a-zA-Z]{3,}$/', response.text)
    
    text = re.findall('\>(.*?)\<', response.text)
    
    max_len = 0
    retrun_text = ''
    for t in text:
        if len(t) > max_len:
            max_len = len(t)
            retrun_text = t
    
    
    print(retrun_text)
    

def getYearsFromDates(dates):
    
    years = np.zeros(len(dates), dtype=np.int64)
    
    for ind, date in enumerate(dates):
        year = np.datetime64(date, 'Y')
        years[ind] = int(str(year))
    
    return years


def adjustByCPI(dates, values):
    """
    Adjust data by consumer price index (inflation)

    Returns
    -------
    """
    cpi_adjusted_values = np.zeros_like(values)
    
    years = getYearsFromDates(dates)
    
    for ind in range(len(values)):
        try:
            cpi_adjusted_values[ind] = cpi.inflate(values[ind], years[ind])#, to=2023)
        except Exception:
            cpi_adjusted_values[ind] = values[ind]
            
    return cpi_adjusted_values


def textWrap(text, font, max_width):
    """Wrap text base on specified width. 
    This is to enable text of width more than the image width to be display
    nicely.
    @params:
        text: str
            text to wrap
        font: obj
            font of the text
        max_width: int
            width to split the text with
    @return
        lines: list[str]
            list of sub-strings
    """
    return_text = ''
    
    # If the text width is smaller than the image width, then no need to split
    # just add it to the line list and return
    if font.getsize(text)[0]  <= max_width:
        return_text = text
        
    else:
        #split the line by spaces to get words
        words = text.split(' ')
        
        line = ''
        # append every word to a line while its width is shorter than the image width
        for i in range(len(words)):
            if font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i]+ ' '
                
                if i == len(words) - 1:
                    return_text = return_text + line
                
            else:
                return_text = return_text + line + '\n'
                line = ''
                
    return return_text


def convertTofloat(array):
    array_new = np.zeros(len(array), dtype=np.float64)
    
    for ind, x in enumerate(array):
        if x == '':
            array_new[ind] = np.nan
        else:
            array_new[ind] = np.float64(x)
            
    return array_new


class StockData:
    
    def __init__(self, ticker, frequency):
        self.ticker = ticker
        self.frequency = frequency
        self.unit = 1e6

    
    def _setStockPrice(self):
        if not hasattr(self, 'stock_price'):
            self.stock_price = getStockPriceHistory(self.ticker)

    
    def _setFinancialStatements(self):
        if not hasattr(self, 'financial_statements'):
            self.financial_statements = getFinancialStatements(self.ticker, frequency=self.frequency)
    
    
    def _setFinancialRatios(self):
        if not hasattr(self, 'financial_ratios'):
            self.financial_ratios = getFinancialRatios(self.ticker, frequency=self.frequency)
    
    
    def _setPriceEarningsRatio(self):
        if not hasattr(self, 'price_earnings_ratio'):
            self.price_earnings_ratio = getRatio(self.ticker, 'pe-ratio', frequency=self.frequency)
    
    
    def _setPriceSalesRatio(self):
        if not hasattr(self, 'price_sales_ratio'):
            self.price_sales_ratio = getRatio(self.ticker, 'price-sales', frequency=self.frequency)
    
    
    def _setPriceBookRatio(self):
        if not hasattr(self, 'price_book_ratio'):
            self.price_book_ratio = getRatio(self.ticker, 'price-book', frequency=self.frequency)
            
            
    def _setPriceFreeCashFlowRatio(self):
        if not hasattr(self, 'price_free_cash_flow_ratio'):
            self.price_free_cash_flow_ratio = getRatio(self.ticker, 'price-fcf', frequency=self.frequency)


    def _setAnalysis(self):
        if not hasattr(self, 'analysis'):
            analysis = getAnalysis(self.ticker)
            
            if self.ticker in other_tickers:
                for ticker in other_tickers[self.ticker]:
                    analysis = getAnalysis(ticker)
                    
                    if len(analysis) > 0:
                        break
            
            self.analysis = analysis
        
    
    def _getStatFromFinancialStatements(self, stat_name):
        self._setFinancialStatements()
        
        dates = self.financial_statements.index.to_numpy()
        data = np.zeros(len(dates), dtype=np.float64)
        
        data_str = None
        if stat_name in self.financial_statements:
            data_str = self.financial_statements[stat_name].to_numpy()
        
        for ind in range(len(data)):
            try:
                data[ind] = np.float64(data_str[ind])
            except Exception:
                data[ind] = 0
        
        return dates[::-1], data[::-1]


    def getDescription(self):
        return getDescription(self.ticker)

    
    def getStockPrice(self, dates=None):
        self._setStockPrice()
        
        if not dates is None:
            data = []
            for date in dates:
                if date in self.stock_price.index:
                    data.append( self.stock_price.loc[date]['Close'] )
                else:
                    data.append( 0.0 )
            
            data = np.array(data, dtype=np.float64)
            
        else:
            data = self.stock_price['Close'].to_numpy()
            dates = self.stock_price.index.to_numpy()
        
        return dates, data
    

    def getRevenue(self, add_forcast=True):
        return self._getStatFromFinancialStatements('revenue')

    
    def getGrossProfit(self):
        return self._getStatFromFinancialStatements('gross-profit')


    def getNetIncome(self):
        return self._getStatFromFinancialStatements('net-income')

    
    def getEBIT(self):
        return self._getStatFromFinancialStatements('ebit')

    
    def getEBITDA(self):
        return self._getStatFromFinancialStatements('ebitda')

    
    def getEPSDiluted(self):
        return self._getStatFromFinancialStatements('eps-earnings-per-share-diluted')


    def getEPS(self):
        return self._getStatFromFinancialStatements('eps-basic-net-earnings-per-share')
    
    
    def getLiabilities(self):
        pass
    

    def getAlternativeRatio(self, dates, data, stat_name):
        
        alternative_calculation = False
        if len(data) == 0:
            alternative_calculation = True
        else:
            L = data == 0
            
            if np.sum(L) > len(data)*0.5:
                alternative_calculation = True
        
        if alternative_calculation:
            self._setFinancialRatios()
            
            dates = self.financial_ratios.index.to_numpy()[::-1]
            data_0 = self.financial_ratios[stat_name].to_numpy()[::-1]
            data_0 = convertTofloat(data_0)
        
            _, stock_price = self.getStockPrice(dates=dates)
        
            data = stock_price / data_0
        
        return dates, data

    
    def getPriceEarningsRatio(self):
        self._setPriceEarningsRatio()
        return self.price_earnings_ratio.index.to_numpy()[::-1], self.price_earnings_ratio['PE Ratio'].to_numpy()[::-1]
    
    
    def getPriceSalesRatio(self):
        self._setPriceSalesRatio()
        return self.price_sales_ratio.index.to_numpy()[::-1], self.price_sales_ratio['Price to Sales Ratio'].to_numpy()[::-1]
    
    
    def getPriceBookRatio(self):
        self._setPriceBookRatio()
        
        dates = self.price_book_ratio.index.to_numpy()[::-1]
        data = self.price_book_ratio['Price to Book Ratio'].to_numpy()[::-1]
        dates, data = self.getAlternativeRatio(dates, data, 'book-value-per-share')
        
        return dates, data
    
    
    def getPriceFreeCashFlowRatio(self):
        self._setPriceFreeCashFlowRatio()
        
        dates = self.price_free_cash_flow_ratio.index.to_numpy()[::-1]
        data = self.price_free_cash_flow_ratio['Price to FCF Ratio'].to_numpy()[::-1]
        dates, data = self.getAlternativeRatio(dates, data, 'free-cash-flow-per-share')
        
        return dates, data
    
    
    def _getForcast(self, stat, estimate_type):
        self._setAnalysis()
        
        data = [0.0, 0.0]
        if stat in self.analysis:
            df = self.analysis[stat]
            
            if self.frequency == 'q':
                data = df.loc[estimate_type, ['Current Qtr.', 'Next Qtr.']]
            else:
                data = df.loc[estimate_type, ['Current Year', 'Next Year']]
        
        date = []
        current_date = datetime.date.today()
        
        for i in range(2):
            if self.frequency == 'q':
                date_new = datetime.date(current_date.year, current_date.month, 31) + pd.DateOffset(months=3+3*i)
            else:
                date_new = datetime.date(current_date.year, 12, 31) + pd.DateOffset(years=i)
                
            date_new = date_new.strftime('%Y.%m.%d')
            date.append(date_new)
        
        date = np.array(date)
        data = np.array(data)
        
        return date, data
    
    
    def getRevenueForcastAverage(self):
        dates, data = self._getForcast('Revenue Estimate', 'Avg. Estimate')
        return dates, data / self.unit
    
    
    def getRevenueForcastHigh(self):
        dates, data = self._getForcast('Revenue Estimate', 'High Estimate')
        return dates, data / self.unit
    
    
    def getRevenueForcastLow(self):
        dates, data = self._getForcast('Revenue Estimate', 'Low Estimate')
        return dates, data / self.unit
    
    
    def getEBITForcastAverage(self):
        return self._getForcast('EBIT Estimate', 'Avg. Estimate')
    

    def getEPSForcastAverage(self):
        return self._getForcast('Earnings Estimate', 'Avg. Estimate')
    
    
    def getEPSForcastHigh(self):
        return self._getForcast('Earnings Estimate', 'High Estimate')
    
    
    def getEPSForcastLow(self):
        return self._getForcast('Earnings Estimate', 'Low Estimate')
    
    
    def _plotData(self,
                  getter_function,
                  axis=None,
                  adjust_by_cpi=False,
                  ylim=np.array([0, 1.1]),
                  plot_type='line',
                  color='C0',
                  alpha=1.0):
        
        dates, data = getter_function()
        
        L = np.isnan(data)
        data[L] = 0
        
        if axis is None:
            fig = plt.figure()
            axis = plt.gca()
        
        label = 'amount'
        if adjust_by_cpi:
            data = adjustByCPI(dates, data)
            label = 'amount adjusted by CPI'
        
        if plot_type == 'line':
            axis.plot(dates, data, color=color, alpha=alpha, label=label)
        elif plot_type == 'bar':
            axis.bar(dates, data, color=color, alpha=alpha, label=label)
        
        axis.xaxis.set_major_locator(plt.MaxNLocator(4))
        
        current_lim = axis.get_ylim()
        if not len(data) == 0:
            max_data = np.max(data)
            if not np.isnan(max_data) and not np.isinf(max_data) \
            and not max_data == 0.0 and max_data > current_lim[1]:
                plt.ylim(ylim*max_data)
        
    
    def plotStockPrice(self, axis=None, adjust_by_cpi=False, plot_type='line', color='C0'):
        
        self._plotData(self.getStockPrice,
                       axis=axis,
                       adjust_by_cpi=adjust_by_cpi,
                       ylim=np.array([0, 1.1]),
                       plot_type=plot_type,
                       color=color)
    
        plt.title('stock price / $', fontweight='semibold', fontsize=10)
        
    
    def plotRevenue(self, axis=None, adjust_by_cpi=False, plot_type='bar', color='C0'):
        
        self._plotData(self.getRevenue,
                       axis=axis,
                       adjust_by_cpi=adjust_by_cpi,
                       ylim=np.array([-1.1, 1.1]),
                       plot_type=plot_type,
                       color=color)
    
        plt.title('revenue / $', fontweight='semibold', fontsize=10)
    
    
    def plotEBIT(self, axis=None, adjust_by_cpi=False, plot_type='bar', color='C0'):
        
        self._plotData(self.getEBIT,
                       axis=axis,
                       adjust_by_cpi=adjust_by_cpi,
                       ylim=np.array([-1.1, 1.1]), 
                       plot_type=plot_type,
                       color=color)
    
        plt.title('EBIT / $', fontweight='semibold', fontsize=10)
    
    
    def plotEPS(self, axis=None, adjust_by_cpi=False, plot_type='bar', color='C0'):
        
        self._plotData(self.getEPS,
                       axis=axis,
                       adjust_by_cpi=adjust_by_cpi,
                       ylim=np.array([-1.1, 1.1]), 
                       plot_type=plot_type,
                       color=color)
    
        plt.title('EPS / $', fontweight='semibold', fontsize=10)
    
    
    def plotRevenueForcast(self, axis=None, plot_type='bar', color='C3'):
        
        self._plotData(self.getRevenueForcastAverage,
                       axis=axis,
                       plot_type=plot_type,
                       color=color,
                       alpha=1.0)


    def plotEBITForcast(self, axis=None, plot_type='bar', color='C3'):
        
        self._plotData(self.getEBITForcastAverage,
                       axis=axis,
                       plot_type=plot_type,
                       color=color,
                       alpha=1.0)


    def plotEPSForcast(self, axis=None, plot_type='bar', color='C3'):
        
        self._plotData(self.getEPSForcastAverage,
                       axis=axis,
                       plot_type=plot_type,
                       color=color,
                       alpha=1.0)
    
    
    def plotPriceEarningsRatio(self, axis=None, plot_type='bar', color='C0'):
        
        self._plotData(self.getPriceEarningsRatio,
                       axis=axis,
                       ylim=np.array([0.0, 1.1]), 
                       plot_type=plot_type,
                       color=color)
    
        plt.title('Price/Earnings Ratio', fontweight='semibold', fontsize=10)
        
        
    def plotPriceSalesRatio(self, axis=None, plot_type='bar', color='C0'):
        
        self._plotData(self.getPriceSalesRatio,
                       axis=axis,
                       ylim=np.array([0.0, 1.1]), 
                       plot_type=plot_type,
                       color=color)
    
        plt.title('Price/Sales Ratio', fontweight='semibold', fontsize=10)
    
    
    def plotPriceBookRatio(self, axis=None, plot_type='bar', color='C0'):
        
        self._plotData(self.getPriceBookRatio,
                       axis=axis,
                       ylim=np.array([0.0, 1.1]), 
                       plot_type=plot_type,
                       color=color)
    
        plt.title('Price/Book Ratio', fontweight='semibold', fontsize=10)
    
    
    def plotPriceFreeCashFlowRatio(self, axis=None, plot_type='bar', color='C0'):
        
        self._plotData(self.getPriceFreeCashFlowRatio,
                       axis=axis,
                       ylim=np.array([0.0, 1.1]), 
                       plot_type=plot_type,
                       color=color)
    
        plt.title('Price/Free Cash Flow Ratio', fontweight='semibold', fontsize=10)
    
    
    def getStockSheet(self):
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12
        LINE_WIDTH = 1.5
        MARKER_SIZE = 6
    
        plt.rc('font', **{'family':'sans-serif','sans-serif':['Liberation Sans']})
        plt.rc('font', family='sans-serif', weight='bold', size=MEDIUM_SIZE)
        
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)
        
        
        mpl.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        mpl.rcParams['mathtext.default'] = 'regular'
        mpl.rcParams['lines.linewidth'] = LINE_WIDTH
        mpl.rcParams['lines.markersize'] = MARKER_SIZE
        
        #https://www.macrotrends.net/stocks/charts/PYPL/paypal-holdings/pe-ratio
        
        fig = plt.figure(figsize=(8.27, 11.69))
        
        ax0 = fig.add_axes([0.05, 0.95, 0.9, 0.01])
        plt.axis('off')
        plt.title(self.ticker)
        description_0 = self.getDescription()
        
        # Create font object with the font file and specify desired size
        # Font style is `arial` and font size is 20
        font_path = 'font/arialbd.ttf'
        font = ImageFont.load_default()
        
        description = ''# textWrap(description_0, font, 800)
        #line_height = font.getsize('hg')[1]
        
        #plt.annotate(description, xy=(0.0, 0.0), xycoords='axes fraction',  backgroundcolor='w', fontsize=8)
        
        txt = plt.text(0, 1, description, va='top', fontsize=8)
        #txt._get_wrap_line_width = lambda : 600
        
        ax1 = fig.add_axes([0.1, 0.65, 0.3, 0.1])
        self.plotStockPrice(axis=ax1, adjust_by_cpi=True, plot_type='line', color='k')
        self.plotStockPrice(axis=ax1, plot_type='line')
        # plt.legend()
        
        ax2 = fig.add_axes([0.1, 0.45, 0.3, 0.1])
        self.plotRevenue(axis=ax2, adjust_by_cpi=True, plot_type='bar', color='k')
        self.plotRevenue(axis=ax2, plot_type='bar')
        self.plotRevenueForcast(axis=ax2, plot_type='bar', color='C3')
        # plt.legend()
    
        ax3 = fig.add_axes([0.1, 0.25, 0.3, 0.1])
        self.plotEBIT(axis=ax3, adjust_by_cpi=True, plot_type='bar', color='k')
        self.plotEBIT(axis=ax3, plot_type='bar')
        self.plotEBITForcast(axis=ax3, plot_type='bar', color='C3')
        # plt.legend()
    
    
        ax4 = fig.add_axes([0.1, 0.05, 0.3, 0.1])
        self.plotEPS(axis=ax4, plot_type='bar')
        self.plotEPSForcast(axis=ax4, plot_type='bar', color='C3')
        # plt.legend()
        
        
        ax5 = fig.add_axes([0.6, 0.65, 0.3, 0.1])
        self.plotPriceEarningsRatio(axis=ax5, plot_type='bar', color='C0')
        
        
        ax6 = fig.add_axes([0.6, 0.45, 0.3, 0.1])
        self.plotPriceSalesRatio(axis=ax6, plot_type='bar', color='C0')
        
        
        ax7 = fig.add_axes([0.6, 0.25, 0.3, 0.1])
        self.plotPriceBookRatio(axis=ax7, plot_type='bar', color='C0')
        
        
        ax8 = fig.add_axes([0.6, 0.05, 0.3, 0.1])
        self.plotPriceFreeCashFlowRatio(axis=ax8, plot_type='bar', color='C0')
        
    
        fig.savefig('stock_sheets/{}.pdf'.format(self.ticker))


if __name__ == "__main__":

    ticker_list = ['QCOM',
                   'PYPL',
                   'VLPNY',
                   'NEM',
                   'FNV',
                   'GOLD',
                   'EADSY',
                   'AIR',
                   'JPM',
                   'AEM',
                   'BAC',
                   'GLNCY',
                   'RIO',
                   ]
    
    ticker = 'BAC'
    
    stock_data = StockData(ticker, 'a')
    
    stock_data.getStockSheet()
    
    #for ticker in ticker_list:
    #    getStockSheet(ticker)






























