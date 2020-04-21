import bs4 as bs
import pickle
import requests
import os
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import numpy as np
import csv
import urllib.request, json
from collections import Counter
import pandas as pd
from sklearn import model_selection as cross_validation
from sklearn import svm, neighbors
from sklearn.model_selection import cross_validate
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from statistics import mean
from sklearn.model_selection import train_test_split

style.use('ggplot')

'''def save_sensex_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/BSE_SENSEX')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text.replace('.','-')
        tickers.append(ticker)

    with open("sensextickers.pickle","wb") as f:
        pickle.dump(tickers,f)
    print(tickers)

    return tickers

save_sensex_tickers()   
'''
# save sensex_tickers()
def convert_to_list():
    with open('ind_nifty500list.csv', 'r') as f:
        reader = csv.reader(f)
        scrips = list(reader)
    tickers = []
    for i in range(len(scrips)):

        tickers.append(scrips[i][1])

    del tickers[0]
    for ticker in tickers:
        ticker = ticker.replace('.','-').strip()
    tickers = ['NSE:'+ x for x in tickers]

    outfile = open("nse_500_tickers.pickle","wb")
    pickle.dump(tickers,outfile)
    outfile.close()
    return tickers

#convert_to_list()
def get_data_from_av(reload_sensex=False):
    if reload_sensex:
        tickers = convert_to_list()
    else:
        with open("nse_500_tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stocks_dfs'):
        os.makedirs('stocks_dfs')

    start = dt.datetime(2019, 6, 12)
    end = dt.datetime.now()
    api_key = 'xxx'
    tick_names = [s.replace(':', '-') for s in tickers]
    
    for ticker in tickers:
        try:
            url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
            file_to_save = 'stock_dfs/%s.csv'%ticker.replace(':','-')
            print(ticker)

            if not os.path.exists(file_to_save):
                with urllib.request.urlopen(url_string) as url:
                    data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
                for k,v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1
                print('Data saved to : %s'%file_to_save)        
                df.to_csv(file_to_save)

        except:
            pass    

#get_data_from_av(True)


def compile_data():
    with open('nse_500_tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    tickers = [s.replace(':','-') for s in tickers]
   # print(tickers[:10])

    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers):

        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)

        #df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        #df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']
        
        df.rename(columns={'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Volume', 'Close','Unnamed: 0'],1,inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('nse_500_tickers_joined_closes.csv')

#compile_data()


def visualize_data():
    df = pd.read_csv('nse_500_tickers_joined_closes.csv')

    df['NSE-TCS'].plot()
    plt.show

    df_corr = df.corr()
    print(df_corr.head())

    df_corr.to_csv('nsecorr.csv')


    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)

    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
        
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig('correlantions.png', dpi=(300))
    plt.show()

#visualize_data()

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('nse_500_tickers_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
       
    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)] ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)


    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    
    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
  
  #K Nearest Neighbors

    #clf = neighbors.KNeighborsClassifier()


    # VotingClassifier
    clf = VotingClassifier([('lsvc', svm.LinearSVC(max_iter=2000)),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])


    clf.fit(X_train, y_train)

    confidence = clf.score(X_test, y_test)

    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts: ', Counter(predictions))
    print()
    print()
   # return confidence



# testcases running:
do_ml('NSE-TCS')
do_ml('NSE-M&M')
do_ml('NSE-ABB')

'''
with open("nse_500_tickers.pickle", "rb") as f:
    tickers = pickle.load(f)

accuracies =[]
for count,ticker in enumerate(tickers):

    if count%10==0:
        print(count)


    accuracy = do_ml(ticker)
    accuracies.append(accuracy)
    print('{} accuracy: {}. Average accuracy:{}'.format(ticker,accuracy,mean(accuracies)))
'''
