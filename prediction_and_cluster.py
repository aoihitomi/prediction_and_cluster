import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(".\\vgsales.csv")

# 输出数据集的一些信息 
def summary(data):
    print(data['Platform'].value_counts())
    print(data['Genre'].value_counts())
    print(data['Publisher'].value_counts())
    print(data.iloc[0:10]['Name'])

# 对全球销售额数据进行多项式拟合
def fit(data):
    year_data = data.groupby('Year').sum()
    year_data = year_data[['Global_Sales']]
    year_data = year_data.loc[1980:2016]
    
    years = []
    sales = []

    for index, row in year_data.iterrows():
        years.append(int(index))
        sales.append(row['Global_Sales'])
    
    years = np.array(years)
    sales = np.array(sales)
    
    f = np.polyfit(years, sales, 10)
    p = np.poly1d(f)
    sales_fit = p(years)
    
    plot1 = plt.plot(years, sales, 's',label='original values')
    plot2 = plt.plot(years, sales_fit, 'r',label='polyfit values')
    plt.xlabel('years')
    plt.ylabel('sales')
    plt.legend(loc=4)
    plt.title('polyfitting')
    plt.show()
    
    for year in [2017, 2018, 2019]:
        print("Predict: Global_Sales of year {} is {}.".format(year, p(year)))

# 根据年份对游戏数据信息进行可视化
# Params:
#   data: 数据集
#   col : 数据列名称，标称型数据
def show_genre_and_publisher(data, col):
    year_data = data.groupby('Year').agg(lambda x: (x.value_counts().index[0], x.value_counts().values[0])).reset_index()
    year_data = year_data[['Year', col]].loc[0:36]

    years = []
    label = []
    value = []
    
    for index, row in year_data.iterrows():
        years.append(int(row['Year']))
        label.append(row[col][0].split(' ')[0])
        value.append(row[col][1])
        
    plt.bar(years, value)
    for a, b, c in zip(years, value, label):
        plt.text(a, b+0.05, c, ha='center', va= 'bottom',fontsize=10)
    plt.show()
        
    
if __name__ == '__main__':
    summary(data)
    fit(data)
    show_genre_and_publisher(data, 'Genre')
    show_genre_and_publisher(data, 'Publisher')
    show_genre_and_publisher(data, 'Platform')
    


    
