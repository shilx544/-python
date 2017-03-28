import mysql.connector
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import numpy as np
import datetime

SQL_config = {'host': '192.168.1.128',
                'user': 'root',
                'password': '123456',
                'database': 'bigdata',
                'charset': 'utf8',}

#获取商品销量变化情况
def get_sale_change(productid):
    sale_info = []
    cnn = mysql.connector.connect(**SQL_config)
    cursor = cnn.cursor()
    sql = 'select productid, tmprice, shopid, tmsaleNum, adddate from tmall'
    cursor.execute(sql)
    values = cursor.fetchall()
    # 提交事务
    cnn.commit()
    cursor.close()
    min_date = values[0][4]
    max_date = values[0][4]
    for line in values:
        if line[0] == productid:
            if line[4] < min_date:
                min_date = line[4]
            elif line[4] > max_date:
                max_date = line[4]
            sale_info.append([line[3], line[4]])
    sale_mat = np.mat(sale_info)
    # m = np.shape(sale_mat)[0]
    # for i in range(m):
    #     y.append(str(sale_mat[i, 1]))
    x = dts.date2num(sale_mat[:, 1])
    datefmt = dts.DateFormatter('%Y-%m-%d')

    #用pyplot画图
    y = (sale_mat[:, 0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(datefmt)

    ax.plot(x,y)
    plt.show()
    return sale_info

def get_comment_change(Porduct_id, start_date, end_date, freq=1):
    pass

if __name__ == '__main__':
    get_sale_change('9043679495')