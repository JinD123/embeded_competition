""" ━━━━━━神兽出没━━━━━━ 
 　　　┏┓　　　┏┓ 
 　　┃　　　　　　　┃ 
 　　┃　　　━　　　┃ 
 　　┃　┳┛　┗┳　┃ 
 　　┃　　　　　　　┃ 
 　　┃　　　┻　　　┃ 
 　　┃　　　　　　　┃ 
 　　┗━┓　　　┏━┛Code is far away from bug with the animal rotecting 
 　　　　┃　　　┃ 神兽保佑,代码无bug 
 　　　　┃　　　┃ 
 　　　　┃　　　┗━━━┓ 
 　　　　┃　　　　　　　┣┓ 
 　　　　┃　　　　　　　┏┛ 
 　　  　┗┓┓┏━┳┓┏┛ 
　　　　　┃┫┫　┃┫┫ 
　　　　　┗┻┛　┗┻┛ 
"""
import pymysql

# 连接数据库
db = pymysql.connect(host='localhost',
                     user='root',
                     password='049913jin',
                     database='dnf')
# 创建访问对象
cursor = db.cursor()

sql = "SELECT * FROM `key`"
# 下发命令
cursor.execute(sql)

# 接受结果
results = cursor.fetchall()

print (results)
print (type(results))

cursor.execute(sql)
results = cursor.fetchone()
print(results)
results = cursor.fetchone()
print(results)

# 关闭数据库连接
db.close()

