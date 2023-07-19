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
import threading
import time

# 定义一个函数作为线程的执行体
def thread_function1():
    while(1):
        print("Thread 1 started")
        time.sleep(2)  # 模拟线程1的一些操作
        print("Thread 1 finished")

def thread_function2():
    while(1):
        print("Thread 2 started")
        time.sleep(3)  # 模拟线程2的一些操作
        print("Thread 2 finished")

# 创建两个线程
thread1 = threading.Thread(target=thread_function1)
thread2 = threading.Thread(target=thread_function2)

# 启动线程
thread1.start()
thread2.start()

# 主线程可以继续执行其他操作
print("Main thread")

# 等待两个线程执行完成
thread1.join()
thread2.join()

# 所有线程执行完毕
print("All threads finished")
