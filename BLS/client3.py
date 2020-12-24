#!/usr/bin/env python3
import os
import numpy as np
import time, _thread
import math,random
from numpy.random import randn
from math import log10
import socket
import socketserver  # 由于TCP起的server同一时间不能跟多个客户端通信，socketserver模块刚好解决了该问题
import hmac   # server端和合法的客户端有相同的密钥，server端给客户端发送要加密的bytes类型
              # 客户端也使用hmac进行加密，将结果返回给server端，server端比较两个加密的结果
secret_key=b"OREO"   # server和合法的客户端之间的密钥（约定好的，双方都已知的）
HOST = '127.0.0.1'
PORT = 8001
ID = '3'
MAC1 = '04:4A:6C:D4:6D:79'
MAC2 = '9c:4A:6C:D4:6D:79'
MAC3 = '36:4A:6C:D4:6D:79'
X1 = 0.8
Y1 = 1.8
X2 = 0.4
Y2 = 7.8
X3 = 8
Y3 = 5.3
X0 = 0
Y0 = 0
Xn = 8
Yn = 11
A1 = 59
n1 = 2
A2 = 59
n2 = 2
A3 = 59
n3 = 2
Tarx = [0.8,8,6]
Tary = [2.8,5,3]
Spex = [0,0,0]
Spey = [0,0,0]

class Mybluetooth:
    'Simulate the raspberry bluetooth'
    def __init__(self,dt,co_x,co_y,A,n):
        self.dt = dt
        self.co_x = co_x
        self.co_y = co_y
        self.A = A
        self.n = n
    
    def get_RSSI(self,tar_x,tar_y,speed_x,speed_y):
        distance = math.sqrt((tar_x+self.dt*speed_x-self.co_x)**2+(tar_y+self.dt*speed_y-self.co_y)**2)
        RSSI = -(self.A+10*self.n*log10(distance))
        RSSI = int(RSSI)
        err = int(1.5*randn())
        final_RSSI = RSSI+err
        if RSSI+err>=100:
            final_RSSI = -99
        return final_RSSI

Raspi3 = Mybluetooth(0.1,X3,Y3,A3,n3)

sk=socket.socket()
sk.connect((HOST,PORT))
msg=sk.recv(1024)  # hmac.new()操作的对象都是bytes类型的，所以先不解码啦
h=hmac.new(secret_key,msg)
client_digest=h.digest()
sk.send(client_digest)  # 把加密后的给server端发过去
while True:
    target1_rssi = Raspi3.get_RSSI(Tarx[0],Tary[0],Spex[0],Spey[0])
    target1_rssistr = str(target1_rssi)
    message1 = ID+',[\''+MAC1+'\', '+ target1_rssistr+']'
    print(message1)
    target２_rssi = Raspi3.get_RSSI(Tarx[1],Tary[1],Spex[1],Spey[1])
    target２_rssistr = str(target2_rssi)
    message2 = ID+',[\''+MAC2+'\', '+ target2_rssistr+']'
    print(message2)
    target3_rssi = Raspi3.get_RSSI(Tarx[2],Tary[2],Spex[2],Spey[2])
    target3_rssistr = str(target3_rssi)
    message3 = ID+',[\''+MAC3+'\', '+ target3_rssistr+']'
    print(message3)
    sk.send(message1.encode("utf-8"))
    time.sleep(0.1)
    sk.send(message2.encode("utf-8"))
    time.sleep(0.1)
    sk.send(message3.encode("utf-8"))
    time.sleep(0.1)
sk.close()

