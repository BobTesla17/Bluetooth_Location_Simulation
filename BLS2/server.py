#!/usr/bin/env python3
import os
import numpy as np
import time, _thread
import math,random
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import plotly
from plotly.offline import init_notebook_mode,iplot
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from numpy import eye,array,asarray,matrix
from math import log,log10
import socketserver  # 由于TCP起的server同一时间不能跟多个客户端通信，socketserver模块刚好解决了该问题
import hmac   # server端和合法的客户端有相同的密钥，server端给客户端发送要加密的bytes类型
              # 客户端也使用hmac进行加密，将结果返回给server端，server端比较两个加密的结果
secret_key=b"OREO"   # server和合法的客户端之间的密钥（约定好的，双方都已知的）

HOST = '127.0.0.1'
#HOST = '192.168.8.18'
PORT = 8001
Data = np.array([np.zeros(6),np.zeros(6),np.zeros(6)],dtype=object)
X1 = 0.8
Y1 = 1.8
X2 = 0.4
Y2 = 7.8
X3 = 8
Y3 = 5.3
X4 = 7
Y4 = 0.5
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
fig1 = go.Figure(data=[go.Scatter(
            x=[X1,X2,X3],
            y=[Y1,Y2,Y3],
            mode='markers',
            marker=dict(
                color=['rgb(255,0,255)','rgb(255,0,255)','rgb(255,0,255)'],
                size=[35,35,35]))])
fig2 = go.Figure(data=[go.Scatter(
            x=[X1,X2,X3],
            y=[Y1,Y2,Y3],
            mode='markers',
            marker=dict(
                color=['rgb(255,0,255)','rgb(255,0,255)','rgb(255,0,255)'],
                size=[35,35,35]))])
fig3 = go.Figure(data=[go.Scatter(
            x=[X1,X2,X3],
            y=[Y1,Y2,Y3],
            mode='markers',
            marker=dict(
                color=['rgb(255,0,255)','rgb(255,0,255)','rgb(255,0,255)'],
                size=[35,35,35]))])
fig4 = go.Figure(data=[go.Scatter(
            x=[X1,X2,X3],
            y=[Y1,Y2,Y3],
            mode='markers',
            marker=dict(
                color=['rgb(255,0,255)','rgb(255,0,255)','rgb(255,0,255)'],
                size=[35,35,35]))])
fig = [fig1,fig2,fig3,fig4]

def Hjacobian_at(x):
    global X1,Y1,X2,Y2,X3,Y3,n1,n2,n3
    """ compute Jacobian of H matrix at x """
    H11 = (-10)*n1*(x[0]-X1)/(log(10)*(pow((x[0]-X1),2)+pow((x[1]-Y1),2)))
    H12 = (-10)*n1*(x[1]-Y1)/(log(10)*(pow((x[0]-X1),2)+pow((x[1]-Y1),2)))
    H21 = (-10)*n2*(x[0]-X2)/(log(10)*(pow((x[0]-X2),2)+pow((x[1]-Y2),2)))
    H22 = (-10)*n2*(x[1]-Y2)/(log(10)*(pow((x[0]-X2),2)+pow((x[1]-Y2),2)))
    H31 = (-10)*n3*(x[0]-X3)/(log(10)*(pow((x[0]-X3),2)+pow((x[1]-Y3),2)))
    H32 = (-10)*n3*(x[1]-Y3)/(log(10)*(pow((x[0]-X3),2)+pow((x[1]-Y3),2)))
    H_obj = array([[H11,H12,0,0],[H21,H22,0,0],[H31,H32,0,0]])
    #H_float = np.frompyfunc(lambda x: x.replace(',',''),1,1)(H_obj).astype(float)
    H_float = H_obj.astype(float)
    return H_float

def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """
    global X1,Y1,X2,Y2,X3,Y3,A1,A2,A3,n1,n2,n3
    h1 = -(A1+5*n1*log10(pow((x[0]-X1),2)+pow((x[1]-Y1),2)))
    h2 = -(A2+5*n2*log10(pow((x[0]-X2),2)+pow((x[1]-Y2),2)))
    h3 = -(A3+5*n3*log10(pow((x[0]-X3),2)+pow((x[1]-Y3),2)))
    h_obj = array([[h1],[h2],[h3]])
    #H_float = np.frompyfunc(lambda x: x.replace(',',''),1,1)(h_obj).astype(float)
    h_float = h_obj.astype(float)
    return h_float

dt = 0.3
range_std = 30
# define Kalman filter 1
rk1 = ExtendedKalmanFilter(dim_x=4,dim_z=3)
rk1.x = array([[5],[5],[0],[0]])
rk1.F = eye(4) + array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])*dt
rk1.R = eye(3)*range_std
rk1.Q = Q_discrete_white_noise(4,dt=dt,var=0.1)
rk1.P *= 50
xs1, track1 = [],[]
# define Kalman filter 2
rk2 = ExtendedKalmanFilter(dim_x=4,dim_z=3)
rk2.x = array([[5],[5],[0],[0]])
rk2.F = eye(4) + array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])*dt
rk2.R = eye(3)*range_std
rk2.Q = Q_discrete_white_noise(4,dt=dt,var=0.1)
rk2.P *= 50
xs2, track2 = [],[]
# define Kalman filter 3
rk3 = ExtendedKalmanFilter(dim_x=4,dim_z=3)
rk3.x = array([[5],[5],[0],[0]])
rk3.F = eye(4) + array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])*dt
rk3.R = eye(3)*range_std
rk3.Q = Q_discrete_white_noise(4,dt=dt,var=0.1)
rk3.P *= 50
xs3, track3 = [],[]
# define Kalman filter list
rk = [rk1,rk2,rk3]
#xs = [xs1,xs2,xs3]
#track = [track1,track2,track3]

def squ_error(i,j,RSSI1,RSSI2,RSSI3):
    distance_cur1 = ((i - X1) ** 2 + (j - Y1) ** 2) ** 0.5
    distance_cur2 = ((i - X2) ** 2 + (j - Y2) ** 2) ** 0.5
    distance_cur3 = ((i - X3) ** 2 + (j - Y3) ** 2) ** 0.5
    rssi1 = -(A1+10*n1*np.log10(distance_cur1))
    rssi2 = -(A2+10*n2*np.log10(distance_cur2))
    rssi3 = -(A3+10*n3*np.log10(distance_cur3))
    error = (rssi1-RSSI1)**2+(rssi2-RSSI2)**2+(rssi3-RSSI3)**2
    alt = 1/error
    return alt


class MyServer(socketserver.BaseRequestHandler):   # 自己定义的类Myserver必须要继承BaseRequestHandler
    def process(self,DATA,mac,rasp,rssi,timetick):
        global A1,n1,A2,n2,A3,n3
        find = False
        for i in range(3):
            if mac == DATA[i][0]:
                find = True
                if rasp == 1:
                    DATA[i][1] = rssi
                    DATA[i][4] = timetick
                elif rasp == 2:
                    DATA[i][2] = rssi
                    DATA[i][4] = timetick
                else:
                    DATA[i][3] = rssi
                    DATA[i][4] = timetick
                if int(DATA[i][1])!=0 and int(DATA[i][2])!=0 and int(DATA[i][3])!=0:
                    DATA[i][5] = 1
            if str(DATA[i][0])=='0':
                find = True
                DATA[i][0] = mac
                if rasp == 1:
                    DATA[i][1] = rssi
                    DATA[i][4] = timetick
                elif rasp == 2:
                    DATA[i][2] = rssi
                    DATA[i][4] = timetick
                else:
                    DATA[i][3] = rssi
                    DATA[i][4] = timetick
        if find == False :
            ticks = [DATA[0][4],DATA[1][4],DATA[2][4]]
            pos = ticks.index(min(ticks))
            DATA[pos][0] = mac
            DATA[pos][5] = 0
            if rasp == 1:
                DATA[pos][1] = rssi
                DATA[pos][4] = timetick
            elif rasp == 2:
                DATA[pos][2] = rssi
                DATA[pos][4] = timetick
            else:
                DATA[pos][3] = rssi
                DATA[pos][4] = timetick
        return DATA
    
    def KALMAN(self,DATA):
        global rk,rk1,rk2,rk3,Loca
        Loca = np.array([np.zeros(4),np.zeros(4),np.zeros(4)],dtype=object)
        Loca[0][3] == 0
        Loca[1][3] == 0
        Loca[2][3] == 0
        for i in range(3):
            if DATA[i][5] == 1:
                RSSI1=float(DATA[i][1])
                RSSI2=float(DATA[i][2])
                RSSI3=float(DATA[i][3])
                z = array([[RSSI1],[RSSI2],[RSSI3]])
                rk[i].update(z,Hjacobian_at,hx)
                rk[i].predict()
                STATE = rk[i].x
                Loca[i][0] = DATA[i][0]
                Loca[i][1] = float(STATE[0])
                Loca[i][2] = float(STATE[1])
                Loca[i][3] = 1
        return Loca
    
    def bubble(self,LOCA,DATA):
        global X1,Y1,X2,Y2,X3,Y3
        global fig1,fig2,fig3,fig4
        rgb1 = 'rgb(0,255,255)'
        rgb2 = 'rgb(0,0,255)'
        rgb3 = 'rgb(0,255,0)'
        rgb = [rgb1,rgb2,rgb3]
        target_name = ['point','point','point','point','raspberry_1','raspberry_2','raspberry_3','dest1','dest2','dest3']
        target_x = [0,9,9,0,X1,X2,X3,Tarx[0],Tarx[1],Tarx[2]]
        target_y = [0,0,9,9,Y1,Y2,Y3,Tary[0],Tary[1],Tary[2]]
        color_list = ['rgb(255,255,255)','rgb(255,255,255)','rgb(255,255,255)','rgb(255,255,255)',
                'rgb(255,0,255)','rgb(255,0,255)','rgb(255,0,255)',rgb1,rgb2,rgb3]
        size_list = [5,5,5,5,25,25,25,15,15,15]
        x,y = np.mgrid[1:8:80j,1:11:110j]
        z = []
        for i in range(3):
            if LOCA[i][3] == 1:
                target_name.append(LOCA[i][0])
                target_x.append(LOCA[i][1])
                target_y.append(LOCA[i][2])
                color_list.append(rgb[i])
                size_list.append(30)
                rssi1 = float(DATA[i][1])
                rssi2 = float(DATA[i][2])
                rssi3 = float(DATA[i][3])
                z.append(squ_error(x,y,rssi1,rssi2,rssi3))
        fig1 = go.Figure(data=[go.Scatter(
            x=target_x,
            y=target_y,
            text=target_name,      
            mode='markers',
            marker=dict(
                color=color_list,
                size=size_list))])
        fig1.update_layout(
                autosize = False,
                width = 1200,
                height = 1200,
                title = 'Bluetooth Location',
                xaxis = dict(
                            title=dict(text='Length'),
                            tickmode='linear',
                            tick0=0,
                            dtick=0.5,
                            ticklen=0.5),
                yaxis = dict(
                            title=dict(text='Width'),
                            tickmode='linear',
                            tick0=0,
                            dtick=0.5,
                            ticklen=0.5),
                            )
        for i in range(3):
            if LOCA[i][3] == 1:
                if i == 0:
                    fig2 = go.Figure(data=[go.Surface(x=x,y=y,z=z[i],colorbar_y=0.81,colorbar=dict(lenmode='fraction',len=4))])
                    fig2.update_layout(
                            autosize = False,
                            width = 2400,
                            height = 1200,
                            title = 'Bluetooth Location',
                            xaxis = dict(
                                        title=dict(text='Length'),
                                        tickmode='linear',
                                        tick0=0,
                                        dtick=0.5,
                                        ticklen=0.5),
                            yaxis = dict(
                                        title=dict(text='Width'),
                                        tickmode='linear',
                                        tick0=0,
                                        dtick=0.5,
                                        ticklen=0.5),
                                        )
                elif i == 1:
                    fig3 = go.Figure(data=[go.Surface(x=x,y=y,z=z[i],colorbar_y=0.81,colorbar=dict(lenmode='fraction',len=4))])
                    fig2.update_layout(
                            autosize = False,
                            width = 2400,
                            height = 1200,
                            title = 'Bluetooth Location',
                            xaxis = dict(
                                        title=dict(text='Length'),
                                        tickmode='linear',
                                        tick0=0,
                                        dtick=0.5,
                                        ticklen=0.5),
                            yaxis = dict(
                                        title=dict(text='Width'),
                                        tickmode='linear',
                                        tick0=0,
                                        dtick=0.5,
                                        ticklen=0.5),
                                        )
                elif i == 2:
                    fig4 = go.Figure(data=[go.Surface(x=x,y=y,z=z[i],colorbar_y=0.81,colorbar=dict(lenmode='fraction',len=4))])
                    fig2.update_layout(
                            autosize = False,
                            width = 2400,
                            height = 1200,
                            title = 'Bluetooth Location',
                            xaxis = dict(
                                        title=dict(text='Length'),
                                        tickmode='linear',
                                        tick0=0,
                                        dtick=0.5,
                                        ticklen=0.5),
                            yaxis = dict(
                                        title=dict(text='Width'),
                                        tickmode='linear',
                                        tick0=0,
                                        dtick=0.5,
                                        ticklen=0.5),
                                        )

    def handle(self):   # 该类中必须要定义的方法
        msg=os.urandom(32)  # 随机生成一个32位的字节，用于发送给client端，进行验证（配合密钥）
        self.request.send(msg)  # self.request相当于socket中的conn表示一个连接
        h=hmac.new(secret_key,msg)  # 获得需要加密的对象 一个是双方约定好的密钥 ，另一个是随机生成的bytes类型的需要加密的信息
        digest=h.digest()  # 加密
        client_digest=self.request.recv(1024)
        ret=hmac.compare_digest(digest,client_digest)
        global Data
        if ret:
            while True:
                ret=self.request.recv(1024).decode("utf-8")
                rasp_ID = int(ret[0])
                mac_ID = ret[4:21]
                rssi = int(ret[24:27])
                ticks = time.time()
                Data = self.process(Data,mac_ID,rasp_ID,rssi,ticks)
                if len(ret) > 28:
                    rasp_ID = int(ret[28])
                    mac_ID = ret[32:49]
                    rssi = int(ret[52:55])
                    ticks = time.time()
                    Data = self.process(Data,mac_ID,rasp_ID,rssi,ticks)
                LOCATION = self.KALMAN(Data)
                self.bubble(LOCATION,Data)
                print(Data)
        else:
            print("不合法的客户端，我要断开连接")
    

#server = socketserver.ThreadingTCPServer((HOST,PORT),MyServer)
#server.allow_reuse_address = True

#server.server_bind()
#server.server_activate()
app = dash.Dash(__name__)
app.layout = html.Div([
    html.P("Bluetooth Location:"),
    dcc.Dropdown(
        id="dropdown",
        options=[
            {'label': x, 'value': x}
            for x in ['Bubble', 'Bluetooth1', 'Bluetooth2','Bluetooth3']
        ],
        value='Bubble',
        clearable=False,
    ),
    dcc.Interval(
            id='interval-component',
            interval=0.5*1000, # in milliseconds
            n_intervals=0,
    ),
    dcc.Graph(id="graph")
])

@app.callback(
    Output("graph", "figure"), 
    [Input("dropdown", "value"),Input('interval-component','n_intervals')])

def display_figure(dropdown_value,n):
    global fig1,fig2,fig2,fig3
    print(dropdown_value)
    if dropdown_value == 'Bluetooth1':
        return fig2
    elif dropdown_value == 'Bluetooth2':
        return fig3
    elif dropdown_value == 'Bluetooth3':
        return fig4
    else:
        return fig1
    
_thread.start_new_thread(app.run_server,()) 

server=socketserver.ThreadingTCPServer((HOST,PORT),MyServer)   # 必须要传两个参数 IP地址 端口号，还有上面定义的类名
print("Hello World!")
server.serve_forever()
#_thread.start_new_thread(server.serve_forever,())   # 永久起一个服务

#app.run_server(debug=True)
