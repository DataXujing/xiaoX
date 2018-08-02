from flask import (Flask,render_template,url_for)
import flask
import datetime
import redis

from lx_bot_3 import *


app=Flask(__name__)

app.secret_key='xujing in inter-credit'

#redis数据库
r=redis.StrictRedis(host='172.16.100.147',port=6379,db=1,decode_responses=True)


def event_stream():
    pubsub=r.pubsub()  #用于查看订阅与发布系统状态，返回由活跃频道组成的列表
    pubsub.subscribe('lx_chat')  #用于订阅给定的一个或多个频道的信息，返回接收到的信息
    for message in pubsub.listen():  #监听活跃频道组成的列表
        print(message)
        yield 'data:{}\n\n'.format(message['data'])


#首页
@app.route('/')
def index():
    if 'user' not in flask.session:
        return flask.redirect('/login')  #重定向到login
    user=flask.session['user']
    return render_template('index.html',user=user)  #index中的user变量



#登录页面
@app.route('/login',methods=['GET','POST'])
def login():
    if 'user' in flask.session:
        return flask.redirect('/')
    if flask.request.method=='POST':   #如果请求方式是POST
        flask.session['user']=flask.request.form['user']   #获取login表单中的数据
        r.publish('lx_chat', '用户{}加入了房间!'.format(flask.session['user']))  #用于将信息发送到指定的频道。
        return flask.redirect('/')
    return render_template('login.html')



#注销
@app.route('/logout')
def logout():
    user=flask.session.pop('user')   #删除用户绘画中的user
    print(user)
    r.publish('lx_chat', '用户{}退出了房间'.format(user))
    return flask.redirect('/login')

#发送消息
@app.route('/send',methods=['POST'])
def post():
    message=flask.request.form['message']
    user=flask.session.get('user','anonymous')
    now=datetime.datetime.now().replace(microsecond=0).time()  #日期去掉毫秒，只取时间
    little_x = predict(message)


    r.publish('lx_chat','[{}] {} : {}xujing_replace[{}] 小X : {}'.format(now.isoformat(),user,message, now.isoformat(),little_x ))  #时间设置成ISO标准格式

   
    return flask.Response(status=204)

#SSE事件流
@app.route('/stream')
def stream():
    return flask.Response(event_stream(),mimetype='text/event-stream')



if __name__ == '__main__':
    app.run(host='172.16.100.147',port=5052,debug=True, threaded=True)