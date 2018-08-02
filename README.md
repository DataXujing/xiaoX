
<img src="static/favo/xiaoX.ico" align="right" alt="logo" height="180" width="180" />


# Flask+seq2seq+redis实现在线的聊天机器人 


徐静

+ flask + redis 实现flask-SSE
+ tensorflow训练seq2seq实现聊天机器人(语料比较少，CPU 10000词迭代周期大约训练了308分钟）
+ 服务器+nginx+uwsgi部署
+ virtualenv提供虚拟Python环境
+ pip install -r requirements.txt 安装必要的Python模块
+ python lx_bot_3.py train 训练seq2seq模型
+ Python lx_bot_3.py pred 与机器人对话


最终实现了线上实时的聊天机器人对话系统。



