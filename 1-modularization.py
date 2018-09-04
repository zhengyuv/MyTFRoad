# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 20:16:38 2018

@author: dell
"""

import tensorflow as tf

#初始化变量和模型参数，定义训练闭环中的运算
W = tf.Variable(initial_value=tf.zeros(shape=[2,1]),name="weights")
b = tf.Variable(initial_value=0.0,name="bias")

    
def inference(X):
    #计算推断模型在数据X上的输出，并将结果返回
    return tf.matmul(X,W) + b
    
def loss(X,Y):
    #依据训练数据X及其期望输出Y计算损失
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y,Y_predicted))

def inputs():
    #读取或生成训练数据X及其期望输出Y
    weight_age = [[84,46],[73,20],[65,52],[70,30],
                  [76,57],[69,25],[63,28],[72,36],
                  [79,57],[75,44],[27,24],[89,31],
                  [65,52],[57,23],[59,60],[69,48],
                  [60,34],[79,51],[75,50],[82,34],
                  [59,46],[67,23],[85,37],[55,40],
                  [63,30]]
    blood_fat_content=[354,190,405,263,451,302,288,385,
                       402,365,209,290,346,254,395,434,
                       220,374,308,220,311,181,274,303,244]
    return tf.to_float(weight_age),tf.to_float(blood_fat_content)
    
def train(total_loss):
    #依据计算的总损失训练或调整模型参数
    learning_rate=0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    
def evaluate(sess,X,Y):
    #对训练得到的模型进行评估
    print(sess.run(inference([[80.0,25.0]])))
    print(sess.run(inference([[65.0,25.0]])))
    
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    X,Y = inputs()
    
    total_loss = loss(X,Y)
    train_op = train(total_loss)
    
    #实际的训练迭代次数
    training_steps = 1000
    for step in range(training_steps):
        sess.run(train_op)
        #出于调试和学习的目的，查看损失在训练过程中的递减情况
        if step % 10 == 0:
            print("loss:",sess.run(total_loss))
            
    evaluate(sess,X,Y)
    
    sess.close()
    
    
    
    
    
    
    
    
    
    
    