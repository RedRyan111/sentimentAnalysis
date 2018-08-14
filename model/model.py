from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import re

def small_lstm(num_inp,x_old,h_old,c_old):
    W_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    W_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    W_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    W_c = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    W_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

    U_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    U_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    U_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    U_c = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    U_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

    B_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    B_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    B_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    B_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

    F_t = tf.sigmoid( tf.multiply(x_old,W_f) + tf.multiply(h_old,U_f) + B_f)
    I_t = tf.sigmoid( tf.multiply(x_old,W_i) + tf.multiply(h_old,U_i) + B_i)
    O_t = tf.sigmoid( tf.multiply(x_old,W_o) + tf.multiply(h_old,U_o) + B_o)
    A_t = tf.tanh(    tf.multiply(x_old,W_a) + tf.multiply(h_old,U_a) + B_a)

    c_t = tf.multiply(I_t,c_old) + tf.multiply(F_t,A_t)
    h_t = tf.multiply(O_t,tf.tanh(c_t))

    return c_t,h_t

def lstm_layer(X_txt,num_inp,max_len):
    curC = 0.0
    curH = 0.0
        
    for i in range(max_len):
        curC, curH = small_lstm(num_inp,X_txt[i],curH,curC)
    return curH

file_path = "~/Downloads/sentiment140/training.1600000.processed.noemoticon.csv"


col_names = ['target','id','date','flag','user','text']

df = pd.read_csv(file_path,encoding='ISO-8859-1',names=col_names)

txt = df['text']

target = df['target']

glove_dir = "../../../../Downloads/GloveWordEmb/glove.840B.300d.txt"


Glove_file = open(glove_dir)

w2v_dict = {}

count = 0
for line in Glove_file:
    line = line.strip('\n').split(" ")
    word = line.pop(0)
    line = np.array(line)
    line = line.astype(np.float)
    
    w2v_dict[word] = line

def sample2Vec(w2v_dict,sample):
    sample_emb = []
    hold = []
    for i in sample:
        if i in w2v_dict:
            hold = w2v_dict[i]
        else:
            hold = np.zeros(300)
        sample_emb.append(hold)
    return sample_emb

print("before dataframes")
vec_df = pd.DataFrame(columns=list(range(300)))
tar_df = pd.DataFrame(columns=['target'])
print("done with dataframes")

def pre_proc_sen(sentence):
    test = sentence.lower()
    test = test.replace("'","").replace('"',"")
    test = re.sub(r'http\S+', '', test)
    test = re.sub(r';\S+','',test)
    test = re.sub(r':\S+','',test)
    test = re.sub('[^a-zA-Z]',' ',test)
    test = test.split()
    return test

max_words = 30
vec_np = 0
txt_fin = []

X_train, X_test, Y_train, Y_test = train_test_split(vec_np,tar_np,test_size=0.30,random_state=42)
x_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(Y_train)


#create batches mini batches
small_len = 100000
num_batches = int(len(X_train)/(small_len))
x_data_arr = []
y_data_arr = []
for i in range(1):#num_batches):
    small_arr = x_train.iloc([i*small_len,i*small_len+small_len])
    x_data_arr = x_data_arr.append(small_arr)

    small_arr = y_train.iloc([i*small_len,i*small_len+small_len])
    y_data_arr = y_data_arr.append(small_arr)


#Create batches of word to vector
x_vect_arr = []
for i in range(1):#num_batches):
    for j in range(small_len): 
        sample_txt = x_data_arr[i][j]
    
        sample_txt = pre_proc_sen(sample_txt)
        
        if(len(sample_txt) < max_words):
            txt_fin.append(sample_txt)
            sample_txt = sample2Vec(w2v_dict,sample_txt)
        
            for j in range(max_words):
                if(j >= len(sample_txt)):
                    sample_txt.append(np.zeros(300))
                
            sample_txt = np.array(sample_txt)
            sample_txt = np.reshape(sample_txt,(1,30,300))        

        x_mini_arr.append(sample_txt)

    x_vect_arr[i].append(x_mini_arr)

#sample_tar = np.reshape(sample_tar,(1,1))

num_inp = 300
num_classes = 3
epochs = 100

inp = tf.placeholder(tf.float32,shape=[max_words,300])

y_true = tf.placeholder(tf.float32,[1,num_classes])

W_out = tf.Variable(tf.truncated_normal(shape=[300,num_classes],stddev=.1))
b_out = tf.Variable(tf.truncated_normal(shape=[num_classes],stddev=.1))

y = lstm_layer(inp,num_inp,max_words)

y = tf.reshape(y,(1,300))

y_out = tf.matmul(y,W_out)+b_out

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_out))

optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y2,1),tf.argmax(y_true,1))

acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

loss_arr = []
tot_loss = 0



init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for ep in range(epochs):
        print("epoch: "+str(ep))
        for i in range(len(X_train)):
            batch_x = X_train[i]

            ind = np.divide(Y_train[i],2)

            batch_y = tf.one_hot(indices = ind, depth = 3)
            batch_y = sess.run(batch_y)
            
            sess.run(optimizer, feed_dict={inp:batch_x,y_true:batch_y})

            hold = sess.run(cross_entropy, feed_dict={inp:batch_x,y_true:batch_y})
            tot_loss += hold
        loss_arr.append(tot_loss)

plt.plot([i for i in range(len(loss_arr))],loss_arr)

