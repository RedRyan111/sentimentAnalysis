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

print("Word embeddings")

count = 0
#for line in Glove_file:
while count<1000:
    count+=1
    line = Glove_file.readline()
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

X_train, X_test, Y_train, Y_test = train_test_split(txt,target,test_size=0.30,random_state=42)
x_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(Y_train)

print("start data mini batches")

#create batches mini batches
small_len = 100000
num_batches = int(len(X_train)/(small_len))
x_data_arr = [None]*num_batches
y_data_arr = [None]*num_batches
for i in range(num_batches):
    small_arr = x_train.iloc[i*small_len:i*small_len+small_len]
    x_data_arr[i] = small_arr

    small_arr = y_train.iloc[i*small_len:i*small_len+small_len]
    y_data_arr[i] = small_arr


print("start word to vector mini batches")
#Create batches of word to vector
x_vect_arr = [np.zeros((30,300))]*num_batches
for i in range(1):#num_batches):
    hold = []
    for j in range(small_len): 
        sample_txt = x_data_arr[i].iloc[j]['text']
        sample_txt = pre_proc_sen(sample_txt)
        
        if(len(sample_txt) < max_words):
            sample_txt = sample2Vec(w2v_dict,sample_txt)
        
            for j in range(max_words):
                if(j >= len(sample_txt)):
                    sample_txt.append(np.zeros(300))
                
            sample_txt = np.array(sample_txt)     
        #hold.append(sample_txt)
    #x_vect_arr[i] = hold
    x_vect_arr[i] = sample_txt

num_inp = 300
num_classes = 3
epochs = 3

inp = tf.placeholder(tf.float32,shape=[max_words,300])

y_true = tf.placeholder(tf.float32,[1,num_classes])

W_out = tf.Variable(tf.truncated_normal(shape=[300,num_classes],stddev=.1))
b_out = tf.Variable(tf.truncated_normal(shape=[num_classes],stddev=.1))

y = lstm_layer(inp,num_inp,max_words)

y = tf.reshape(y,(1,300))

y_out = tf.matmul(y,W_out)+b_out

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_out))

optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_out,1),tf.argmax(y_true,1))

acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

loss_arr = []
tot_loss = 0

print("starting graph")

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for ep in range(epochs):
        print("epoch: "+str(ep))
        tot_loss = 0
        #for k in range(len(x_vect_arr)):
        #for i in range(len(x_vect_arr[k])):
        for i in range(len(x_vect_arr)):

            batch_x = np.array(list(x_vect_arr[i]))
            batch_x = np.reshape(batch_x,(30,300))

            batch_y = [[0,0,0]]
            if(Y_train.iloc[i] == 0):
                batch_y = [[1,0,0]]
            elif(Y_train.iloc[i] == 2):
                batch_y = [[0,1,0]]
            elif(Y_train.iloc[i] == 4):
                batch_y = [[0,0,1]]

            batch_y = np.array(batch_y)  

            print("current sentence: "+str(i))

            sess.run(optimizer, feed_dict={inp:batch_x,y_true:batch_y})

            hold = sess.run(cross_entropy, feed_dict={inp:batch_x,y_true:batch_y})

            tot_loss += hold

        loss_arr.append(tot_loss)

print("done with graph")

plt.plot([i for i in range(len(loss_arr))],loss_arr)
plt.xlabel("Epochs")
plt.ylabel("Loss (cross entropy)")
plt.title("Loss vs Epochs")
plt.show()

ex_str = "I can't believe the service, like, really?"

ex_str = pre_proc_sen(ex_str)

ex_str = sample2Vec(w2v_dict,ex_str)

ex_str = list(ex_str)

for i in range(max_words):
    if(i >= len(ex_str)):
        ex_str.append(np.zeros(300))
                
ex_str = np.array(ex_str)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    batch_x = np.array(list(ex_str))
    batch_x = np.reshape(batch_x,(30,300))

    hold = sess.run(y_out, feed_dict={inp:batch_x})

    print(hold[0])
    if( (hold[0][0] > hold[0][1]) and (hold[0][0] > hold[0][2]) ):
        print("negative")
    elif( (hold[0][1] > hold[0][0]) and (hold[0][1] > hold[0][2]) ):
        print("neutral")
    elif ( (hold[0][2] > hold[0][0]) and (hold[0][2] > hold[0][1])  ):
        print("positive")


print("done with graph")

