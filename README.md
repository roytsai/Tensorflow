# Tensorflow

在此路徑下的./com/tensorflow/classification/classification.py
在這個範例中我要透過tensorflow的classification來達到預測使用者行為喜好的model
在建立model的過程中很重要的兩個步驟
## 1. 資料整理
   * 在你所收集到的原始資料裡，你要決定何者為有用的input資料
   * 接著需決定input與相對應的output為何，這樣才是一筆有用的資料
   * 最後整理成n筆有用的資料後，再去traning model
	
	
#### 範例說明
   * 本範例的原始資料是，在某時間點使用了某個skill。
   * 把時間改成二進制，防止操作tf.nn.softmax()時造成w= nan ，這是因為運算式子的關係當你直接用23去算就有可能會造成此問題。
   * input我設計成第n筆資料的時間+除了自己外的前10筆資料為一筆input x。
   * 相對應的output為第n筆實際執行的skill。
	
[原始資料]
  ~~~
月, 日, 時, 音樂skill(0), 股市skill(1), 新聞skill(2), 天氣skill(3), 食譜skill(4), 星座skill(5) 
1 , 1 , 19, 1           , 0           , 0           , 0           , 0           , 0	
1 , 1 , 20, 1           , 0           , 0           , 0           , 0           , 0	
1 , 1 , 23, 1           , 0           , 0           , 0           , 0           , 0	
1 , 2 , 19, 1           , 0           , 0           , 0           , 0           , 0	
1 , 3 , 7 , 0           , 0           , 0           , 1           , 0           , 0	
  ~~~

[整理後的x]
  ~~~
[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ~~~

[整理後的y]
  ~~~
[1, 0, 0, 0, 0, 0]
  ~~~
	
## 2. Training model
   * 透過此layer可建立出，用前10次使用者行為來預測當次的使用者意圖。
  ~~~
	def add_layer(inputs, in_size, out_size, activation_function = None):
		global Weights
		global biases
		Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="Variable_W")
		biases = tf.Variable(tf.zeros([1,out_size])+0.1, name="Variable_b") 
		Wx_plus_b = tf.matmul(inputs, Weights)+biases
		Wx_plus_b = tf.nn.dropout(Wx_plus_b, 0.6)
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)    
		return outputs	
	
	prediction = add_layer(xs, 65, 6, activation_function = tf.nn.softmax)	
  ~~~
	
	
   * tensorflow有個好處可以輸出.pb，而此model可供android或ios使用，以達到devices上傳資料至server，server 訓練好模型後供devices下載使用，則不需要每次都query server。
	
  ~~~	
	_W = Weights.eval(session)
    _b = biases.eval(session)
    g_2 = tf.Graph()
    with g_2.as_default():
        x_2 = tf.placeholder(tf.float32, [None, 65], name="input")
        W_2 = tf.constant( _W, name="constant_W")
        b_2 = tf.constant( _b, name="constant_b")
        y_2 = tf.nn.softmax(tf.add(tf.matmul(x_2, W_2),b_2), name="output")
     
        sess_2 = tf.Session()
        sess_2.run(tf.global_variables_initializer())  
        graph_def = g_2.as_graph_def() 
        tf.train.write_graph(graph_def, './model',
                                         'beginner-graph.pb', as_text=False)
        sess_2.close()
  ~~~
