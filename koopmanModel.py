"""""
by lujing
lujing_cafuc@nuaa.edu.cn

"""""
import tensorflow as tf
from tensorflow.keras import layers

MAX_EPOCHS = 2

class KoopmanLayer(layers.Layer):
  def __init__(self,num_features, **kwargs):
    super(KoopmanLayer,self).__init__(**kwargs)
    # self.num_features = inputs.train_df.shape[1]
    self.num_features = num_features
    self.k = self.add_variable(
        name='Koopman',
        shape=[self.num_features, self.num_features],
        initializer='uniform',
        trainable=True)
    print('KoopmanLayer initialized,self.k=%s'%str(self.k.shape))

  def call(self,inputs):
      inputs_array=inputs
      print('inputs_array._dims:',inputs_array.shape.ndims)
      if inputs_array.shape.ndims == 3 :  ##tensor
          n_G = inputs_array.shape[1] - 1
          n_A = inputs_array.shape[1]
          print("koopman_layer called:n_G=%d n_A=%d inputs.shape=%s" % (n_G, n_A, str(inputs_array.shape)))
          G = tf.constant(0, dtype=tf.float32, shape=(inputs_array.shape[2], inputs_array.shape[2]))
          A = tf.constant(0, dtype=tf.float32, shape=(inputs_array.shape[2], inputs_array.shape[2]))
          # print("G.shape",G.shape)
          for i in range(n_G):
              x_i = inputs_array[:, i, :]
              # x_i_index=inputs
              y_i = inputs_array[:, i + 1, :]
              temp_G = tf.matmul(x_i, x_i, transpose_a=True)
              temp_A = tf.matmul(x_i, y_i, transpose_a=True)
              G = tf.add(temp_G, G)
              A = tf.add(temp_A, A)
              print("------%d x_i=%s temp_G=%s G=%s" % (i, str(x_i.shape), str(temp_G.shape), str(G.shape)))
          G = (1 / n_G) * G
          A = (1 / n_A) * A
          self.k = tf.matmul(tf.linalg.pinv(G), A)
          print('self.k.shape=', self.k.shape)
          outputs = tf.matmul(inputs, self.k)
      else:
          outputs = tf.matmul(inputs_array, self.k)
      print('inputs.shape=%s, outputs.shape=%s'%(str(inputs.shape),str(outputs.shape)))
      return outputs

class koopman_Model(tf.keras.Model):
    def __init__(self,inputs,units,**kwargs):
        super(koopman_Model, self).__init__(**kwargs)
        print("-------------Koopman_Model initialized----------------------")
        self.num_features=inputs.train_df.shape[1]
        ##定义网络
        self.layer1 = layers.Dense(units=units, activation='tanh')
        self.layer2 = layers.Dense(units=units, activation='tanh')
        self.layer3 = layers.Dense(units=units, activation='tanh')
        self.layer4 = layers.Dense(units=self.num_features)
        self.layer5 = KoopmanLayer(num_features=self.num_features)

    def call(self,inputs):
        print("koopman_Model called :")
        # print('With inputs.shape=',inputs.shape)
        x1 = self.layer1(inputs)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        outputs_dense = self.layer4(x3)
        print("With outputs=self.layer4(x3).shape=",outputs_dense.shape)
        outputs_koopman = self.layer5(outputs_dense)
        print("With outputs_k = self.layer5(outputs).shape=", outputs_koopman.shape)
        # return outputs_dense
        return outputs_koopman

def compile_and_fit(model, window, patience=2,max_epochs=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    print('Koopman model Compile starting:')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    print('Koopman model fit starting:')
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history
