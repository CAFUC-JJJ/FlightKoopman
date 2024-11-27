"""""
by lujing
lujing_cafuc@nuaa.edu.cn

"""""
################################################################
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import koopmanModel
################################################################
MAX_EPOCHS = 2
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, test_df, columns, step=1, val_df=None,
               label_columns=None):
    # Store the raw data.
    self.step = step
    self.train_df = train_df
    if val_df==None:
        self.val_df = train_df
    self.test_df = test_df
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(columns)}
    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift
    self.total_window_size = input_width + shift
    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels
WindowGenerator.split_window = split_window

def plot(self, plot_col,model=None, max_subplots=3,title=None):
  inputs, labels = self.example
  fig=plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  ax=[]
  for n in range(max_n):
    ax.append(plt.subplot(max_n, 1, n+1))
    if n == 0:
        ax[n].set_title(title)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)
    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index
    if label_col_index is None:
      continue
    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)
    if n == 0:
      plt.legend()
  plt.xlabel('Time [t]')
WindowGenerator.plot = plot

@property
def train(self):
    return self.make_dataset(self.train_df,step=self.step)

@property
def val(self):
    return self.make_dataset(self.val_df,step=self.step)

@property
def test(self):
    return self.make_dataset(self.test_df,step=self.step)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

def make_dataset(self, data,step=1):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=step,
        shuffle=True,
        batch_size=32, )
     # 2500/64=39  2500/32=78
    ds = ds.map(self.split_window)
    return ds

WindowGenerator.make_dataset = make_dataset
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
      print("Baseline called,inputs=" ,inputs.shape)
      if self.label_index is None:
          return inputs
      result = inputs[:, :, self.label_index]
      return result[:, :, tf.newaxis]

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

def myModel(train_inputdata,test_inputdata,window_width=24,step=1,colforshow=None,columns=None):
    #data
    train_df=train_inputdata
    test_df=test_inputdata
    if columns!=None:
        column_indices = {name: i for i, name in enumerate(columns)}
    num_features=train_df.shape[1]
    if colforshow!=None:
        nn=min(num_features-1,colforshow)
    else:
        nn=num_features-1
    col_for_show = columns[nn]
    print('Col_for_show:%s[x%d]'%(col_for_show,nn))
    #window
    single_step_window = WindowGenerator(
        # `WindowGenerator` returns all features as labels if you
        # don't set the `label_columns` argument.
        input_width=1, label_width=1, shift=step, train_df=train_df, test_df=test_df, columns=columns)
    # single_step_window.plot(plot_col=col_for_show)

    wide_window = WindowGenerator(input_width=window_width, label_width=window_width, shift=step, train_df=train_df, test_df=test_df, columns=columns)
    for example_inputs, example_labels in wide_window.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')
    wide_window.plot(plot_col=col_for_show,title="None Model{Col_for_show:%s[x%d]}" % (col_for_show, nn))

    #model
    val_performance = {}
    performance = {}
    score = {}
    #Baseline
    print("-------------Baseline Model initialized----------------------")
    baseline = Baseline()
    history_baseline=baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.MeanAbsoluteError()])
    val_performance['Baseline'] = baseline.evaluate(wide_window.val)
    performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)

    #Dense
    print("-------------Dense Model initialized----------------------")
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=num_features)
    ])
    history_dense = compile_and_fit(dense, wide_window)
    val_performance['Dense'] = dense.evaluate(wide_window.val)
    performance['Dense'] = dense.evaluate(wide_window.test, verbose=0)
    wide_window.plot(model=dense,plot_col=col_for_show,title="Dense_Model{Col_for_show:%s[x%d]}"%(col_for_show,nn))

    #Lstm
    print("-------------LSTM Model initialized----------------------")
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=num_features)
    ])
    history_lstm = compile_and_fit(lstm_model, wide_window)
    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
    wide_window.plot(model=lstm_model, plot_col=col_for_show,title="Lstm_Model{Col_for_show:%s[x%d]}"%(col_for_show,nn))

    #Koopman
    koopman = koopmanModel.koopman_Model(inputs=wide_window,units=64)
    history_koopman = koopmanModel.compile_and_fit(koopman,wide_window,max_epochs=MAX_EPOCHS)
    val_performance['Koopman'] = koopman.evaluate(wide_window.val)
    performance['Koopman'] = koopman.evaluate(wide_window.test, verbose=0)
    wide_window.plot(model=koopman,plot_col=col_for_show,title="Koopman_Model{Col_for_show:%s[x%d]}"%(col_for_show,nn))

    #mae
    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    metric_index = dense.metrics_names.index('mean_absolute_error')
    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in performance.values()]

    #Plot
    plt.figure(figsize=(12, 8))
    plt.ylabel('mean_absolute_error [normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
               rotation=45)
    _ = plt.legend()

    predictions_train = koopman(train_df)
    predictions_test = koopman(test_df)

    return predictions_train,predictions_test
