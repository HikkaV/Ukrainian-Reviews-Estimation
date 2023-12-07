# %%
!pip install tokenizers
import tensorflow as tf
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, Regex
import tokenizers
import pandas as pd

# %%
model_name = 'deep_lstm_attention_w2v_huber'

# %%
"""
# Load data
"""

# %%
df = pd.read_csv('/home/user/files_for_research_Vova/processed_data.csv',\
                 usecols=['review_translate',
                                                            'dataset_name',
                                                            'rating',
                                                           'translated'])

# %%
df.head()

# %%
subsets = pd.read_csv('/home/user/files_for_research_Vova/train_val_test_indices.csv')

# %%
subsets.head()

# %%
subsets = subsets.merge(df[['dataset_name', 'translated']], left_on='index', right_index=True)

# %%
"""
# Filter data
"""

# %%
bad_indices = pd.read_csv('/home/user/files_for_research_Vova/files_to_check.csv')

# %%
subsets = subsets[~subsets.index.isin(bad_indices['id'].values)]

# %%
df = df[~df.index.isin(bad_indices['id'].values)]

# %%
df, subsets = df.reset_index().drop(columns='index'), subsets.reset_index().drop(columns='index')

# %%
"""
# Load tokenizer
"""

# %%
tokenizer = Tokenizer(models.BPE.from_file(vocab='/home/user/files_for_research_Vova/tokenizer_30k.json',
            merges='/home/user/files_for_research_Vova/merges_tokenizer.txt',
                                          end_of_word_suffix='</w>'))
tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\w'-]+|[^\w\s'-]+"),'removed', True)

# %%
"""
# Encode text
"""

# %%
import seaborn as sns
import numpy as np

# %%
sns.set()

# %%
df['review_translate'] = df['review_translate'].str.lower()

# %%
df['encoded'] = tokenizer.encode_batch(df['review_translate'].values)

# %%
df['encoded'] = df['encoded'].apply(lambda x: x.ids)

# %%
sns.distplot(np.log10(df['encoded'].apply(len)))

# %%
np.percentile(df['encoded'].apply(len), 99)

# %%
encoded_tokens = df['encoded'].values

# %%
from itertools import chain

# %%
padded_tokens = tf.keras.preprocessing.sequence\
.pad_sequences(encoded_tokens, maxlen=300, padding="post")


# %%
padded_tokens.shape

# %%
"""
# Get embeddings
"""

# %%
!pip install gensim

# %%
import gensim

# %%
def load_w2vec(path, vocab, embed_dim=300, glove_backup={}):
    vectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    emb_matrix = np.zeros(shape = (len(vocab) + 1, embed_dim))
    missed = 0
    for word, idx in vocab.items():
        if idx!=0:
            try:
                emb_matrix[idx,:] = vectors[word]
            except KeyError:
                if glove_backup:
                    try:
                        emb_matrix[idx,:] = glove_backup[word]
                    except:
                        missed+=1
                else:
                    missed+=1
    print(f'Missed words : {missed}')
    return emb_matrix, vectors

# %%
emb_matrix, vectors = load_w2vec('/home/user/files_for_research_Vova/embeddings_w2v.bin',
                                tokenizer.get_vocab())

# %%
"""
# Get labels and split data
"""

# %%
mapping = dict([(i,c) for c,i in enumerate(df['rating'].unique())])

# %%
mapping

# %%
y = df['rating'].map(mapping).values

# %%
num_classes = len(set(y))

# %%
train_indices, val_indices, test_indices = subsets[subsets['split']=='train'].index.tolist(),\
subsets[subsets['split']=='val'].index.tolist(),\
subsets[subsets['split']=='test'].index.tolist()


# %%
train_y, val_y, test_y = y[train_indices], y[val_indices], y[test_indices]

# %%
train_x, val_x, test_x = padded_tokens[train_indices], padded_tokens[val_indices],\
padded_tokens[test_indices]

# %%
train_x.shape

# %%
"""
# Create  model
"""

# %%
class Attention(tf.keras.layers.Layer):
    def __init__(self,  
                 units=128, **kwargs):
        super(Attention,self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W1=self.add_weight(name='attention_weights_1', shape=(input_shape[-1], self.units), 
                               initializer='glorot_uniform', trainable=True)
        
        self.W2=self.add_weight(name='attention_weights_2', shape=(1, self.units), 
                               initializer='glorot_uniform', trainable=True) 
        
        super(Attention, self).build(input_shape)
        
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        attention = tf.nn.softmax(tf.matmul(self.W2, tf.nn.tanh(tf.matmul(self.W1, x))))
        weighted_context = tf.reduce_sum(x * attention, axis=-1)
        return weighted_context, attention
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units
        })
        return config


# %%
tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)
# define layers
attention = Attention(units=128, name='attention')
input_layer = tf.keras.layers.Input(shape=(300,), name='input')
word_embedding = tf.keras.layers.Embedding(input_dim=tokenizer.get_vocab_size()+1,
                                                   output_dim=300,
                                                   trainable=True,
                                           name='embedding',
                                           mask_zero=True,
                                                   weights=[emb_matrix])
batch_norm = tf.keras.layers.LayerNormalization(axis=-1)
spatial_dropout = tf.keras.layers.SpatialDropout1D(0.3, name='spatial_dropout')
lstm1 = tf.keras.layers.LSTM(256, name='lstm1',
                            return_sequences=True)
lstm2 = tf.keras.layers.LSTM(128, name='lstm2',
                            return_sequences=True, return_state=True)
dense1 = tf.keras.layers.Dense(128, activation='relu', name='dense')
dropout = tf.keras.layers.Dropout(0.5, name='dropout')
logits_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')

#actual flow
embedded = spatial_dropout(word_embedding(input_layer))
lstm_lvl1 = lstm1(embedded)
normed = batch_norm(lstm_lvl1)
context_vector, state_h, _ = lstm2(normed)
weighted_context, attention_scores = attention(context_vector)
final_attn_output = tf.concat([state_h, weighted_context], axis=1)
x = dense1(final_attn_output)
x = dropout(x)
x = logits_layer(x)
model = tf.keras.Model(input_layer, x)

# %%
"""
# Compile model
"""

# %%
model.compile(loss=tf.keras.losses.Huber(1.0), \
              optimizer=tf.keras.optimizers.Adam(),
             metrics=['acc'])

# %%
"""
# Early stopping
"""

# %%
import operator
class EarlyStopping:
    def __init__(self, tolerance=5, mode='min'):
        assert mode in ['min','max'], 'Mode should be min or max'
        self.mode = operator.lt if mode=='min' else operator.gt 
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.extremum_value = None
        self.best_model = None
    
    @staticmethod
    def copy_model(model):
        copied_model = tf.keras.models.clone_model(model)
        copied_model.set_weights(model.get_weights())
        return copied_model
        
    def __call__(self, val, model):
        if self.extremum_value is None:
            self.extremum_value = val
            self.best_model = self.copy_model(model)
        else:
            if not self.mode(val, self.extremum_value):
                self.counter+=1
            else:
                self.extremum_value = val
                self.best_model = self.copy_model(model)
                self.counter = 0
        
        if self.counter==self.tolerance:
            self.early_stop=True

# %%
"""
# Train model
"""

# %%
from sklearn.metrics import f1_score

# %%
def evaluate_on_datasets(y_true, y_pred, split='val'):
    d = {}
    for dataset_name in subsets['dataset_name'].unique():
            idx = subsets[subsets['split']==split].copy()
            idx['index'] = list(range(idx.shape[0]))
            idx = idx[(idx['dataset_name']==dataset_name)]\
            ['index'].values.tolist()
            score = f1_score(y_true=y_true[idx], y_pred=y_pred[idx],
                                 average='macro')
            print(f'{split} f1 score for dataset {dataset_name} : {score}')
            d[f'{split}_f1_{dataset_name}'] = score
            
    for flag in [True, False]:
        idx = subsets[subsets['split']==split].copy()
        idx['index'] = list(range(idx.shape[0]))
        idx = idx[idx['translated']==flag]['index'].values.tolist()
        score = f1_score(y_true=y_true[idx], y_pred=y_pred[idx],
                                 average='macro')
        print(f'{split} f1 score for translated=={flag} : {score}')
        d[f'{split}_f1_translated=={flag}'] = score
    return d

# %%
def update_history(history, d):
    for key, value in d.items():
        res = history.get(key, [])
        res.append(value)
        history[key] = res

# %%
early_stopping = EarlyStopping(mode='max', tolerance=4)

# %%
def training_loop(model, train_x, train_y, val_x, val_y, epochs=10, batch_size=128,
                 shuffle=True):
    dict_history = {}
    for i in range(epochs):
        if shuffle and i==0:
            indices = np.arange(len(train_x))
            np.random.shuffle(indices)
            train_x = train_x[indices]
            train_y = train_y[indices]
            
        #train model
        history = model.fit(train_x,tf.one_hot(train_y,num_classes), \
                            validation_data=(val_x,tf.one_hot(val_y,num_classes)), 
          epochs=1, batch_size=batch_size,
                           verbose=0, shuffle=False)
        train_loss, val_loss = history.history['loss'][-1], history.history['val_loss'][-1]
        
        #evaluate model
        train_prediction = np.argmax(model.predict(train_x, batch_size=batch_size), axis=-1)
        val_prediction = np.argmax(model.predict(val_x, batch_size=batch_size), axis=-1)
        train_f1 = f1_score(y_true=train_y, y_pred=train_prediction,
                           average='macro')
        val_f1 = f1_score(y_true=val_y, y_pred=val_prediction,
                         average='macro')
        
        #printing evaluation
        print(f'Epoch {i}')
        print(f'Overall train f1 : {train_f1}, overall val f1: {val_f1}')
        print(f'Train loss : {train_loss}, val loss: {val_loss}')
        d_train = evaluate_on_datasets(y_true=train_y, y_pred=train_prediction, split='train')
        d_val = evaluate_on_datasets(y_true=val_y, y_pred=val_prediction, split='val')
            
        if i!=epochs-1:
            print('-'*30)
            
        #save history
        update_history(dict_history, d_train)
        update_history(dict_history, d_val)
        update_history(dict_history, {'train_f1': train_f1})
        update_history(dict_history, {'val_f1': val_f1})
        update_history(dict_history, {'train_loss': train_loss})
        update_history(dict_history, {'val_loss': val_loss})
        #early stopping
        
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print('Stopping early')
            model = early_stopping.best_model
            break
        
    return dict_history, model

# %%
dict_history, model = \
training_loop(model, train_x, train_y, 
              val_x, val_y, epochs=20, batch_size=2048, shuffle=True)

# %%
dict_history

# %%
"""
# Show charts
"""

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
def plot_history(dict_history, columns):
    plt.figure(figsize=(12,8))
    for i in columns:
        to_plot = dict_history[i]
        plt.plot(range(len(to_plot)), to_plot, 'o-')
    plt.xticks(range(len(to_plot)), range(len(to_plot)))
    plt.xlabel('Epochs')
    plt.legend(columns)

# %%
plot_history(dict_history, ['val_loss', 'train_loss'])

# %%
plot_history(dict_history, ['val_f1', 'train_f1'])

# %%
"""
# Evaluate model
"""

# %%
test_predictions = np.argmax(model.predict(test_x, batch_size=2048), axis=-1)

# %%
test_f1 = f1_score(y_true=test_y, y_pred=test_predictions,
                         average='macro')
print(f'Overall test f1-score : {test_f1}')

# %%
test_results = evaluate_on_datasets(y_true=test_y, y_pred=test_predictions,split='test')
                     

# %%
"""
# Confusion matrix
"""

# %%
inverse_mapping = dict([(v,k) for k,v in mapping.items()])

# %%
from sklearn.metrics import confusion_matrix

# %%
np.unique(test_y)

# %%
matrix = confusion_matrix(test_y, test_predictions)
matrix_scaled = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14,10))
sns.heatmap(matrix_scaled, annot=True, cmap=plt.cm.Blues, xticklabels=[inverse_mapping[i] for i in np.unique(test_y)],\
            yticklabels=[inverse_mapping[i] for i in np.unique(test_y)])
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.show()

# %%
test_df = df[subsets['split']=='test'].copy()

# %%
test_df['predicted_rating'] = [inverse_mapping[i] for i in test_predictions]

# %%
"""
# Save history results
"""

# %%
history = pd.DataFrame(dict_history)
for k,v in test_results.items():
    history[k] = v

# %%
history['model'] = model_name

# %%
history.to_csv("/home/user/jupyter_notebooks/Ukranian-SA/notebooks/training/training_results_filtered.csv", mode='a', header=None, index=None)

# %%
"""
# Save model
"""

# %%
model.save(f'/home/user/files_for_research_Vova/{model_name}.h5')

# %%
