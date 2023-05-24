# Ukranian-Reviews-Estimation
Rating explanation through the usage of explainable AI based on trained models for rating estimation. 

Dataset: https://huggingface.co/datasets/vkovenko/cross_domain_uk_reviews.

Model for rating estimation: https://huggingface.co/vkovenko/deep_lstm_attention_ukr_reviews_rating_estimation.

How to load the tokenizer:

```python  
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, Regex
import tokenizers
from tokenizers import Tokenizer, models, decoders, processors
from tokenizers import pre_tokenizers, trainers, Regex
import huggingface_hub

class BPETokenizer:
    def __init__(self, vocab, merges):
        self.suffix = '</w>'
        self.tokenizer = Tokenizer(models.BPE.from_file(vocab=vocab,
            merges=merges, end_of_word_suffix=self.suffix))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\w'-]+|[^\w\s'-]+"),'removed', True)
        self.id_to_token = self.tokenizer.id_to_token
        self.encode_batch = self.tokenizer.encode_batch
        self.token_to_id = self.tokenizer.token_to_id
        self.encode = self.tokenizer.encode
        
    def tokens_to_ids(self, tokens):
        return list(map(self.token_to_id, tokens))
    
    def ids_to_tokens(self, ids):
        return list(map(self.id_to_token, ids))
        

    def decode(self, tokens, return_indices=False):
        decoded = []
        merged_indices = []
        i = 0
        while i<len(tokens):
            if tokens[i].endswith(self.suffix):
                decoded.append(tokens[i])
                merged_indices.append([i])
                i+=1
            else:
                merged_token = ''
                tmp_indc = []
                while not tokens[i].endswith(self.suffix):
                    merged_token+=tokens[i]
                    tmp_indc.append(i)
                    i+=1
                merged_token+=tokens[i]
                tmp_indc.append(i)
                decoded.append(merged_token)
                merged_indices.append(tmp_indc)
                i+=1
                
        if return_indices:
            return decoded, merged_indices
        else:
            return decoded
#download tokenizer
tokenizer = BPETokenizer(vocab=huggingface_hub.hf_hub_download('vkovenko/deep_lstm_attention_ukr_reviews_rating_estimation',
                                'tokenizer_30k.json',
                               local_dir='model'),
            merges=huggingface_hub.hf_hub_download('vkovenko/deep_lstm_attention_ukr_reviews_rating_estimation',
                                'merges_tokenizer.txt',
                               local_dir='model')
                        )

```

How to laod model for rating estimation:

```python  
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

#download model
model = tf.keras.models.load_model(huggingface_hub.hf_hub_download('vkovenko/deep_lstm_attention_ukr_reviews_rating_estimation',
                                'deep_lstm_attention_w2v_huber.h5',
                               local_dir='model'),
                                  compile=False,
                                  custom_objects={'Attention':Attention})
```

The repository is part of shevchenko.ai project.
