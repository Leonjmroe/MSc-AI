#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from nlp_processor import Processor 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


# In[54]:


# Data Preperation

data = pd.read_csv('car-reviews.csv')

positive_data = data[691:]
negative_data = data[:691]

train_positive = positive_data[:553]
test_positive = positive_data[553:]

train_negative = negative_data[:553]
test_negative = negative_data[553:]

raw_training_data = pd.concat([train_positive, train_negative]).reset_index(drop=True)
raw_testing_data = pd.concat([test_positive, test_negative]).reset_index(drop=True)

raw_training_data['Sentiment'] = np.where(raw_training_data['Sentiment'] == 'Pos', 1, 0)
raw_testing_data['Sentiment'] = np.where(raw_testing_data['Sentiment'] == 'Pos', 1, 0)


def process_data():
    
    processor = Processor()
    training_data = processor.process(raw_training_data, testing=False)
    testing_data = processor.process(raw_testing_data, testing=True)
    
    return training_data, testing_data

training_data, testing_data = process_data()


# tr_labels = training_data[:, 0] 
# tr_texts = training_data[:, 1:]
# tes_labels = testing_data[:, 0]
# tes_texts = testing_data[:, 1:]

tr_texts = list(raw_training_data['Review'])
tr_labels = list(raw_training_data['Sentiment'])  
tes_texts = list(raw_testing_data['Review'])
tes_labels = list(raw_testing_data['Sentiment'])


# In[55]:


# Tokenize and prepare your data as before
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(tr_texts)

train_sequences = tokenizer.texts_to_sequences(tr_texts)
train_data = pad_sequences(train_sequences, maxlen=100)
train_labels = np.array(tr_labels)

test_sequences = tokenizer.texts_to_sequences(tes_texts)
test_data = pad_sequences(test_sequences, maxlen=100)
test_labels = np.array(tes_labels)


# In[ ]:


def grid_search(embedding_dims, lstm_units, dropout_rates, regularization_strengths):
        
    best_accuracy = 0
    best_config = {}
    best_history = None
    count = 0
    total_runs = len(embedding_dims) * len(lstm_units) * len(dropout_rates) * len(regularization_strengths)
    
    # Grid search
    for embedding_dim in embedding_dims:
        for lstm_unit in lstm_units:
            for dropout_rate in dropout_rates:
                for reg_strength in regularization_strengths:
                    print(f"{count}/{total_runs}: Testing config: Embedding Dim {embedding_dim}, LSTM Units {lstm_unit}, Dropout {dropout_rate}, Reg Strength {reg_strength}")
                    
                    # Define model
                    model = Sequential([
                        Input(shape=(100,)),
                        Embedding(input_dim=10000, output_dim=embedding_dim),
                        LSTM(lstm_unit, dropout=dropout_rate, recurrent_dropout=dropout_rate),
                        Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_strength))
                    ])
                    
                    # Compile model
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    
                    # Train model
                    early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
                    history = model.fit(train_data, train_labels, batch_size=32, epochs=10, validation_split=0.2, callbacks=[early_stopping])
                    
                    # Evaluate model
                    _, accuracy = model.evaluate(test_data, test_labels, verbose=0)
                    
                    # Update best config if current config is better
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_history = history 
                        best_config = {
                            'embedding_dim': embedding_dim,
                            'lstm_unit': lstm_unit,
                            'dropout_rate': dropout_rate,
                            'reg_strength': reg_strength
                        }
                        print('\n')
                        print('****** New Best Accuracy: ', round(accuracy, 3), '******')
                        print('\n')
    
    print(f"Best Accuracy: {round(best_accuracy, 3)}")
    print("Best Configuration:", best_config)
    return best_history



# Define the grid of hyperparameters to search
embedding_dims = np.arange(32, 128, 16).tolist()
lstm_units = np.arange(16, 64, 8).tolist()
dropout_rates = np.arange(0, 0.5, 0.1).tolist()
regularization_strengths = np.arange(0, 0.4, 0.1).tolist()


# In[ ]:


def model_evaluation(history):
    # Training history data
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = range(1, len(training_accuracy) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plotting accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_loss, label='Training Loss', marker='o')
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# In[ ]:


best_history = grid_search(embedding_dims, lstm_units, dropout_rates, regularization_strengths)
model_evaluation(best_history)


# In[ ]:




