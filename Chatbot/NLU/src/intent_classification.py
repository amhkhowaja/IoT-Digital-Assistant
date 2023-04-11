#!/usr/bin/env python

'''

Different model training and their analysis

'''

# importing the libraries
import numpy as np
from tensorflow.keras import models, layers, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import data_generation

# tokenization

df = data_generation.import_dataframe("../data/final_data.csv")

# For Intent classification we only need Intent and question
df = df[["Questions", "Intent"]]
features = df["Questions"]
labels = df["Intent"]

le = LabelEncoder()
labels = le.fit_transform(labels)

# training testing split
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
#%%

label_count = df["Intent"].value_counts()
sns.barplot(label_count.index, label_count)
plt.gca().set_ylabel("samples")

#%%


tokenizer = Tokenizer(num_words=120, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

#%%
# padding train set
training_seq = tokenizer.texts_to_sequences(x_train)
train_padded = pad_sequences(training_seq, maxlen=8, padding="post", truncating="post")
# padding validation set
validation_seq = tokenizer.texts_to_sequences(x_val)
validation_padded = pad_sequences(
    validation_seq, maxlen=8, padding="post", truncating="post"
)
# padding test set
testing_seq = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_seq, maxlen=8, padding="post", truncating="post")

#%%
# Bidirectional LSTM
# setting up the values of some parameters
vocab_size, embedding_dim, max_length = (120, 8, 8)
bilstm = models.Sequential(
    [
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.Bidirectional(layers.LSTM(embedding_dim)),
        layers.Dense(5, activation="softmax"),
    ]
)
# compilation of the model
#%%
bilstm.compile(
    loss="sparse_categorical_crossentropy", metrics="accuracy", optimizer="adam"
)
# callbacks
early_stop = callbacks.EarlyStopping(patience=4)
csv_log_bilstm = callbacks.CSVLogger(
    "../models/history/bilstm_history.csv", append=False
)
callback = [early_stop]
#%%
# model training
print(train_padded)
print(y_train)
print(type(train_padded[0][0]))
print(type(train_padded[0][0]))
print([(i.shape, i.dtype) for i in bilstm.inputs])
print([(o.shape, o.dtype) for o in bilstm.outputs])
print([(l.name, l.input_shape, l.dtype) for l in bilstm.layers])

history = bilstm.fit(
    train_padded,
    y_train,
    epochs=100,
    callbacks=callback,
    validation_data=(validation_padded, y_val),
    shuffle=True,
    #batch_size=8
)
models.save_model(bilstm, "../models/bilstm.h5")
#%%
bilstm = models.load_model("../models/bilstm.h5")
#%%
# loss curves and accuracy curves
def plot_train_val_analysis(history, title):
    epoch_range = range(len(history.history["loss"]))
    # validation loss and training loss
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title(title + " Loss graph")
    plt.plot(epoch_range, history.history["loss"], label="Training Loss")
    plt.plot(epoch_range, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title(title + " Accuracy graph")
    plt.plot(epoch_range, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epoch_range, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("../plots/" + title + "_loss_acc_graph")


plot_train_val_analysis(history, "BILSTM")
#%%
# Prediction
y_pred = bilstm.predict(testing_padded)
y_pred = [np.argmax(x) for x in y_pred]

#%%
# confusion Matrix


def generate_conf_matrix(y_test, y_pred, title):
    disp = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(disp, annot=True, cmap="Blues").set(
        title="Confusion Matrix of " + title + "model"
    )
    plt.savefig("../plots/" + title + "_confusion_matrix")


generate_conf_matrix(y_test, y_pred, "BILSTM")
#%%
# Convolutional Neural Network CNN 1D layer
vocab_size, embedding_dim, max_length = (120, 8, 8)
cnn = models.Sequential(
    [
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.Conv1D(16, 8, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(5, activation="softmax"),
    ]
)
#%%
# compilation ofcnn
cnn.compile(
    loss="sparse_categorical_crossentropy", metrics="accuracy", optimizer="adam"
)
#%%
csv_log_cnn = callbacks.CSVLogger("../models/history/cnn_history.csv", append=False)
history2 = cnn.fit(
    train_padded,
    y_train,
    epochs=100,
    callbacks=callback,
    validation_data=(validation_padded, y_val),
)
models.save_model(cnn, "../models/cnn.h5")
cnn = models.load_model("../models/cnn.h5")
#%%
plot_train_val_analysis(history2, "CNN")
cnn_pred = cnn.predict(testing_padded)
cnn_pred = [np.argmax(x) for x in cnn_pred]
#%%
generate_conf_matrix(y_test, cnn_pred, "CNN")
#%%
# lets test some manual tests:


def analyze_sentence(model, sentences):
    sequence = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequence, maxlen=8, padding="post", truncating="post")
    y_pred = model.predict(padded)
    y_pred = [np.argmax(x) for x in y_pred]
    classes = le.classes_
    y_pred = [classes[int(x)] for x in y_pred]
    return y_pred


#%% CNN
sentences = [
    "how can i add filter?",
    "is my billing state active?",
    "can i customize my view?",
    "list all of my billing states",
    "Why network connectivity giving me error",
    "what is my name ?",
]
print(analyze_sentence(cnn, sentences))
#%%
print(analyze_sentence(bilstm, sentences))
