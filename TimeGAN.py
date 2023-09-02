<font size="+3">Time-series Generative Adversarial Network (TimeGAN)</font>
# Imports & Settings
Adapted from the excellent paper by Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar:  
[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks),  
Neural Information Processing Systems (NeurIPS), 2019.

- Last updated Date: April 24th 2020
- [Original code](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/timegan/) author: Jinsung Yoon (jsyoon0823@gmail.com)
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import seaborn as sns
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")
sns.set_style("white")
# Experiment Path
# Prepare Data
## Parameters
df = pd.read_csv("Full_CrossCheck_Hourly_Data.csv")
df = df.set_index("study_id").sort_index()
df = df.drop(["hour", "day", "filled_pct", "weighted_distance_sum"], axis=1)
df.head()
seq_len = 24
n_seq = 27
batch_size = 128
# make a list of the columns in the dataframe
cols = df.columns.tolist()
## Plot Series
axes = df.div(df.iloc[0]).plot(
    subplots=True,
    figsize=(14, 6),
    layout=(14, 2),
    title=cols,
    legend=False,
    rot=0,
    lw=1,
    color="k",
)
for ax in axes.flatten():
    ax.set_xlabel("")

plt.suptitle("Normalized Sensor Series")
plt.gcf().tight_layout()
sns.despine()
## Correlation
sns.clustermap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(h_neg=20, h_pos=220),
    center=0,
)
## Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df).astype(np.float32)
## Create rolling window sequences
data = []
for i in range(len(df) - seq_len):
    data.append(scaled_data[i : i + seq_len])

n_windows = len(data)
## Create tf.data.Dataset
real_series = (
    tf.data.Dataset.from_tensor_slices(data)
    .shuffle(buffer_size=n_windows)
    .batch(batch_size)
)
real_series_iter = iter(real_series.repeat())
## Set up random series generator
def make_random_data():
    while True:
        yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))
We use the Python generator to feed a `tf.data.Dataset` that continues to call the random number generator as long as necessary and produces the desired batch size.
random_series = iter(
    tf.data.Dataset.from_generator(make_random_data, output_types=tf.float32)
    .batch(batch_size)
    .repeat()
)
# TimeGAN Components
The design of the TimeGAN components follows the author's sample code.
##  Network Parameters
hidden_dim = 43
num_layers = 3
## Input place holders
X = Input(shape=[seq_len, n_seq], name="RealData")
Z = Input(shape=[seq_len, n_seq], name="RandomData")
## RNN block generator
We keep it very simple and use a very similar architecture for all four components. For a real-world application, they should be tailored to the data.
def make_rnn(n_layers, hidden_units, output_units, name):
    return Sequential(
        [
            GRU(units=hidden_units, return_sequences=True, name=f"GRU_{i + 1}")
            for i in range(n_layers)
        ]
        + [Dense(units=output_units, activation="sigmoid", name="OUT")],
        name=name,
    )
## Embedder & Recovery
embedder = make_rnn(
    n_layers=3, hidden_units=hidden_dim, output_units=hidden_dim, name="Embedder"
)
recovery = make_rnn(
    n_layers=3, hidden_units=hidden_dim, output_units=n_seq, name="Recovery"
)
## Generator & Discriminator
generator = make_rnn(
    n_layers=3, hidden_units=hidden_dim, output_units=hidden_dim, name="Generator"
)
discriminator = make_rnn(
    n_layers=3, hidden_units=hidden_dim, output_units=1, name="Discriminator"
)
supervisor = make_rnn(
    n_layers=2, hidden_units=hidden_dim, output_units=hidden_dim, name="Supervisor"
)
# TimeGAN Training
## Settings
train_steps = 50000
gamma = 1
## Generic Loss Functions
mse = MeanSquaredError()
bce = BinaryCrossentropy()
# Phase 1: Autoencoder Training
## Architecture
H = embedder(X)
X_tilde = recovery(H)

autoencoder = Model(inputs=X, outputs=X_tilde, name="Autoencoder")
autoencoder.summary()
plot_model(autoencoder)
## Autoencoder Optimizer
autoencoder_optimizer = Adam()
## Autoencoder Training Step
@tf.function
def train_autoencoder_init(x):
    with tf.GradientTape() as tape:
        x_tilde = autoencoder(x)
        embedding_loss_t0 = mse(x, x_tilde)
        e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

    var_list = embedder.trainable_variables + recovery.trainable_variables
    gradients = tape.gradient(e_loss_0, var_list)
    autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
    return tf.sqrt(embedding_loss_t0)
## Autoencoder Training Loop
for step in tqdm(range(train_steps)):
    X_ = next(real_series_iter)
    step_e_loss_t0 = train_autoencoder_init(X_)
## Persist model
# autoencoder.save(log_dir / 'autoencoder')
# Phase 2: Supervised training
## Define Optimizer
supervisor_optimizer = Adam()
## Train Step
@tf.function
def train_supervisor(x):
    with tf.GradientTape() as tape:
        h = embedder(x)
        h_hat_supervised = supervisor(h)
        g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

    var_list = supervisor.trainable_variables
    gradients = tape.gradient(g_loss_s, var_list)
    supervisor_optimizer.apply_gradients(zip(gradients, var_list))
    return g_loss_s
## Training Loop
for step in tqdm(range(train_steps)):
    X_ = next(real_series_iter)
    step_g_loss_s = train_supervisor(X_)
## Persist Model
# supervisor.save(log_dir / 'supervisor')
# Joint Training
## Generator
### Adversarial Architecture - Supervised
E_hat = generator(Z)
H_hat = supervisor(E_hat)
Y_fake = discriminator(H_hat)

adversarial_supervised = Model(
    inputs=Z, outputs=Y_fake, name="AdversarialNetSupervised"
)
adversarial_supervised.summary()
plot_model(adversarial_supervised, show_shapes=True)
### Adversarial Architecture in Latent Space
Y_fake_e = discriminator(E_hat)

adversarial_emb = Model(inputs=Z, outputs=Y_fake_e, name="AdversarialNet")
adversarial_emb.summary()
plot_model(adversarial_emb, show_shapes=True)
### Mean & Variance Loss
X_hat = recovery(H_hat)
synthetic_data = Model(inputs=Z, outputs=X_hat, name="SyntheticData")
synthetic_data.summary()
plot_model(synthetic_data, show_shapes=True)
def get_generator_moment_loss(y_true, y_pred):
    y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
    y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
    g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
    g_loss_var = tf.reduce_mean(
        tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6))
    )
    return g_loss_mean + g_loss_var
## Discriminator
### Architecture: Real Data
Y_real = discriminator(H)
discriminator_model = Model(inputs=X, outputs=Y_real, name="DiscriminatorReal")
discriminator_model.summary()
plot_model(discriminator_model, show_shapes=True)
## Optimizers
generator_optimizer = Adam()
discriminator_optimizer = Adam()
embedding_optimizer = Adam()
## Generator Train Step
@tf.function
def train_generator(x, z):
    with tf.GradientTape() as tape:
        y_fake = adversarial_supervised(z)
        generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake), y_pred=y_fake)

        y_fake_e = adversarial_emb(z)
        generator_loss_unsupervised_e = bce(
            y_true=tf.ones_like(y_fake_e), y_pred=y_fake_e
        )
        h = embedder(x)
        h_hat_supervised = supervisor(h)
        generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_hat = synthetic_data(z)
        generator_moment_loss = get_generator_moment_loss(x, x_hat)

        generator_loss = (
            generator_loss_unsupervised
            + generator_loss_unsupervised_e
            + 100 * tf.sqrt(generator_loss_supervised)
            + 100 * generator_moment_loss
        )

    var_list = generator.trainable_variables + supervisor.trainable_variables
    gradients = tape.gradient(generator_loss, var_list)
    generator_optimizer.apply_gradients(zip(gradients, var_list))
    return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss
## Embedding Train Step
@tf.function
def train_embedder(x):
    with tf.GradientTape() as tape:
        h = embedder(x)
        h_hat_supervised = supervisor(h)
        generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_tilde = autoencoder(x)
        embedding_loss_t0 = mse(x, x_tilde)
        e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

    var_list = embedder.trainable_variables + recovery.trainable_variables
    gradients = tape.gradient(e_loss, var_list)
    embedding_optimizer.apply_gradients(zip(gradients, var_list))
    return tf.sqrt(embedding_loss_t0)
## Discriminator Train Step
@tf.function
def get_discriminator_loss(x, z):
    y_real = discriminator_model(x)
    discriminator_loss_real = bce(y_true=tf.ones_like(y_real), y_pred=y_real)

    y_fake = adversarial_supervised(z)
    discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake), y_pred=y_fake)

    y_fake_e = adversarial_emb(z)
    discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e), y_pred=y_fake_e)
    return (
        discriminator_loss_real
        + discriminator_loss_fake
        + gamma * discriminator_loss_fake_e
    )
@tf.function
def train_discriminator(x, z):
    with tf.GradientTape() as tape:
        discriminator_loss = get_discriminator_loss(x, z)

    var_list = discriminator.trainable_variables
    gradients = tape.gradient(discriminator_loss, var_list)
    discriminator_optimizer.apply_gradients(zip(gradients, var_list))
    return discriminator_loss
## Training Loop
step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
file_writer = tf.summary.create_file_writer("logs/fit")
file_writer.set_as_default()
for step in range(train_steps):
    # Train generator (twice as often as discriminator)
    for kk in range(2):
        X_ = next(real_series_iter)
        Z_ = next(random_series)

        # Train generator
        step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_)
        tf.summary.scalar("Generator loss unsupervised", data=step_g_loss_u, step=step)
        tf.summary.scalar("Generator loss supervised", data=step_g_loss_s, step=step)
        tf.summary.scalar("Generator loss variance", data=step_g_loss_v, step=step)
        # Train embedder
        step_e_loss_t0 = train_embedder(X_)
        tf.summary.scalar("Embedder loss", data=step_e_loss_t0, step=step)

    X_ = next(real_series_iter)
    Z_ = next(random_series)
    step_d_loss = get_discriminator_loss(X_, Z_)
    tf.summary.scalar("Discriminator loss", data=step_d_loss, step=step)
    if step_d_loss > 0.15:
        step_d_loss = train_discriminator(X_, Z_)

    if step % 1000 == 0:
        print(
            f"{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | "
            f"g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}"
        )
    tf.summary.flush()
## Persist Synthetic Data Generator
log_dir = Path("logs/model")
synthetic_data.save(log_dir / "synthetic_data")
# Generate Synthetic Data
generated_data = []
for i in range(int(n_windows / batch_size)):
    Z_ = next(random_series)
    d = synthetic_data(Z_)
    generated_data.append(d)
len(generated_data)
generated_data = np.array(np.vstack(generated_data))
generated_data.shape
log_dir = Path("logs/data")
import os

# Create the directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
np.save(log_dir / "generated_data.npy", generated_data)
## Rescale
generated_data = scaler.inverse_transform(generated_data.reshape(-1, n_seq)).reshape(
    -1, seq_len, n_seq
)
generated_data.shape
## Persist Data
# make a list of the columns in the dataframe
cols = df.columns.tolist()
pd.DataFrame(generated_data.reshape(-1, n_seq), columns=cols).to_csv("synthetic.csv")
## Plot sample Series
## Plot Synthetic Overlapping Real Data Line Plots for Each of the 27 Features

fig, axes = plt.subplots(nrows=14, ncols=2, figsize=(14, 50))
axes = axes.flatten()

index = list(range(1, 25))
synthetic = generated_data[np.random.randint(n_windows)]

idx = np.random.randint(len(df) - seq_len)
real = df.iloc[idx : idx + seq_len]

for j, col in enumerate(cols):
    ax = axes[j]
    pd.DataFrame({"Real": real.iloc[:, j].values, "Synthetic": synthetic[:, j]}).plot(
        ax=ax, title=col, lw=1
    )
    ax.legend()

sns.despine()
fig.tight_layout(pad=2.0)
<font size="+3">Visualize Real and Synthetic Data</font>
seq_len = 24
n_seq = 27
# Load Data
def get_real_data():
    df = pd.read_csv("Full_CrossCheck_Hourly_Data.csv")
    df = df.set_index("study_id").sort_index()
    df = df.drop(["hour", "day", "filled_pct"], axis=1)

    # Preprocess the dataset:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    data = []
    for i in range(len(df) - seq_len):
        data.append(scaled_data[i : i + seq_len])
    return data
real_data = get_real_data()
n = len(real_data)
np.asarray(real_data).shape
synthetic_data = np.load("logs/data/generated_data.npy")
synthetic_data.shape
real_data = real_data[: synthetic_data.shape[0]]
# Prepare Sample
sample_size = 250
idx = np.random.permutation(len(real_data))[:sample_size]
# Data preprocessing
real_sample = np.asarray(real_data)[idx]
synthetic_sample = np.asarray(synthetic_data)[idx]
# Reshape real and synthetic samples
synthetic_sample_2d = synthetic_sample.reshape(-1, seq_len)
real_sample_2d = real_sample.reshape(-1, seq_len)
real_sample_2d.shape, synthetic_sample_2d.shape
# Visualization in 2D: A Qualitative Assessment of Diversity
## Run PCA
pca = PCA(n_components=2)
pca.fit(real_sample_2d)
pca_real = pd.DataFrame(pca.transform(real_sample_2d)).assign(Data="Real")
pca_synthetic = pd.DataFrame(pca.transform(synthetic_sample_2d)).assign(
    Data="Synthetic"
)
pca_result = pd.concat([pca_real, pca_synthetic]).rename(
    columns={0: "1st Component", 1: "2nd Component"}
)
## Run t-SNE
tsne_data = np.concatenate((real_sample_2d, synthetic_sample_2d), axis=0)

tsne = TSNE(n_components=2, verbose=1, perplexity=40)
tsne_result = tsne.fit_transform(tsne_data)
tsne_result = pd.DataFrame(tsne_result, columns=["X", "Y"]).assign(Data="Real")
tsne_result.loc[sample_size * 6 :, "Data"] = "Synthetic"
## Plot Result
fig, axes = plt.subplots(ncols=2, figsize=(14, 5))

sns.scatterplot(
    x="1st Component",
    y="2nd Component",
    data=pca_result,
    hue="Data",
    style="Data",
    ax=axes[0],
    s=100,  # Increase size of the points
    alpha=0.7,  # Increase transparency of the points
)
sns.despine()
axes[0].set_title("PCA Result")


sns.scatterplot(
    x="X",
    y="Y",
    data=tsne_result,
    hue="Data",
    style="Data",
    ax=axes[1],
    s=100,
    alpha=0.7,
)
sns.despine()
for i in [0, 1]:
    axes[i].set_xticks([])
    axes[i].set_yticks([])

axes[1].set_title("t-SNE Result")
fig.suptitle(
    "Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions",
    fontsize=14,
)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
# plotly tsne and pca plots
import plotly.express as px

pca_result["Data"] = pca_result["Data"].astype("category")
pca_result["Data"] = pca_result["Data"].cat.reorder_categories(["Real", "Synthetic"])

tsne_result["Data"] = tsne_result["Data"].astype("category")
tsne_result["Data"] = tsne_result["Data"].cat.reorder_categories(["Real", "Synthetic"])

fig = px.scatter(
    pca_result,
    x="1st Component",
    y="2nd Component",
    color="Data",
    symbol="Data",
    opacity=0.7,
    title="PCA Result",
)
fig.show()

fig = px.scatter(
    tsne_result,
    x="X",
    y="Y",
    color="Data",
    symbol="Data",
    opacity=0.7,
    title="t-SNE Result",
)
fig.show()
# Time Series Classification: A quantitative Assessment of Fidelity
## Prepare Data
real_data = get_real_data()
real_data = np.array(real_data)[: len(synthetic_data)]
real_data.shape
synthetic_data.shape
n_series = real_data.shape[0]
idx = np.arange(n_series)
n_train = int(0.8 * n_series)
train_idx = idx[:n_train]
test_idx = idx[n_train:]
train_data = np.vstack((real_data[train_idx], synthetic_data[train_idx]))
test_data = np.vstack((real_data[test_idx], synthetic_data[test_idx]))
n_train, n_test = len(train_idx), len(test_idx)
train_labels = np.concatenate((np.ones(n_train), np.zeros(n_train)))
test_labels = np.concatenate((np.ones(n_test), np.zeros(n_test)))
train_data.shape, train_labels.shape, test_data.shape, test_labels.shape
## Create Classifier
ts_classifier = Sequential(
    [
        GRU(27, input_shape=(24, 27), name="GRU"),
        Dense(1, activation="sigmoid", name="OUT"),
    ],
    name="Time_Series_Classifier",
)
ts_classifier.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=[AUC(name="AUC"), "accuracy"]
)
ts_classifier.summary()
result = ts_classifier.fit(
    x=train_data,
    y=train_labels,
    validation_data=(test_data, test_labels),
    epochs=250,
    batch_size=128,
    verbose=1,
)
ts_classifier.evaluate(x=test_data, y=test_labels)
history = pd.DataFrame(result.history)
history.info()
from matplotlib.ticker import FuncFormatter
sns.set_style("white")
fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
history[["AUC", "val_AUC"]].rename(columns={"AUC": "Train", "val_AUC": "Test"}).plot(
    ax=axes[1], title="ROC Area under the Curve", style=["-", "--"], xlim=(0, 250)
)
history[["accuracy", "val_accuracy"]].rename(
    columns={"accuracy": "Train", "val_accuracy": "Test"}
).plot(ax=axes[0], title="Accuracy", style=["-", "--"], xlim=(0, 250))
for i in [0, 1]:
    axes[i].set_xlabel("Epoch")

axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
axes[0].set_ylabel("Accuracy (%)")
axes[1].set_ylabel("AUC")
sns.despine()
fig.suptitle("Assessing Fidelity: Time Series Classification Performance", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.85)
# Train on Synthetic, test on real: Assessing usefulness
real_data = get_real_data()
real_data = np.array(real_data)[: len(synthetic_data)]
real_data.shape, synthetic_data.shape
real_train_data = real_data[train_idx, :23, :]
real_train_label = real_data[train_idx, -1, :]

real_test_data = real_data[test_idx, :23, :]
real_test_label = real_data[test_idx, -1, :]
real_train_data.shape, real_train_label.shape, real_test_data.shape, real_test_label.shape
synthetic_train = synthetic_data[:, :23, :]
synthetic_label = synthetic_data[:, -1, :]
synthetic_train.shape, synthetic_label.shape
def get_model():
    model = Sequential([GRU(12, input_shape=(23, 27)), Dense(27)])

    model.compile(optimizer=Adam(), loss=MeanAbsoluteError(name="MAE"))
    return model
ts_regression = get_model()
synthetic_result = ts_regression.fit(
    x=synthetic_train,
    y=synthetic_label,
    validation_data=(real_test_data, real_test_label),
    epochs=100,
    batch_size=128,
    verbose=1,
)
ts_regression = get_model()
real_result = ts_regression.fit(
    x=real_train_data,
    y=real_train_label,
    validation_data=(real_test_data, real_test_label),
    epochs=100,
    batch_size=128,
    verbose=1,
)
synthetic_result = pd.DataFrame(synthetic_result.history).rename(
    columns={"loss": "Train", "val_loss": "Test"}
)
real_result = pd.DataFrame(real_result.history).rename(
    columns={"loss": "Train", "val_loss": "Test"}
)
fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)
synthetic_result.plot(
    ax=axes[0], title="Train on Synthetic, Test on Real", logy=True, xlim=(0, 100)
)
real_result.plot(
    ax=axes[1], title="Train on Real, Test on Real", logy=True, xlim=(0, 100)
)
for i in [0, 1]:
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("Mean Absolute Error (log scale)")

sns.despine()
fig.suptitle("Assessing Usefulness: Time Series Prediction Performance", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.85)