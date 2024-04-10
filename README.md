TimeGAN for Mental Health Monitoring Using CrossCheck Data
==========================================================

Overview
--------

This repository presents an IPython Notebook detailing the application of Time-series Generative Adversarial Networks (TimeGAN) to the CrossCheck dataset, aiming to generate synthetic time-series data for monitoring severe mental illness symptoms. This project is part of an extensive study to assess the effectiveness and fidelity of synthetic data in the context of mental health research, preserving patient privacy while providing rich data for analysis.

Contents
--------

*   **TimeGAN\_CrossCheck\_Analysis.ipynb**: An IPython Notebook that encapsulates the entire process from data preprocessing, model training, synthetic data generation, to evaluation of synthetic data.
*   **TimeGAN\_CrossCheck\_Evaluation.ipynb**: An IPython Notebook that presents the process for the further evaluation of the synthetic data.
*   **Requirements.txt**: Specifies the Python packages required to run the notebook.

Notebook Features
-----------------

*   **Data Preprocessing**: Instructions on how to prepare the CrossCheck dataset for TimeGAN.
*   **Model Overview**: An introduction to the TimeGAN architecture adapted to generate synthetic time-series data.
*   **Model Training**: Detailed steps for training the TimeGAN model on the preprocessed data.
*   **Synthetic Data Generation**: Procedures to generate synthetic data mimicking the original dataset's characteristics.
*   **Evaluation**: Metrics and methods to evaluate the quality and utility of the generated synthetic data against the real dataset.
*   **Visualisation**: Visual aids to compare real and synthetic data distributions, including PCA and t-SNE analyses.

Getting Started
---------------

To explore the notebook:

1.  Ensure Python 3.6+ is installed on your system.
2.  Clone this repository to your local machine.
3.  Install the required packages using `pip install -r requirements.txt`.
4.  Open the `TimeGAN_CrossCheck_Analysis.ipynb` notebook in Jupyter Lab or Notebook.

Data Privacy
------------

The CrossCheck dataset used in this study contains sensitive information and is not included in this repository. The methods and analyses are documented for academic purposes, assuming access to the dataset under appropriate ethical guidelines.

Citation & Acknowledgments
--------------------------

If you find this work useful, please consider citing the original TimeGAN paper and any other relevant literature:

`@inproceedings{yoon2019time,   title={Time-series generative adversarial networks},   author={Yoon, Jinsung and Jarrett, Daniel and van der Schaar, Mihaela},   booktitle={Advances in neural information processing systems},   year={2019} }`

Special thanks to Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar for their foundational work on TimeGAN, and to the providers of the CrossCheck dataset for supporting mental health research.


