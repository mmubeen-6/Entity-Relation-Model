# Entity Relation Extraction

## Introduction

This is a simple repo to create an entity relation model and train it to predict the relation between two entities in a sentence. The model is trained on the [kbp37_formatted](https://huggingface.co/datasets/DFKI-SLT/kbp37) dataset.

An entity relation model is a model that takes two entities and a sentence as input and predicts the relation between the two entities in the sentence.

For example, given the sentence "The company Apple was founded by Steve Jobs", the model should predict that the relation between "Apple" and "Steve Jobs" is "founders".

This repo uses a simple BERT model to predict the relation between two entities in a sentence.

## How to run

This is trained and tested using on Google Colab. To run clik on the following button:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uQ-5fiGbabOb8UveWciz5yoQmwmTDBya?usp=sharing)

Step by step instructions are provided in the notebook including how to setup the environment and download the dataset.
