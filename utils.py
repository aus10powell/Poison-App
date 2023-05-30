"""
This module contains custom metric functions for binary classification models using TensorFlow and Keras.

Functions:
- f1(y_true, y_pred): Computes the F1 score metric for binary classification models.
- recall(y_true, y_pred): Computes the recall metric for binary classification models.
- precision(y_true, y_pred): Computes the precision metric for binary classification models.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())

    return f1_score


def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    recall = tp / (tp + fn + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    return precision
