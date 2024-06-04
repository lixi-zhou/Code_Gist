import numpy as np


def quant_layer(layer):
    # quant
    weight = layer.get_weights()
    quant_stat = []
    for idx in range(len(weight)):
        w = weight[idx]
        min_val = np.min(w)
        max_val = np.max(w)
        step = (max_val - min_val) / 256
        quant_stat.append((min_val, max_val))
        weight[idx] = np.floor((w-min_val) / step)
    layer.set_weights(weight)
    return quant_stat

def dequant_layer(layer, quant_stat):
    # dequant
    weight = layer.get_weights()
    for idx in range(len(weight)):
        min_val, max_val = quant_stat[idx]
        step = (max_val - min_val) / 256
        weight[idx] = (weight[idx]*step) + min_val
    layer.set_weights(weight)

def quant_model(model):
    quant_data = []
    for layer in model.layers:
        quant_stat = quant_layer(layer)
        quant_data.append(quant_stat)
    return model, quant_data

def dequant_model(model, quant_data):
    for idx in range(len(model.layers)):
        layer = model.layers[idx]
        quant_stat = quant_data[idx]
        dequant_layer(layer, quant_stat)
    return model