import numpy as np
import pandas as pd

def conv(x, kernel, pad, stride):
    batch, H, W, c_in = x.shape
    K_h, K_w, _, c_out = kernel.shape
    H_out = 1 + (H + 2 * pad - K_h) // stride
    W_out = 1 + (W + 2 * pad - K_w) // stride
    # pad_x = np.pad(x, ((pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((batch, H_out, W_out, c_out))
    for b in range(batch):
      for i in range(H_out):
          for j in range(W_out):
              region = tensor[b, i * stride:i * stride + K_h, j * stride:j * stride + K_w, :]
              out[b, i, j, :] = np.tensordot(region, kernel, axes=([0, 1, 2], [0, 1, 2])) + bias
    return out

tensor = np.load('tensor.npy')
kernel = np.load('kernel.npy')
bias = np.load('bias.npy')
stride = pd.read_csv('task.csv')['stride'].iloc[0]
print(tensor.shape)
print(kernel.shape)
print(bias.shape)
print(stride)

output = conv(tensor, kernel, 0, stride)

np.save('seminar03_conv.npy', output, allow_pickle=False)