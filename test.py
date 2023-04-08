
import numpy as np
kernel_size = 5
num_kernels = 15
h, w = 28, 28
output = np.zeros(shape=int(h-(kernel_size-1)),
                  int(w-(kernel_size-1)),
                  )
print(output)
print(output.shape)
