import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Initialize PyCUDA and manually create a context
drv.init()
device = drv.Device(0)  # Select the first GPU
context = device.make_context()

try:
    print("Device context initialized")

    # Define a simple CUDA kernel for testing
    mod = SourceModule("""
    __global__ void multiply_by_two(float *a, int size) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < size) {
            a[idx] *= 2;
        }
    }
    """)

    multiply_by_two = mod.get_function("multiply_by_two")

    # Create large random data (e.g., 10 million elements) to increase GPU load
    size = 10000000
    a = np.random.randn(size).astype(np.float32)
    a_gpu = drv.mem_alloc(a.nbytes)
    drv.memcpy_htod(a_gpu, a)

    # Launch the kernel with a large grid and block size
    block_size = 256
    grid_size = (size + block_size - 1) // block_size

    # Force heavy GPU computation by running the kernel repeatedly
    for _ in range(1000):
        multiply_by_two(a_gpu, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Retrieve the result from the GPU
    a_doubled = np.empty_like(a)
    drv.memcpy_dtoh(a_doubled, a_gpu)

    print("Original array:", a[:10])
    print("Doubled array:", a_doubled[:10])

finally:
    # Ensure the context is popped and released after execution
    context.pop()
    context.detach()
