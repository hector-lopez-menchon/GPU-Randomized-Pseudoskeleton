#!/bin/bash

nvcc --compiler-options '-fPIC' -o functionsExternal_cuda.so --shared user_impedance_for_julia_cuda.cu -lm -lcublas -lcusolver


