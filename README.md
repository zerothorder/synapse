# synapse

![banner](media/nn_results.png)

[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-purple.svg)]()
[![AVX2](https://img.shields.io/badge/SIMD-AVX2-orange.svg)]()

**synapse** is a multi-layer perceptron built entirely in C++ no ML libraries, no BLAS.  
Raw matrix math, backprop from scratch, and SGD with learning rate decay.

---

## index

1. [overview](#overview)
2. [architecture](#architecture)
3. [optimizations](#optimizations)
4. [results](#results)
5. [build & run](#build--run)
6. [file structure](#file-structure)
7. [what i learned](#what-i-learned)
8. [future improvements](#future-improvements)
9. [disclosure](#disclosure)

---

## overview

started as an attempt to understand how neural networks actually work under the hood.  
built everything from scratch: matrix operations, forward pass, backpropagation, softmax, cross-entropy.  
then optimized it until it couldn't go faster without a full architectural rewrite.

---

## architecture

```
Input (784) → Dense (128) → tanh → Dense (128) → tanh → Dense (10) → Softmax
```

| hyperparameter | value |
|---|---|
| loss | cross-entropy |
| optimizer | SGD |
| learning rate | 0.01 / (epoch + 1) |
| weight init | Xavier |
| epochs | 3 |
| training samples | 60,000 |

---

## optimizations

| technique | what it does | impact |
|---|---|---|
| IKJ loop order | sequential memory access in matmul inner loop | ~4x faster than naive ijk |
| AVX2 + FMA | 8 floats per instruction via 256-bit SIMD | meaningful speedup on matmul |
| cache tiling | 32×32 blocks to maximize L1/L2 cache reuse | helps on larger matrices |
| Xavier init | `stddev = sqrt(1/fan_in)` — prevents vanishing gradients at init | better convergence |
| LR decay | reduces learning rate each epoch | smoother loss curve |
| heap optimization | x and y matrices allocated once outside training loop | eliminates 180k heap allocs |

**compile command:**
```bash
g++ main.cpp -O2 -march=native -mavx2 -mfma -o nn
```

> Note: OpenMP (`-fopenmp`) actually slows things down on Windows/MinGW for these matrix sizes — thread spawning overhead exceeds the gains. On Linux it helps slightly.

---

## results

```
Test Accuracy:  97.59%   (10,000 unseen images)
Training Time:  ~320s    (Ryzen 7 7735HS, Windows)
Epochs:         3
Final Loss:     0.055
```

---

## build & run

**requirements:** g++ with AVX2 support (any modern x86 CPU from ~2013+)

```bash
# clone
git clone https://github.com/zerothorder/synapse
cd synapse

# download MNIST and place files in mnist/ folder
# https://yann.lecun.com/exdb/mnist/

# compile
g++ main.cpp -O2 -march=native -mavx2 -mfma -o nn

# train + evaluate
./nn
```

**visualize training results:**
```bash
pip install matplotlib pandas numpy seaborn
python visualize.py
```

---

## file structure

```
.
├── main.cpp              — training loop, MNIST loading, evaluation
├── matrix.h              — templated matrix library
├── neuralnetwork.h       — MLP: forward, softmax, backprop, cross-entropy
├── mnist.h               — MNIST binary loader (MIT, Nuri Park)
├── visualization.py      — loss curve, confusion matrix, per-digit accuracy
├── media/
│   └── nn_results.png    — training dashboard
├── mnist/                — dataset files (not included, download separately)
└── README.md
```

---

## what i learned

- cache locality matters more than parallelism at small matrix sizes
- AVX2 intrinsics are approachable once you understand memory layout
- ReLU needs babysitting (dying neurons). tanh just works for shallow nets
- OpenMP thread overhead can exceed gains on small matrices

---

> Note: Uses the MNIST loader from:
https://github.com/projectgalateia/mnist
Author: Nuri Park
License: MIT

---

## future improvements(maybe)

- [ ] mini-batch SGD
- [ ] momentum / Adam optimizer
- [ ] model save / load
- [ ] deeper / wider network

---

## disclosure

this was a learning project.
built this to understand neural networks at the lowest level.  
matrix library, forward pass, backpropagation, softmax, and all optimizations written by me. 

---

## license

MIT