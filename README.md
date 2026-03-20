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
4. [benchmarks](#benchmarks)
5. [results](#results)
6. [build & run](#build--run)
7. [file structure](#file-structure)
8. [what i learned](#what-i-learned)
9. [future improvements](#future-improvements)
10. [disclosure](#disclosure)

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
| learning rate | 0.1 / (epoch + 1) |
| weight init | Xavier |
| epochs | 50 |
| batch size | 32 |
| training samples | 60,000 |
| test samples | 10,000 |

---

## optimizations

| technique | what it does | impact |
|---|---|---|
| IKJ loop order | sequential memory access in matmul inner loop | ~4x faster than naive ijk |
| AVX2 + FMA | 8 floats per instruction via 256-bit SIMD | meaningful speedup on matmul |
| cache tiling | 32×32 blocks to fit in L1 cache | no measurable impact at these matrix sizes |
| Xavier init | `stddev = sqrt(1/fan_in)` — prevents vanishing gradients at init | better convergence |
| LR decay | reduces learning rate each epoch | smoother loss curve |
| heap optimization | x and y matrices allocated once outside training loop | eliminates 180k heap allocs |

**compile command:**
```bash
g++ main.cpp -O2 -march=native -mavx2 -mfma -o nn
```

> Note: OpenMP (`-fopenmp`) actually slows things down on Windows/MinGW for these matrix sizes — thread spawning overhead exceeds the gains. On Linux it helps slightly.

---

## benchmarks

| configuration | time | accuracy |
|---|---|---|
| single-sample SGD, 3 epochs | ~330s | 97.34% |
| + AVX2 SIMD | ~320s | 97.34% |
| mini-batching (batch=32), 3 epochs | ~85s | 97.41% |
| mini-batching, 50 epochs | ~393s | 97.80% |

*tested on Ryzen 7 7735HS, Windows, MinGW*

**key finding:** mini-batching gave a 13x speedup over single-sample SGD for the same 3 epochs — from 330s to 85s — while maintaining equivalent accuracy. further gains came from simply running more epochs within the same time budget rather than architectural changes.

---

## results

```
Test Accuracy:  97.80%   (10,000 unseen images)
Training Time:  ~393s    (Ryzen 7 7735HS, Windows)
Epochs:         50
Final Loss:     0.0301
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
pip install matplotlib pandas numpy
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
- the architecture is the real ceiling. no amount of micro-optimization fixes a fundamentally limited model

---

> Note: Uses the MNIST loader from:
https://github.com/projectgalateia/mnist
Author: Nuri Park
License: MIT

---

## future improvements(maybe)

- [x] mini-batch SGD
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