#pragma once
#include "matrix.h"
#include <random>
#include <cmath>
#include <cassert>
#include <utility>
#include <algorithm>
using namespace linalg;

namespace nn {
    template<typename T>
    class MLP {
        public:
            std::vector<size_t> units_per_layer;
            std::vector<Matrix<T>> weight_matrices;
            std::vector<Matrix<T>> bias_vectors;
            std::vector<Matrix<T>> activations;
            float lr;
            
            //constructor
            explicit MLP(std::vector<size_t> units_per_layer, float lr = .001f) :
                units_per_layer(units_per_layer),
                weight_matrices(),
                bias_vectors(),
                activations(this->units_per_layer.size()),
                lr(lr) {
                    for(size_t i=0; i<this->units_per_layer.size()-1; i++) {
                        size_t in_channels = this->units_per_layer[i];
                        size_t out_channels = this->units_per_layer[i+1];

                        auto W = linalg::matrix<T>::randn(out_channels, in_channels);
                        weight_matrices.push_back(W);

                        auto b = linalg::matrix<T>::randn(out_channels, 1);
                        bias_vectors.push_back(b);
                    }
                }

            //forward pass
            inline T tanh_act(T x) const {
                return std::tanh(x);
            }
            // inline T relu(T x) const {
            //     return x > T(0) ? x : T(0);
            // }

            //softmax
            Matrix<T> softmax(Matrix<T> x) {
                Matrix<T> output(x.shape.first, x.shape.second);

                for(size_t c=0; c<x.shape.second; c++) {
                    T max_val = T(-1e9);
                    for(size_t i=0; i<x.shape.first; i++)
                        max_val = std::max(max_val, x(i, c));
                    
                    T sum = T(0);
                    for(size_t i=0; i<x.shape.first; i++) {
                        output(i, c) = std::exp(x(i, c)-max_val);
                        sum += output(i, c);
                    }

                    for(size_t i=0; i<x.shape.first; i++) {
                        output(i, c) /= sum;                        
                    }
                }
                return output;
            }

            
            //cross entropy
            T cross_entropy(Matrix<T>& y_hat, Matrix<T>& y_true) {
                T loss = T(0);

                for(size_t i=0; i<y_hat.shape.first; i++) {
                    T p = std::max(y_hat(i, 0), T(1e-7));
                    loss -= y_true(i, 0)*std::log(p);
                }
                return loss;
            }

            auto forward(Matrix<T> x) {
                assert(x.shape.first == units_per_layer[0]);

                activations[0] = x;
                Matrix<T> prev = x;
                for(size_t i=0; i<units_per_layer.size()-1; i++) {

                    Matrix<T> y = weight_matrices[i].matmul(prev);
                    for(size_t col = 0; col < y.shape.second; col++)
                        for(size_t row = 0; row < y.shape.first; row++)
                            y(row, col) += bias_vectors[i](row, 0);

                    if(i < units_per_layer.size()-2) {
                        y = y.apply_function([this](T v) { return tanh_act(v); });
                        // y = y.apply_function([this](T v) { return relu(v); });
                    }  else {
                        y = softmax(y);
                    }
                    activations[i+1] = y;
                    prev = y;
                }
                return prev;
            }

            //bacward pass
            inline T d_tanh_from_output(T y) const {
                // derivative using output of tanh
                return T(1) - y * y;
            }
            // inline T d_relu(T y) const {
            //     return y > T(0) ? T(1) : T(0);
            // }

            void backprop(Matrix<T> target) {
                assert(target.shape.first == units_per_layer.back());

                auto y_hat = activations.back();
                auto error = (y_hat - target); // MSE derivative
                size_t batch_size = target.shape.second;

                for(int i = weight_matrices.size() - 1; i >= 0; i--) {

                    Matrix<T> gradients = error;

                    if(i < weight_matrices.size() - 1) {
                        auto d_outputs = activations[i+1]
                            .apply_function([this](T v) { return d_tanh_from_output(v); });
                        // auto d_outputs = activations[i+1]
                        //     .apply_function([this](T v) { return d_relu(v); });

                        gradients = gradients.multiply_elementwise(d_outputs);
                    }

                    auto a_trans = activations[i].T();
                    auto weight_gradients = gradients.matmul(a_trans);
                    
                    weight_gradients = weight_gradients.multiply_scalar(1.0f/batch_size);
                    
                    Matrix<T> bias_grad(gradients.shape.first, 1);
                    for(size_t r=0; r<gradients.shape.first; r++) {
                        T sum = T(0);
                        for(size_t c=0; c<batch_size; c++)
                            sum += gradients(r, c);
                        bias_grad(r, 0) = sum/batch_size;
                    }

                    auto Wt = weight_matrices[i].T();

                    auto bias_update = bias_grad.multiply_scalar(lr);
                    auto weight_update = weight_gradients.multiply_scalar(lr);
                    bias_vectors[i] = bias_vectors[i].subtract(bias_update);
                    weight_matrices[i] = weight_matrices[i].subtract(weight_update);

                    error = Wt.matmul(error);
                }
            }
    };

    template<typename T>
    MLP<T> make_model(size_t in_channels, size_t out_channels, size_t hidden_units_per_layer, int hidden_layers, float lr) {
                
                std::vector<size_t> units_per_layer;

                units_per_layer.push_back(in_channels);

                for (int i = 0; i < hidden_layers; ++i)
                    units_per_layer.push_back(hidden_units_per_layer);

                units_per_layer.push_back(out_channels);

                return MLP<T>(units_per_layer, lr);
            }
}