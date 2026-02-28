#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <cassert>
#include <random>
#include <functional>
#include <omp.h>         //--using openmp
#include <immintrin.h>   //--Avx2(simd)

namespace linalg {
    template<typename Type>
    class Matrix {
        size_t rows;
        size_t cols;

        public:
            std::vector<Type> data;
            std::pair<size_t, size_t> shape;

            Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data({}) {
                data.resize(rows*cols, Type());
                shape = {rows, cols};
            }
            
            Matrix() : rows(0), cols(0), data({}) {
                shape = {rows, cols};
            };

            size_t numel() const {
                return rows*cols;
            }

            void print_shape() {
                std::cout << "Matrix Size([" << rows << ", " << cols << "])\n";
            }
            void print() {
                for(size_t r=0; r<rows; r++) {
                    for(size_t c=0; c<cols; c++) {
                        std::cout << (*this)(r,c) << " ";
                    }
                    std::cout <<std::endl;
                }
                std::cout <<std::endl;
            }

            //flatening
            Type& operator()(size_t row, size_t col) {
                assert(row<rows && col<cols);
                return data[row*cols+col];
            }

            //Matrix Multiplication(kill me) -- added a simple but effective technque istead of ijk i do ikj due to some memory/cache reasons it works faster(credit som random guy on reddit)
            Matrix matmul(Matrix &target) {
                assert(cols == target.rows);
                Matrix output(rows, target.cols);
                
                #pragma omp parallel for
                for(size_t r=0; r<output.rows; r++) {
                    for(size_t k=0; k<target.rows; k++) {
                        __m256 a_val = _mm256_set1_ps((*this)(r, k));           //--[a,a,a,a,a,a,a,a]
                        size_t c = 0;
                        for(; c+8<=target.cols; c+=8) {
                            __m256 b_vec = _mm256_loadu_ps(&target.data[k * target.cols + c]);   //--load 8 consecutive floats from row k

                            __m256 out_vec = _mm256_loadu_ps(&output.data[r * output.cols + c]);  //-load 8 consecutive output

                            out_vec = _mm256_fmadd_ps(a_val, b_vec, out_vec);

                            _mm256_storeu_ps(&output.data[r * output.cols + c], out_vec);
                        }
                        //leftover columns
                        for(; c < target.cols; c++) {
                            output.data[r * output.cols + c] += (*this)(r,k) * target.data[k * target.cols + c];
                        }
                    }
                }
                return output;
            }

            //Element wise multiplication
            Matrix multiply_elementwise(Matrix &target) {
                assert(shape == target.shape);
                Matrix output((*this));

                for(size_t r=0; r<output.rows; r++) {
                    for(size_t c=0; c<output.cols; c++) {
                        output(r, c) = (*this)(r, c)*target(r, c);
                    }
                }
                return output;
            }

            //Matrix squaring
            Matrix square() { 
                Matrix output((*this));
                output = multiply_elementwise(output);
                return output;
            }

            //scalar multiplication (tis is love)
            Matrix multiply_scalar(Type scalar) {
                Matrix output((*this));
                for (size_t r = 0; r < output.rows; ++r) {
                    for (size_t c = 0; c < output.cols; ++c) {
                        output(r, c) = scalar * (*this)(r, c);
                    }
                }
                return output;
            }

            //addition
            Matrix add(Matrix &target) {
                assert(shape == target.shape);
                Matrix output(rows, target.cols);

                for (size_t r=0; r<output.rows; r++) {
                    for (size_t c=0; c<output.cols; c++) {
                        output(r, c) = (*this)(r, c) + target(r, c);
                    }
                }
                return output;
            }
            Matrix operator+(Matrix &target) {
                return add(target);
            }

            //matrix subtraction
            Matrix subtract(Matrix &target) {
                assert(shape == target.shape);
                Matrix output(rows, target.cols);

                for (size_t r=0; r<output.rows; r++) {
                    for (size_t c=0; c<output.cols; c++) {
                        output(r, c) = (*this)(r, c) - target(r, c);
                    }
                }
                return output;
            }
            Matrix operator-(Matrix &target) {
                return subtract(target);
            }

            //Matrix Transpose
            Matrix transpose() {
                size_t new_rows {cols};
                size_t new_cols {rows};
                Matrix transposed(new_rows, new_cols);

                for(size_t r=0; r<new_rows; r++) {
                    for(size_t c=0; c<new_cols; c++) {
                        transposed(r, c) = (*this)(c, r);
                    }
                }
                return transposed;
            }
            Matrix T() {
                return transpose();
            }
            
            //Actualy appllying non linear fxn ReLu etc
            Matrix apply_function(const std::function<Type(const Type &)> &function) {
                Matrix output((*this));
                for(size_t r=0; r<rows; r++) {
                    for(size_t c=0; c<cols; c++) {
                        output(r, c) = function((*this)(r, c));
                    }
                }
                return output;
            }
            

    };

    //Final stats application
    template<typename T>
    struct matrix {
        private:
            static std::mt19937& get_generator() {
                static std::random_device rd{};
                static std::mt19937 gen{rd()};
                return gen;
            }

        public:
            static Matrix<T> randn(size_t rows, size_t cols) {
                Matrix<T> M(rows, cols);
                T n = M.numel();
                T stddev = 1/std::sqrt(M.shape.second);        //----instead of n(total ele) i directly use the input cols
                // T stddev = std::sqrt(2.0 / M.shape.second);  // went for He initialization for ReLU instead of xavier  --revoking this cuz of fking dying relu
                std::normal_distribution<T> d{0, stddev};
        
                auto& gen = get_generator();
        
                for(size_t r=0; r<rows; r++) {
                    for(size_t c=0; c<cols; c++) {
                        M(r, c) = d(gen);
                    }
                }
                return M;
            }

            //null matrix
            static Matrix<T> null(size_t rows, size_t cols) {
                return Matrix<T>(rows, cols);
            }

            //identity matrix
            static Matrix<T> identity(size_t n) {
                Matrix<T> I(n, n);
                
                for(size_t i=0; i<n; i++) {
                    I(i, i) = T(1);
                }
                return I;
            }

            //ones matrix
            static Matrix<T> ones(size_t rows, size_t cols) {
                Matrix<T> One(rows, cols);

                for(size_t r=0; r<rows; r++) {
                    for(size_t c=0; c<cols; c++) {
                        One(r, c) = T(1);
                    }
                }
                return One;
            }
    };
}