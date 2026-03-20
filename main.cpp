#include "matrix.h"
#include "neuralnetwork.h"
#include <iostream>
#include <fstream>
#include <chrono>
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

using namespace nn;

int main() {

    //Training data
    mnist_data* data;
    unsigned int count;
    mnist_load("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &data, &count);

    //Test data
    mnist_data* test_data;
    unsigned int test_count;
    mnist_load("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", &test_data, &test_count);

    int correct = 0; 

    auto model = make_model<float>(784, 10, 128, 2, 0.01f);

    int epochs = 50;

    int batch_size = 32;

    Matrix<float> x(784, batch_size);
    Matrix<float> y(10, batch_size);

    std::ofstream log_file("training_log.csv");
    log_file << "epoch,loss\n";

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Training Images...\n";

    for(int e = 0; e < epochs; e++) {

        model.lr = 0.1f / (e + 1);       // variable lr -- lr reduces as it converges
        // model.lr = 0.01f*std::pow(0.85f, e);
        float total_loss = 0;


        for(unsigned int i = 0; i < count; i+=batch_size) {
            
            
            // guard
            if(i + batch_size > count) break;

            std::fill(x.data.begin(), x.data.end(), 0.0f);
            std::fill(y.data.begin(), y.data.end(), 0.0f);

            for(int b = 0; b < batch_size; b++) {
                for(int row = 0; row < 28; row++)
                    for(int col = 0; col < 28; col++)
                        x(row * 28 + col, b) = data[i + b].data[row][col];
                y(data[i + b].label, b) = 1.0f;
            }

            auto y_hat = model.forward(x);
            model.backprop(y);

            // average loss over batch
            for(int b = 0; b < batch_size; b++) {
                Matrix<float> y_hat_col(10, 1);
                Matrix<float> y_col(10, 1);
                for(int j = 0; j < 10; j++) {
                    y_hat_col(j, 0) = y_hat(j, b);
                    y_col(j, 0) = y(j, b);
                }
                total_loss += model.cross_entropy(y_hat_col, y_col);
            }
        }

        float avg_loss = total_loss / count;

        log_file << e << "," << avg_loss << "\n";

        if(e % 1 == 0) {
            std::cout << "Epoch " << e << " Loss: " << avg_loss << "\n";
        }

    }

    log_file.close();

    std::ofstream pred_file("predictions.csv");
    pred_file << "index,true_label,predicted\n";

    std::cout << "\nTesting:\n";


    for(int i = 0; i < test_count; i++) {

        for(int row = 0; row < 28; row++) {
            for(int col = 0; col < 28; col++) {
                x(row * 28 + col, 0) = test_data[i].data[row][col];                
            }
        }   
        auto pred = model.forward(x);

        int predicted = 0;
        for(int j = 1; j < 10; j++) {
            if(pred(j,0) > pred(predicted, 0)) {
                predicted = j;
            }
        }
        if(predicted == (int)test_data[i].label) correct++;
        pred_file << i << "," << test_data[i].label << "," << predicted << "\n";        
    }
    
    pred_file.close();

    std::cout << "Test Accuracy: " << (correct * 100.0f / test_count) << "%\n";
    
    //timer 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Training time: " << elapsed.count() << " seconds\n";

    return 0;
}