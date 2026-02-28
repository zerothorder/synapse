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

    int epochs = 3;

    std::ofstream log_file("training_log.csv");
    log_file << "epoch,loss\n";

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Training Images...\n";

    for(int e = 0; e < epochs; e++) {

        model.lr = 0.01f / (e + 1);       // variable lr -- lr reduces as it converges
        // model.lr = 0.01f*std::pow(0.85f, e);
        float total_loss = 0;

        Matrix<float> x(784, 1);
        Matrix<float> y(10,1);

        for(unsigned int i = 0; i < count; i++) {
            
            
            for(int row = 0; row < 28; row++) {
                for(int col = 0; col < 28; col++) {
                    x(row * 28 + col, 0) = data[i].data[row][col];
                }
            }
            
            //reset val
            std::fill(y.data.begin(), y.data.end(), 0.0f);

            y(data[i].label, 0) = 1.0f;

            auto y_hat = model.forward(x);
            model.backprop(y);

            //cross-entropy
            float loss = model.cross_entropy(y_hat, y);
            total_loss += loss;
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

    Matrix<float> x(784, 1);
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