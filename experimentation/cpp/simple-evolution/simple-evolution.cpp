// INCOMPLETE: IN PROGRESS

#include <iostream>
#include <torch/torch.h>

struct NeuralNetwork : torch::nn::Module {
    NeuralNetwork()
        : linear1(register_module("linear1", torch::nn::Linear(2, 4))),
          relu(register_module("relu", torch::nn::ReLU())),
          linear2(register_module("linear2", torch::nn::Linear(4, 1))),
          sigmoid(register_module("sigmoid", torch::nn::Sigmoid())) {}
    
    torch::Tensor forward(torch::Tensor x) {
        x = linear1->forward(x);
        x = relu->forward(x);
        x = linear2->forward(x);
        x = sigmoid->forward(x);
        return x;
    }

    torch::nn::Linear linear1;
    torch::nn::ReLU relu;
    torch::nn::Linear linear2;
    torch::nn::Sigmoid sigmoid;
};

int main() {
    auto model = std::make_shared<NeuralNetwork>();
    int ITERATIONS = 100;
    float x_train[][] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    float y_train[][] = {
        {0},
        {1},
        {1},
        {0}
    };
    torch::Tensor x_train_tensor = torch::from_blob(x_train, {4, 2});
    torch::Tensor y_train_tensor = torch::from_blob(y_train, {4, 1});
    std::cout << "Hello, world!" << std::endl;
    return 0;
}