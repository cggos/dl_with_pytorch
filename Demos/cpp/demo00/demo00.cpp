#include <torch/torch.h>

#include <iostream>

int main() {
  std::cout << "cuda is available: " << torch::cuda::is_available() << std::endl;

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}