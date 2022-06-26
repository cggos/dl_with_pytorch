#include <torch/torch.h>

#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "cuda is available: " << torch::cuda::is_available() << std::endl;

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  return 0;
}