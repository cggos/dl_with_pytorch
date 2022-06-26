#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "cuda is available: " << torch::cuda::is_available() << std::endl;

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module =
      std::make_shared<torch::jit::script::Module>(torch::jit::load(argv[1]));  // .pt file path

  assert(module != nullptr);

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module->forward(inputs).toTensor();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}