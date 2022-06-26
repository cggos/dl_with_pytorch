/**
 * @file lenet5_test.cpp
 * @author Gavin Gao (cggos@outlook.com)
 * @brief 一个简单的前馈神经网络(feed-forward network）
 * @ref https://blog.csdn.net/defi_wang/article/details/107589456
 * @version 0.1
 * @date 2022-06-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <torch/torch.h>

struct LeNet5 : torch::nn::Module {
  LeNet5()
      : C1(register_module("C1", torch::nn::Conv2d(1, 6, 5))),
        C3(register_module("C3", torch::nn::Conv2d(6, 16, 5))),
        F5(register_module("F5", torch::nn::Linear(16 * 5 * 5, 120))),
        F6(register_module("F6", torch::nn::Linear(120, 84))),
        OUTPUT(register_module("OUTPUT", torch::nn::Linear(84, 10))) {}

  ~LeNet5() {}

  int64_t num_flat_features(torch::Tensor input) {
    int64_t num_features = 1;
    auto sizes = input.sizes();
    for (auto s : sizes) {
      num_features *= s;
    }
    return num_features;
  }

  torch::Tensor forward(torch::Tensor input) {
    namespace F = torch::nn::functional;
    // 2x2 Max pooling
    auto x = F::max_pool2d(F::relu(C1(input)), F::MaxPool2dFuncOptions({2, 2}));
    // 如果是方阵,则可以只使用一个数字进行定义
    x = F::max_pool2d(F::relu(C3(x)), F::MaxPool2dFuncOptions(2));
    x = x.view({-1, num_flat_features(x)});
    x = F::relu(F5(x));
    x = F::relu(F6(x));
    x = OUTPUT(x);
    return x;
  }

  torch::nn::Conv2d C1;
  torch::nn::Conv2d C3;
  torch::nn::Linear F5;
  torch::nn::Linear F6;
  torch::nn::Linear OUTPUT;
};

int main() {
  LeNet5 net1;
  std::cout << "LeNet-5 layout:\n" << net1 << '\n';

  // 打印各层参数，比如权重，和偏置
  for (auto const& p : net1.named_parameters()) std::cout << p.key() << ":\n\t" << p.value().sizes() << '\n';

  // 产生随机输入，并在该网络进行处理
  auto input = torch::randn({1, 1, 32, 32});
  auto out = net1.forward(input);
  std::cout << "out: " << out << '\n';

  // 清零所有参数的梯度缓存，然后进行随机梯度的反向传播
  net1.zero_grad();
  out.backward(torch::randn({1, 10}));

  // 一个损失函数接受一对(output, target)作为输入，计算一个值来估计网络的输出和目标值相差多少
  auto output = net1.forward(input);
  auto target = torch::randn({10});
  target = target.view({1, -1});
  auto criterion = torch::nn::MSELoss();

  auto loss = criterion(output, target);
  std::cout << "loss: " << loss << '\n';

  return 0;
}
