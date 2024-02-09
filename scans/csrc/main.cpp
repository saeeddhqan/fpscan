#include <vector>

// std::vector<torch::Tensor> myscan_forward(at::Tensor &Ax, at::Tensor &Bx, const unsigned int GT);
// at::Tensor myscan_forward(const at::Tensor &Ax, const at::Tensor &Bx);
at::Tensor partial_scan_forward(const at::Tensor &Ax, const at::Tensor &Bx);
at::Tensor partial_scan_backward(const at::Tensor &Ax, const at::Tensor &Bx);
