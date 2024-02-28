#include <vector>
#include <torch/extension.h>

// torch::Tensor partial_scan_forward(
// 	torch::Tensor &Ax,
// 	torch::Tensor &Bx,
// 	torch::Tensor &Bh,
// 	torch::Tensor &Wq,
// 	torch::Tensor &Wk
// );
at::Tensor partial_scan_forward(
	at::Tensor &Ax,
	at::Tensor &Bx,
	at::Tensor &Bh,
	at::Tensor &Wq,
	at::Tensor &Wk
);
// at::Tensor partial_scan_forward(const at::Tensor &Ax, const at::Tensor &Bx);
at::Tensor partial_scan_backward(const at::Tensor &Ax, const at::Tensor &Bx);
