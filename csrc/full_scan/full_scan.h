
#include <torch/extension.h>

#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_HALF_OR_BFLOAT_OR_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16 || x.dtype() == torch::kFloat32, #x " must be float16 or bfloat16 or float32")


#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_IS_HALF_OR_BFLOAT_OR_FLOAT(x)

at::Tensor warpscan_forward(
    const at::Tensor &Ax,
    const at::Tensor &Bx,
    const at::Tensor &out,
    const bool reverse
);



at::Tensor warpscan_fwd(
    const at::Tensor &Ax,
    const at::Tensor &Bx,
    const at::Tensor &out,
    const bool reverse)
{
    CHECK_INPUT(Ax);
    CHECK_INPUT(Bx);

    return warpscan_forward(Ax, Bx, out, reverse);
}

