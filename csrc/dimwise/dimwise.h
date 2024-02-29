
#include <torch/extension.h>

#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_HALF_OR_BFLOAT_OR_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16 || x.dtype() == torch::kFloat32, #x " must be float16 or bfloat16 or float32")


#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_IS_HALF_OR_BFLOAT_OR_FLOAT(x)

void dimwise_pscan(
    torch::Tensor &Ax,
    torch::Tensor &Bx,
    torch::Tensor &Bh,
    torch::Tensor &Wq,
    torch::Tensor &Wk);



void dimwise_fwd(
    torch::Tensor &Ax,
    torch::Tensor &Bx,
    torch::Tensor &Bh,
    torch::Tensor &Wq,
    torch::Tensor &Wk)
{
    CHECK_INPUT(Ax);
    CHECK_INPUT(Bx);
    CHECK_INPUT(Bh);
    CHECK_INPUT(Wq);
    CHECK_INPUT(Wk);

    dimwise_pscan(Ax, Bx, Bh, Wq, Wk);
}

