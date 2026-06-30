
#include <torch/extension.h>
#include "dimwise/dimwise.h"
#include "full_scan/full_scan.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dimwise_forward", &dimwise_fwd, "dimwise scan forward (CUDA)");
    m.def("fullscan_forward", &warpscan_fwd, "full scan forward (CUDA)");
}
