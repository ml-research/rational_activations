
#include <torch/extension.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)




    
	at::Tensor pau_cuda_forward_A_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor pau_forward_A_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_A_5_4(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_A_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_A_5_4(grad_output, x, n, d);
    }
    

    
	at::Tensor pau_cuda_forward_B_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor pau_forward_B_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_B_5_4(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_B_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_B_5_4(grad_output, x, n, d);
    }
    

    
	at::Tensor pau_cuda_forward_C_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor pau_forward_C_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_C_5_4(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_C_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_C_5_4(grad_output, x, n, d);
    }
    
    
	at::Tensor pau_cuda_forward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor pau_forward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_D_5_4(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_D_5_4(training, iteration, grad_output, x, n, d);
    }
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    
    m.def("forward_A_5_4", &pau_forward_A_5_4, "PAU forward A_5_4");
    m.def("backward_A_5_4", &pau_backward_A_5_4, "PAU backward A_5_4");
    
    m.def("forward_B_5_4", &pau_forward_B_5_4, "PAU forward B_5_4");
    m.def("backward_B_5_4", &pau_backward_B_5_4, "PAU backward B_5_4");
    
    m.def("forward_C_5_4", &pau_forward_C_5_4, "PAU forward C_5_4");
    m.def("backward_C_5_4", &pau_backward_C_5_4, "PAU backward C_5_4");
    
    m.def("forward_D_5_4", &pau_forward_D_5_4, "PAU forward D_5_4");
    m.def("backward_D_5_4", &pau_backward_D_5_4, "PAU backward D_5_4");
    }
    