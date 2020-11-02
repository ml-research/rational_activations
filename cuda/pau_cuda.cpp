
#include <torch/extension.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)




    
	at::Tensor pau_cuda_forward_A_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_A_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_A_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_A_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_A_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_A_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_A_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_A_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_A_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor pau_forward_A_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_A_3_3(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_A_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_A_3_3(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_A_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_A_4_4(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_A_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_A_4_4(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_A_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_A_5_5(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_A_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_A_5_5(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_A_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_A_6_6(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_A_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_A_6_6(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_A_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_A_7_7(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_A_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_A_7_7(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_A_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_A_8_8(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_A_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_A_8_8(grad_output, x, n, d);
    }
    
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
    
    at::Tensor pau_forward_A_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_A_7_6(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_A_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_A_7_6(grad_output, x, n, d);
    }
    

    
	at::Tensor pau_cuda_forward_B_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_B_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_B_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_B_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_B_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_B_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_B_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_B_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_B_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor pau_forward_B_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_B_3_3(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_B_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_B_3_3(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_B_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_B_4_4(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_B_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_B_4_4(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_B_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_B_5_5(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_B_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_B_5_5(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_B_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_B_6_6(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_B_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_B_6_6(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_B_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_B_7_7(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_B_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_B_7_7(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_B_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_B_8_8(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_B_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_B_8_8(grad_output, x, n, d);
    }
    
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
    
    at::Tensor pau_forward_B_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_B_7_6(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_B_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_B_7_6(grad_output, x, n, d);
    }
    

    
	at::Tensor pau_cuda_forward_C_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_C_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_C_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_C_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_C_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_C_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_C_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_C_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_C_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor pau_forward_C_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_C_3_3(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_C_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_C_3_3(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_C_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_C_4_4(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_C_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_C_4_4(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_C_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_C_5_5(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_C_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_C_5_5(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_C_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_C_6_6(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_C_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_C_6_6(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_C_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_C_7_7(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_C_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_C_7_7(grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_C_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_C_8_8(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_C_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_C_8_8(grad_output, x, n, d);
    }
    
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
    
    at::Tensor pau_forward_C_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_C_7_6(x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_C_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_C_7_6(grad_output, x, n, d);
    }
    
    
	at::Tensor pau_cuda_forward_D_3_3(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_3_3(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_D_4_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_4_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_D_5_5(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_5_5(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_D_6_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_6_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_D_7_7(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_7_7(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_D_8_8(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_8_8(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor pau_cuda_forward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> pau_cuda_backward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor pau_forward_D_3_3(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_D_3_3(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_D_3_3(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_D_3_3(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_D_4_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_D_4_4(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_D_4_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_D_4_4(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_D_5_5(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_D_5_5(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_D_5_5(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_D_5_5(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_D_6_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_D_6_6(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_D_6_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_D_6_6(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_D_7_7(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_D_7_7(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_D_7_7(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_D_7_7(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor pau_forward_D_8_8(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_D_8_8(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_D_8_8(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_D_8_8(training, iteration, grad_output, x, n, d);
    }
    
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
    
    at::Tensor pau_forward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_forward_D_7_6(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> pau_backward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return pau_cuda_backward_D_7_6(training, iteration, grad_output, x, n, d);
    }
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    
    m.def("forward_A_3_3", &pau_forward_A_3_3, "PAU forward A_3_3");
    m.def("backward_A_3_3", &pau_backward_A_3_3, "PAU backward A_3_3");
    
    m.def("forward_B_3_3", &pau_forward_B_3_3, "PAU forward B_3_3");
    m.def("backward_B_3_3", &pau_backward_B_3_3, "PAU backward B_3_3");
    
    m.def("forward_C_3_3", &pau_forward_C_3_3, "PAU forward C_3_3");
    m.def("backward_C_3_3", &pau_backward_C_3_3, "PAU backward C_3_3");
    
    m.def("forward_D_3_3", &pau_forward_D_3_3, "PAU forward D_3_3");
    m.def("backward_D_3_3", &pau_backward_D_3_3, "PAU backward D_3_3");
    
    
    m.def("forward_A_4_4", &pau_forward_A_4_4, "PAU forward A_4_4");
    m.def("backward_A_4_4", &pau_backward_A_4_4, "PAU backward A_4_4");
    
    m.def("forward_B_4_4", &pau_forward_B_4_4, "PAU forward B_4_4");
    m.def("backward_B_4_4", &pau_backward_B_4_4, "PAU backward B_4_4");
    
    m.def("forward_C_4_4", &pau_forward_C_4_4, "PAU forward C_4_4");
    m.def("backward_C_4_4", &pau_backward_C_4_4, "PAU backward C_4_4");
    
    m.def("forward_D_4_4", &pau_forward_D_4_4, "PAU forward D_4_4");
    m.def("backward_D_4_4", &pau_backward_D_4_4, "PAU backward D_4_4");
    
    
    m.def("forward_A_5_5", &pau_forward_A_5_5, "PAU forward A_5_5");
    m.def("backward_A_5_5", &pau_backward_A_5_5, "PAU backward A_5_5");
    
    m.def("forward_B_5_5", &pau_forward_B_5_5, "PAU forward B_5_5");
    m.def("backward_B_5_5", &pau_backward_B_5_5, "PAU backward B_5_5");
    
    m.def("forward_C_5_5", &pau_forward_C_5_5, "PAU forward C_5_5");
    m.def("backward_C_5_5", &pau_backward_C_5_5, "PAU backward C_5_5");
    
    m.def("forward_D_5_5", &pau_forward_D_5_5, "PAU forward D_5_5");
    m.def("backward_D_5_5", &pau_backward_D_5_5, "PAU backward D_5_5");
    
    
    m.def("forward_A_6_6", &pau_forward_A_6_6, "PAU forward A_6_6");
    m.def("backward_A_6_6", &pau_backward_A_6_6, "PAU backward A_6_6");
    
    m.def("forward_B_6_6", &pau_forward_B_6_6, "PAU forward B_6_6");
    m.def("backward_B_6_6", &pau_backward_B_6_6, "PAU backward B_6_6");
    
    m.def("forward_C_6_6", &pau_forward_C_6_6, "PAU forward C_6_6");
    m.def("backward_C_6_6", &pau_backward_C_6_6, "PAU backward C_6_6");
    
    m.def("forward_D_6_6", &pau_forward_D_6_6, "PAU forward D_6_6");
    m.def("backward_D_6_6", &pau_backward_D_6_6, "PAU backward D_6_6");
    
    
    m.def("forward_A_7_7", &pau_forward_A_7_7, "PAU forward A_7_7");
    m.def("backward_A_7_7", &pau_backward_A_7_7, "PAU backward A_7_7");
    
    m.def("forward_B_7_7", &pau_forward_B_7_7, "PAU forward B_7_7");
    m.def("backward_B_7_7", &pau_backward_B_7_7, "PAU backward B_7_7");
    
    m.def("forward_C_7_7", &pau_forward_C_7_7, "PAU forward C_7_7");
    m.def("backward_C_7_7", &pau_backward_C_7_7, "PAU backward C_7_7");
    
    m.def("forward_D_7_7", &pau_forward_D_7_7, "PAU forward D_7_7");
    m.def("backward_D_7_7", &pau_backward_D_7_7, "PAU backward D_7_7");
    
    
    m.def("forward_A_8_8", &pau_forward_A_8_8, "PAU forward A_8_8");
    m.def("backward_A_8_8", &pau_backward_A_8_8, "PAU backward A_8_8");
    
    m.def("forward_B_8_8", &pau_forward_B_8_8, "PAU forward B_8_8");
    m.def("backward_B_8_8", &pau_backward_B_8_8, "PAU backward B_8_8");
    
    m.def("forward_C_8_8", &pau_forward_C_8_8, "PAU forward C_8_8");
    m.def("backward_C_8_8", &pau_backward_C_8_8, "PAU backward C_8_8");
    
    m.def("forward_D_8_8", &pau_forward_D_8_8, "PAU forward D_8_8");
    m.def("backward_D_8_8", &pau_backward_D_8_8, "PAU backward D_8_8");
    
    
    m.def("forward_A_5_4", &pau_forward_A_5_4, "PAU forward A_5_4");
    m.def("backward_A_5_4", &pau_backward_A_5_4, "PAU backward A_5_4");
    
    m.def("forward_B_5_4", &pau_forward_B_5_4, "PAU forward B_5_4");
    m.def("backward_B_5_4", &pau_backward_B_5_4, "PAU backward B_5_4");
    
    m.def("forward_C_5_4", &pau_forward_C_5_4, "PAU forward C_5_4");
    m.def("backward_C_5_4", &pau_backward_C_5_4, "PAU backward C_5_4");
    
    m.def("forward_D_5_4", &pau_forward_D_5_4, "PAU forward D_5_4");
    m.def("backward_D_5_4", &pau_backward_D_5_4, "PAU backward D_5_4");
    
    
    m.def("forward_A_7_6", &pau_forward_A_7_6, "PAU forward A_7_6");
    m.def("backward_A_7_6", &pau_backward_A_7_6, "PAU backward A_7_6");
    
    m.def("forward_B_7_6", &pau_forward_B_7_6, "PAU forward B_7_6");
    m.def("backward_B_7_6", &pau_backward_B_7_6, "PAU backward B_7_6");
    
    m.def("forward_C_7_6", &pau_forward_C_7_6, "PAU forward C_7_6");
    m.def("backward_C_7_6", &pau_backward_C_7_6, "PAU backward C_7_6");
    
    m.def("forward_D_7_6", &pau_forward_D_7_6, "PAU forward D_7_6");
    m.def("backward_D_7_6", &pau_backward_D_7_6, "PAU backward D_7_6");
    }
    