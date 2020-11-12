
#include <torch/extension.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)




    
	at::Tensor rational_cuda_forward_A_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_A_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_A_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_A_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_A_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_A_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_A_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_A_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_A_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_A_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_A_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_A_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_A_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_A_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_A_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_A_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor rational_forward_A_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_A_3_3(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_A_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_A_3_3(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_A_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_A_4_4(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_A_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_A_4_4(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_A_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_A_5_5(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_A_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_A_5_5(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_A_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_A_6_6(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_A_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_A_6_6(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_A_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_A_7_7(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_A_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_A_7_7(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_A_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_A_8_8(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_A_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_A_8_8(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_A_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_A_5_4(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_A_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_A_5_4(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_A_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_A_7_6(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_A_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_A_7_6(grad_output, x, n, d);
    }
    

    
	at::Tensor rational_cuda_forward_B_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_B_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_B_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_B_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_B_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_B_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_B_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_B_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_B_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_B_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_B_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_B_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_B_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_B_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_B_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_B_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor rational_forward_B_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_B_3_3(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_B_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_B_3_3(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_B_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_B_4_4(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_B_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_B_4_4(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_B_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_B_5_5(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_B_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_B_5_5(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_B_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_B_6_6(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_B_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_B_6_6(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_B_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_B_7_7(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_B_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_B_7_7(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_B_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_B_8_8(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_B_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_B_8_8(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_B_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_B_5_4(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_B_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_B_5_4(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_B_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_B_7_6(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_B_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_B_7_6(grad_output, x, n, d);
    }
    

    
	at::Tensor rational_cuda_forward_C_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_C_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_C_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_C_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_C_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_C_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_C_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_C_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_C_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_C_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_C_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_C_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_C_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_C_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_C_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_C_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor rational_forward_C_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_C_3_3(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_C_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_C_3_3(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_C_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_C_4_4(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_C_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_C_4_4(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_C_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_C_5_5(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_C_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_C_5_5(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_C_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_C_6_6(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_C_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_C_6_6(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_C_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_C_7_7(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_C_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_C_7_7(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_C_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_C_8_8(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_C_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_C_8_8(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_C_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_C_5_4(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_C_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_C_5_4(grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_C_7_6(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_C_7_6(x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_C_7_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_C_7_6(grad_output, x, n, d);
    }
    
    
	at::Tensor rational_cuda_forward_D_3_3(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_D_3_3(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_D_4_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_D_4_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_D_5_5(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_D_5_5(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_D_6_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_D_6_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_D_7_7(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_D_7_7(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_D_8_8(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_D_8_8(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    
	at::Tensor rational_cuda_forward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    std::vector<torch::Tensor> rational_cuda_backward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
    

    
    at::Tensor rational_forward_D_3_3(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_D_3_3(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_D_3_3(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_D_3_3(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_D_4_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_D_4_4(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_D_4_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_D_4_4(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_D_5_5(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_D_5_5(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_D_5_5(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_D_5_5(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_D_6_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_D_6_6(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_D_6_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_D_6_6(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_D_7_7(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_D_7_7(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_D_7_7(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_D_7_7(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_D_8_8(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_D_8_8(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_D_8_8(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_D_8_8(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_D_5_4(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_D_5_4(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_D_5_4(training, iteration, grad_output, x, n, d);
    }
    
    at::Tensor rational_forward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_forward_D_7_6(training, iteration, x, n, d);
    }
    std::vector<torch::Tensor> rational_backward_D_7_6(const bool training, const unsigned long long iteration, torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);

        return rational_cuda_backward_D_7_6(training, iteration, grad_output, x, n, d);
    }
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    
    m.def("forward_A_3_3", &rational_forward_A_3_3, "Rational forward A_3_3");
    m.def("backward_A_3_3", &rational_backward_A_3_3, "Rational backward A_3_3");
    
    m.def("forward_B_3_3", &rational_forward_B_3_3, "Rational forward B_3_3");
    m.def("backward_B_3_3", &rational_backward_B_3_3, "Rational backward B_3_3");
    
    m.def("forward_C_3_3", &rational_forward_C_3_3, "Rational forward C_3_3");
    m.def("backward_C_3_3", &rational_backward_C_3_3, "Rational backward C_3_3");
    
    m.def("forward_D_3_3", &rational_forward_D_3_3, "Rational forward D_3_3");
    m.def("backward_D_3_3", &rational_backward_D_3_3, "Rational backward D_3_3");
    
    
    m.def("forward_A_4_4", &rational_forward_A_4_4, "Rational forward A_4_4");
    m.def("backward_A_4_4", &rational_backward_A_4_4, "Rational backward A_4_4");
    
    m.def("forward_B_4_4", &rational_forward_B_4_4, "Rational forward B_4_4");
    m.def("backward_B_4_4", &rational_backward_B_4_4, "Rational backward B_4_4");
    
    m.def("forward_C_4_4", &rational_forward_C_4_4, "Rational forward C_4_4");
    m.def("backward_C_4_4", &rational_backward_C_4_4, "Rational backward C_4_4");
    
    m.def("forward_D_4_4", &rational_forward_D_4_4, "Rational forward D_4_4");
    m.def("backward_D_4_4", &rational_backward_D_4_4, "Rational backward D_4_4");
    
    
    m.def("forward_A_5_5", &rational_forward_A_5_5, "Rational forward A_5_5");
    m.def("backward_A_5_5", &rational_backward_A_5_5, "Rational backward A_5_5");
    
    m.def("forward_B_5_5", &rational_forward_B_5_5, "Rational forward B_5_5");
    m.def("backward_B_5_5", &rational_backward_B_5_5, "Rational backward B_5_5");
    
    m.def("forward_C_5_5", &rational_forward_C_5_5, "Rational forward C_5_5");
    m.def("backward_C_5_5", &rational_backward_C_5_5, "Rational backward C_5_5");
    
    m.def("forward_D_5_5", &rational_forward_D_5_5, "Rational forward D_5_5");
    m.def("backward_D_5_5", &rational_backward_D_5_5, "Rational backward D_5_5");
    
    
    m.def("forward_A_6_6", &rational_forward_A_6_6, "Rational forward A_6_6");
    m.def("backward_A_6_6", &rational_backward_A_6_6, "Rational backward A_6_6");
    
    m.def("forward_B_6_6", &rational_forward_B_6_6, "Rational forward B_6_6");
    m.def("backward_B_6_6", &rational_backward_B_6_6, "Rational backward B_6_6");
    
    m.def("forward_C_6_6", &rational_forward_C_6_6, "Rational forward C_6_6");
    m.def("backward_C_6_6", &rational_backward_C_6_6, "Rational backward C_6_6");
    
    m.def("forward_D_6_6", &rational_forward_D_6_6, "Rational forward D_6_6");
    m.def("backward_D_6_6", &rational_backward_D_6_6, "Rational backward D_6_6");
    
    
    m.def("forward_A_7_7", &rational_forward_A_7_7, "Rational forward A_7_7");
    m.def("backward_A_7_7", &rational_backward_A_7_7, "Rational backward A_7_7");
    
    m.def("forward_B_7_7", &rational_forward_B_7_7, "Rational forward B_7_7");
    m.def("backward_B_7_7", &rational_backward_B_7_7, "Rational backward B_7_7");
    
    m.def("forward_C_7_7", &rational_forward_C_7_7, "Rational forward C_7_7");
    m.def("backward_C_7_7", &rational_backward_C_7_7, "Rational backward C_7_7");
    
    m.def("forward_D_7_7", &rational_forward_D_7_7, "Rational forward D_7_7");
    m.def("backward_D_7_7", &rational_backward_D_7_7, "Rational backward D_7_7");
    
    
    m.def("forward_A_8_8", &rational_forward_A_8_8, "Rational forward A_8_8");
    m.def("backward_A_8_8", &rational_backward_A_8_8, "Rational backward A_8_8");
    
    m.def("forward_B_8_8", &rational_forward_B_8_8, "Rational forward B_8_8");
    m.def("backward_B_8_8", &rational_backward_B_8_8, "Rational backward B_8_8");
    
    m.def("forward_C_8_8", &rational_forward_C_8_8, "Rational forward C_8_8");
    m.def("backward_C_8_8", &rational_backward_C_8_8, "Rational backward C_8_8");
    
    m.def("forward_D_8_8", &rational_forward_D_8_8, "Rational forward D_8_8");
    m.def("backward_D_8_8", &rational_backward_D_8_8, "Rational backward D_8_8");
    
    
    m.def("forward_A_5_4", &rational_forward_A_5_4, "Rational forward A_5_4");
    m.def("backward_A_5_4", &rational_backward_A_5_4, "Rational backward A_5_4");
    
    m.def("forward_B_5_4", &rational_forward_B_5_4, "Rational forward B_5_4");
    m.def("backward_B_5_4", &rational_backward_B_5_4, "Rational backward B_5_4");
    
    m.def("forward_C_5_4", &rational_forward_C_5_4, "Rational forward C_5_4");
    m.def("backward_C_5_4", &rational_backward_C_5_4, "Rational backward C_5_4");
    
    m.def("forward_D_5_4", &rational_forward_D_5_4, "Rational forward D_5_4");
    m.def("backward_D_5_4", &rational_backward_D_5_4, "Rational backward D_5_4");
    
    
    m.def("forward_A_7_6", &rational_forward_A_7_6, "Rational forward A_7_6");
    m.def("backward_A_7_6", &rational_backward_A_7_6, "Rational backward A_7_6");
    
    m.def("forward_B_7_6", &rational_forward_B_7_6, "Rational forward B_7_6");
    m.def("backward_B_7_6", &rational_backward_B_7_6, "Rational backward B_7_6");
    
    m.def("forward_C_7_6", &rational_forward_C_7_6, "Rational forward C_7_6");
    m.def("backward_C_7_6", &rational_backward_C_7_6, "Rational backward C_7_6");
    
    m.def("forward_D_7_6", &rational_forward_D_7_6, "Rational forward D_7_6");
    m.def("backward_D_7_6", &rational_backward_D_7_6, "Rational backward D_7_6");
    }
    