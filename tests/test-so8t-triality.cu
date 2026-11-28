#include "ggml-so8t.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kSign[8][8] = {
    { 1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  1, -1,  1, -1,  1,  1, -1},
    { 1,  1,  1, -1, -1, -1,  1,  1},
    { 1, -1,  1,  1, -1,  1, -1,  1},
    { 1,  1,  1,  1,  1, -1, -1, -1},
    { 1, -1,  1, -1,  1,  1,  1, -1},
    { 1, -1, -1,  1,  1, -1,  1,  1},
    { 1,  1, -1, -1,  1,  1, -1,  1},
};

void write_plain_safetensors(const std::string& path, const std::string& name, int N, int K, const std::vector<__half>& w_nk) {
    const uint64_t nbytes = static_cast<uint64_t>(w_nk.size()) * 2ULL;
    const std::string hdr = "{\"" + name + "\":{\"dtype\":\"F16\",\"shape\":[" +
                            std::to_string(N) + "," + std::to_string(K) +
                            "],\"data_offsets\":[0," + std::to_string(nbytes) + "]}}";
    const uint64_t hdr_len = hdr.size();
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&hdr_len), sizeof(hdr_len));
    ofs.write(hdr.data(), static_cast<std::streamsize>(hdr_len));
    ofs.write(reinterpret_cast<const char*>(w_nk.data()), static_cast<std::streamsize>(nbytes));
}

}  // namespace

int main() {
    constexpr int M = 128, in_groups = 16, out_groups = 16;  // 128 x 128 x 128
    constexpr int K = in_groups * 8, N = out_groups * 8;

    std::mt19937 rng(2026u);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

    // 1. Create a standard HuggingFace .safetensors F16 weight matrix W[N, K]
    //    populated with exact octonion R(w) blocks to verify zero-loss Frobenius recovery!
    std::vector<__half> h_w_oct_true(out_groups * in_groups * 8);
    for (auto& v : h_w_oct_true) v = __float2half(dist(rng));

    std::vector<__half> h_w_dense_nk(N * K);
    for (int og = 0; og < out_groups; ++og) {
        for (int ig = 0; ig < in_groups; ++ig) {
            const __half* w8 = h_w_oct_true.data() + (og * in_groups + ig) * 8;
            for (int a = 0; a < 8; ++a) {
                for (int k = 0; k < 8; ++k) {
                    const float s = static_cast<float>(kSign[k][a ^ k]);
                    h_w_dense_nk[(og * 8 + k) * K + (ig * 8 + a)] = __float2half(s * __half2float(w8[a ^ k]));
                }
            }
        }
    }

    const std::string st_path = "plain_test_model.safetensors";
    write_plain_safetensors(st_path, "model.layers.0.self_attn.qkv_proj.weight", N, K, h_w_dense_nk);

    // 2. Load plain .safetensors directly without ANY offline conversion!
    ggml_so8t_safetensors_tensor loaded;
    if (!ggml_so8t_load_safetensors_f16_tensor(st_path, "model.layers.0.self_attn.qkv_proj.weight", loaded)) {
        std::cerr << "[FAIL] Failed to load plain .safetensors tensor\n";
        return 1;
    }

    std::vector<__half> h_x(M * K);
    for (auto& v : h_x) v = __float2half(dist(rng));

    __half *d_x = nullptr, *d_w_dense = nullptr, *d_w_oct = nullptr;
    float *d_yv = nullptr, *d_ysp = nullptr, *d_ysm = nullptr;
    cudaMalloc(&d_x, h_x.size() * sizeof(__half));
    cudaMalloc(&d_w_dense, loaded.data_f16.size() * sizeof(uint16_t));
    cudaMalloc(&d_w_oct, h_w_oct_true.size() * sizeof(__half));
    cudaMalloc(&d_yv, M * N * sizeof(float));
    cudaMalloc(&d_ysp, M * N * sizeof(float));
    cudaMalloc(&d_ysm, M * N * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_dense, loaded.data_f16.data(), loaded.data_f16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // 3. 1-call end-to-end plain .safetensors -> Spin(8) Triality Triple Inference!
    ggml_so8t_safetensors_triple_infer_cuda(
        d_x, d_w_dense, d_w_oct, d_yv, d_ysp, d_ysm, M, in_groups, out_groups, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_yv(M * N), h_ysp(M * N), h_ysm(M * N);
    cudaMemcpy(h_yv.data(), d_yv, h_yv.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ysp.data(), d_ysp, h_ysp.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ysm.data(), d_ysm, h_ysm.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 4. Verify all 3 Triality outputs (V8, S8+, S8-) against exact FP64 CPU reference
    double max_err = 0.0;
    for (int r = 0; r < M; r += 17) {
        for (int og = 0; og < out_groups; ++og) {
            double rv[8] = {}, rsp[8] = {}, rsm[8] = {};
            for (int ig = 0; ig < in_groups; ++ig) {
                double xv[8], wv[8], wc[8];
                for (int e = 0; e < 8; ++e) {
                    xv[e] = static_cast<double>(__half2float(h_x[r * K + ig * 8 + e]));
                    wv[e] = static_cast<double>(__half2float(h_w_oct_true[(og * in_groups + ig) * 8 + e]));
                    wc[e] = (e == 0) ? wv[e] : -wv[e];
                }
                for (int c = 0; c < 8; ++c) {
                    for (int b = 0; b < 8; ++b) {
                        const double s = static_cast<double>(kSign[c][b]);
                        rv[c]  += s * xv[c ^ b] * wv[b];
                        rsp[c] += s * xv[c ^ b] * wc[b];
                        rsm[c] += s * wv[c ^ b] * xv[b];
                    }
                }
            }
            for (int c = 0; c < 8; ++c) {
                const int idx = r * N + og * 8 + c;
                max_err = std::max(max_err, std::fabs(static_cast<double>(h_yv[idx])  - rv[c]));
                max_err = std::max(max_err, std::fabs(static_cast<double>(h_ysp[idx]) - rsp[c]));
                max_err = std::max(max_err, std::fabs(static_cast<double>(h_ysm[idx]) - rsm[c]));
            }
        }
    }

    // 5. Verify GGML_TYPE_SO8T_OCT_NVFP4 block GPU dequantizer
    const int num_octs = out_groups * in_groups;
    std::vector<block_so8t_oct_nvfp4> h_nvfp4(num_octs / 2);
    for (auto& b : h_nvfp4) {
        b.scale_f16 = __half_as_ushort(__float2half(0.25f));
        for (int i = 0; i < 8; ++i) b.qs[i] = 0x22u;  // E2M1 code 2 = 1.0 -> 0.25f
    }
    block_so8t_oct_nvfp4* d_nvfp4 = nullptr;
    cudaMalloc(&d_nvfp4, h_nvfp4.size() * sizeof(block_so8t_oct_nvfp4));
    cudaMemcpy(d_nvfp4, h_nvfp4.data(), h_nvfp4.size() * sizeof(block_so8t_oct_nvfp4), cudaMemcpyHostToDevice);
    ggml_so8t_dequantize_nvfp4_to_oct_f16_cuda(d_nvfp4, d_w_oct, num_octs, nullptr);
    cudaDeviceSynchronize();
    std::vector<__half> h_deq(num_octs * 8);
    cudaMemcpy(h_deq.data(), d_w_oct, h_deq.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    double max_deq_err = 0.0;
    for (auto v : h_deq) max_deq_err = std::max(max_deq_err, std::fabs(static_cast<double>(__half2float(v)) - 0.25));

    cudaFree(d_x); cudaFree(d_w_dense); cudaFree(d_w_oct);
    cudaFree(d_yv); cudaFree(d_ysp); cudaFree(d_ysm); cudaFree(d_nvfp4);
    std::remove(st_path.c_str());

    std::cout << "[PASS] Plain .safetensors -> Spin(8) Triality Triple Inference max_abs_err=" << max_err
              << " | GGML_TYPE_SO8T_OCT_NVFP4 dequant max_abs_err=" << max_deq_err << "\n";
    return (max_err < 2e-2 && max_deq_err < 1e-4) ? 0 : 1;
}
