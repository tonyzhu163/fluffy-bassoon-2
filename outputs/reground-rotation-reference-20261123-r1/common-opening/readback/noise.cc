// Recreate the released float32 CPU randn(50,32) path without loading PyTorch or CUDA.
// Uniform conversion and 16-wide Box-Muller follow the installed ATen CPU headers.
#include <ATen/native/cpu/avx_mathfun.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::mt19937 engine(static_cast<uint32_t>(std::stoull(argv[1])));
    std::array<float, 50 * 32> out{};
    for (auto& value : out) value = (engine() & 0xffffffu) * 0x1p-24f;
    const auto tau = _mm256_set1_ps(static_cast<float>(2.0 * std::acos(-1.0)));
    for (size_t offset = 0; offset < out.size(); offset += 16) {
        auto* block = out.data() + offset;
        const auto u1 = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_loadu_ps(block));
        const auto theta = _mm256_mul_ps(tau, _mm256_loadu_ps(block + 8));
        const auto radius = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), log256_ps(u1)));
        __m256 sine, cosine;
        sincos256_ps(theta, &sine, &cosine);
        _mm256_storeu_ps(block, _mm256_mul_ps(radius, cosine));
        _mm256_storeu_ps(block + 8, _mm256_mul_ps(radius, sine));
    }
    return std::fwrite(out.data(), sizeof(float), out.size(), stdout) == out.size() ? 0 : 1;
}
