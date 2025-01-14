#include <sycl/sycl.hpp>
#include <iostream>
#include "iso3dfd.hpp"

void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, int n1,
                int n2, int n3) {
  auto dim2 = n2 * n1;

  for (auto i = 0; i < n3; i++) {
    for (auto j = 0; j < n2; j++) {
      auto offset = i * dim2 + j * n1;

      for (auto k = 0; k < n1; k++) {
        ptr_prev[offset + k] = 0.0f;
        ptr_next[offset + k] = 0.0f;
        ptr_vel[offset + k] =
            2250000.0f * dt * dt;  // Integration of the v*v and dt*dt here
      }
    }
  }
  std::cout << ptr_vel[2] << std::endl;
  // Then we add a source
  float val = 1.f;
  for (auto s = 5; s >= 0; s--) {
    for (auto i = n3 / 2 - s; i < n3 / 2 + s; i++) {
      for (auto j = n2 / 4 - s; j < n2 / 4 + s; j++) {
        auto offset = i * dim2 + j * n1;
        for (auto k = n1 / 4 - s; k < n1 / 4 + s; k++) {
          ptr_prev[offset + k] = val;
        }
      }
    }
    std::cout << val << std::endl;
    val *= 10;
  }
}

void printStats (double time, amrex::Box const& domain, int nIterations)
{
    std::cout << domain.d_numPts() << std::endl;
    auto normalized_time = time / double(nIterations);
    auto throughput_mpoints = domain.d_numPts() / normalized_time / 1.e6;

    auto mflops = (7.0 * double(kHalfLength) + 5.0) * throughput_mpoints / 1.e3;
    auto mbytes = 12.0 * throughput_mpoints / 1.e3;

    std::cout << "--------------------------------------\n";
    std::cout << "time         : " << time << " secs\n";
    std::cout << "throughput   : " << throughput_mpoints << " Mpts/s\n";
    std::cout << "flops        : " << mflops << " GFlops\n";
    std::cout << "bytes        : " << mbytes << " GBytes/s\n";
    std::cout << "\n--------------------------------------\n";
    std::cout << "\n--------------------------------------\n";

}

void printStats(double time, size_t n1, size_t n2, size_t n3,
                size_t num_iterations) {
  std::cout << n1*n2*n3 << std::endl;
  std::cout << "Print Stats " << std::endl;
  std::cout << time << std::endl;
  float throughput_mpoints = 0.0f, mflops = 0.0f, normalized_time = 0.0f;
  double mbytes = 0.0f;

  normalized_time = (double)time / num_iterations;
  std::cout << normalized_time << std::endl;
  throughput_mpoints = ((n1 ) * (n2) *
                        (n3)) /
                       (normalized_time * 1e3f);
  mflops = (7.0f * kHalfLength + 5.0f) * throughput_mpoints;
  mbytes = 12.0f * throughput_mpoints;

  std::cout << "--------------------------------------\n";
  std::cout << "time         : " << time / 1e3f << " secs\n";
  std::cout << "throughput   : " << throughput_mpoints << " Mpts/s\n";
  std::cout << "flops        : " << mflops / 1e3f << " GFlops\n";
  std::cout << "bytes        : " << mbytes / 1e3f << " GBytes/s\n";
  std::cout << "\n--------------------------------------\n";
  std::cout << "\n--------------------------------------\n";
}

bool WithinEpsilon(float* output, float* reference, const size_t dim_x,
                   const size_t dim_y, const size_t dim_z,
                   const unsigned int radius, const int zadjust = 0,
                   const float delta = 0.01f) {
  std::ofstream error_file;
  error_file.open("error_diff.txt");

  bool error = false;
  double norm2 = 0;

  for (size_t iz = 0; iz < dim_z; iz++) {
    for (size_t iy = 0; iy < dim_y; iy++) {
      for (size_t ix = 0; ix < dim_x; ix++) {
        if (ix >= radius && ix < (dim_x - radius) && iy >= radius &&
            iy < (dim_y - radius) && iz >= radius &&
            iz < (dim_z - radius + zadjust)) {
          float difference = fabsf(*reference - *output);
          norm2 += difference * difference;
          if (difference > delta) {
            error = true;
            error_file << " ERROR: " << ix << ", " << iy << ", " << iz << "   "
                       << *output << "   instead of " << *reference
                       << "  (|e|=" << difference << ")\n";
          }
        }
        ++output;
        ++reference;
      }
    }
  }

  error_file.close();
  norm2 = sqrt(norm2);
  if (error) std::cout << "error (Euclidean norm): " << norm2 << "\n";
  return error;
}

void inline iso3dfdCPUIteration(float* ptr_next_base, float* ptr_prev_base,
                                float* ptr_vel_base, float* coeff,
                                const size_t n1, const size_t n2,
                                const size_t n3) {
  auto dimn1n2 = n1 * n2;

  auto n3_end = n3 - kHalfLength;
  auto n2_end = n2 - kHalfLength;
  auto n1_end = n1 - kHalfLength;

  for (auto iz = kHalfLength; iz < n3_end; iz++) {
    for (auto iy = kHalfLength; iy < n2_end; iy++) {
      float* ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1;
      float* ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1;
      float* ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1;

      for (auto ix = kHalfLength; ix < n1_end; ix++) {
        float value = ptr_prev[ix] * coeff[0];
        value += STENCIL_LOOKUP(1);
        value += STENCIL_LOOKUP(2);
        value += STENCIL_LOOKUP(3);
        value += STENCIL_LOOKUP(4);
        value += STENCIL_LOOKUP(5);
        value += STENCIL_LOOKUP(6);
        value += STENCIL_LOOKUP(7);
        value += STENCIL_LOOKUP(8);

        ptr_next[ix] = 2.0f * ptr_prev[ix] - ptr_next[ix] + value * ptr_vel[ix];
      }
    }
  }
}

void CalculateReference(float* next, float* prev, float* vel, float* coeff,
                        const size_t n1, const size_t n2, const size_t n3,
                        const size_t nreps) {
  for (auto it = 0; it < nreps; it += 1) {
    iso3dfdCPUIteration(next, prev, vel, coeff, n1, n2, n3);
    std::swap(next, prev);
  }
}

void VerifyResult( float* prev,  float* next,  float* vel, float* coeff,
                  const size_t n1, const size_t n2, const size_t n3,
                  const size_t nreps) {
  std::cout << "Running CPU version for result comparasion: ";
  auto nsize = n1 * n2 * n3;
  //std::cout << nsize << std::endl;
  float* temp = new float[nsize];
  memcpy(temp, prev, nsize * sizeof(float));
  initialize(prev, next, vel, n1, n2, n3);
  CalculateReference(next, prev, vel, coeff, n1, n2, n3, nreps);
  bool error = WithinEpsilon(temp, prev, n1, n2, n3, kHalfLength, 0, 0.1f);
  if (error) {
    std::cout << "Final wavefields from SYCL device and CPU are not "
              << "equivalent: Fail\n";
  } else {
    std::cout << "Final wavefields from SYCL device and CPU are equivalent:"
              << " Success\n";
  }
  delete[] temp;
}
