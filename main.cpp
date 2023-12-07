#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_ParmParse.H>
#include <iostream>
#include "Utils.hpp"
using namespace amrex;



namespace {
    int use_array4 = true;
    int use_array4_hack = false;
}

void Initialize (FArrayBox& prev, FArrayBox& next, FArrayBox& vel)
{
    std::cout << "Initializing ... \n";

    prev.template setVal<RunOn::Device>(0.0f);
    next.template setVal<RunOn::Device>(0.0f);
    vel.template setVal<RunOn::Device>(2250000.0f * dt * dt);

    // Add a source to initial wavefield as an initial condition
    auto nx = prev.box().length(0);
    auto ny = prev.box().length(1);
    auto nz = prev.box().length(2);
    std::cout << "nx" << nx << std::endl;
    std::cout << "ny" << ny << std::endl;
    std::cout << "nz" << nz << std::endl;
    float val = 1.f;
    for (int s = 5; s >= 0; s--)
    {
        Box b(IntVect((nx / 4) - s, (ny / 4) - s, (nz / 2) - s),
              IntVect((nx / 4) + s - 1, (ny / 4) + s + 1, (nz / 2) + s - 1));
        // std::cout << val << std::endl;
        prev.template setVal<RunOn::Device>(val, b);
        val *= 10.f;
    }
    //std::cout << prev.array()(0,0,0)<< std::endl;
    /*amrex::Real host_data;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, prev.dataPtr(), prev.dataPtr() + 255, &host_data);
    std::cout << host_data << std::endl;*/
    amrex::Print() << "Initial min, max, 1-norm, 2-norm, inf-norm, sum: "
                   << prev.template min<RunOn::Device>() << ", "
                   << prev.template max<RunOn::Device>() << ", "
                   << prev.template norm<RunOn::Device>(1) << ", "
                   << prev.template norm<RunOn::Device>(2) << ", "
                   << prev.template norm<RunOn::Device>(0) << ", "
                   << prev.template sum<RunOn::Device>(0) << "\n";
}
          
void Iso3dfd_opt (FArrayBox& nextfab, FArrayBox& prevfab, FArrayBox const& velfab,
              Gpu::DeviceVector<float> const& coeffdv, int nIterations, int n1, int n2, int n3, int n1_block, int n2_block, int z_offset, int full_end_z)
{
    std::cout << "Using opt" << std::endl;
    Box const& b = amrex::grow(nextfab.box(), -kHalfLength);
    auto nx = n1;
    auto nxy = n1 * n2;
    auto bx = kHalfLength;
    auto by = kHalfLength;
    auto grid_size = nxy * n3;
    auto const* coeff = coeffdv.data();
    for (auto it = 0; it < nIterations; it += 1) 
    {
        auto const& next = (it % 2 == 0) ? nextfab.array() : prevfab.array();
        auto const& prev = (it % 2 == 0) ? prevfab.const_array() : nextfab.const_array();
        auto const& vel = velfab.const_array();

        ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto *pn = next.ptr(i,j,k);
                auto const* pp = prev.ptr(i,j,k);
                auto const* pv = vel.ptr(i,j,k);
                float value = (*pp) * coeff[0];
                //Iso3dfdIterationGlobal(i,j,k, pn, pp, pv, nx, nxy, bx, by, n3_block, end_z);
                {
                    auto begin_z = i * z_offset + kHalfLength;
                    auto end_z = begin_z + z_offset;
                    if (end_z > full_end_z)
                        end_z = full_end_z;
                    int gid = (k + bx) + (j + by) + (begin_z * nxy);

                    Real front[kHalfLength + 1];
                    Real back[kHalfLength];
                    Real c[kHalfLength + 1];

                    for (auto iter = 0; iter <= kHalfLength; iter++)
                    {
                        front[iter] = pp[gid + iter * nxy];
                    }

                    float value = c[0] * front[0];
                #pragma unroll(kHalfLength)
                    for (auto iter = 1; iter <= kHalfLength; iter++)
                    {
                        value += c[iter] * (front[iter] + back[iter - 1] + pp[gid + iter] + pp[gid - iter] + pp[gid + iter * nx] + pp[gid - iter * nx]);
                    }
                    pn[gid] = 2.0f * front[0] - pn[gid] + value * pv[gid];

                    gid += nxy;
                    begin_z++;

                    while (begin_z < end_z)
                    {
                        // Input data in front and back are shifted to discard the
                        // oldest value and read one new value.
                        for (auto iter = kHalfLength - 1; iter > 0; iter--)
                        {
                            back[iter] = back[iter - 1];
                        }
                        back[0] = front[0];

                        for (auto iter = 0; iter < kHalfLength; iter++)
                        {
                            front[iter] = front[iter + 1];
                        }

                        // Only one new data-point read from global memory
                        // in z-dimension (depth)
                        front[kHalfLength] = pp[gid + kHalfLength * nxy];

                        // Stencil code to update grid point at position given by global id (gid)
                        float value = c[0] * front[0];
                #pragma unroll(kHalfLength)
                        for (auto iter = 1; iter <= kHalfLength; iter++)
                        {
                            value += c[iter] * (front[iter] + back[iter - 1] + pp[gid + iter] +
                                                pp[gid - iter] + pp[gid + iter * nx] +
                                                pp[gid - iter * nx]);
                        }

                        pn[gid] = 2.0f * front[0] - pn[gid] + value * pv[gid];

                        gid += nxy;
                        begin_z++;
                    }
                                                    
                }
                
            });
        
    }

}



void Iso3dfd (FArrayBox& nextfab, FArrayBox& prevfab, FArrayBox const& velfab,
              Gpu::DeviceVector<float> const& coeffdv, int num_iterations)
{
    Box const& b = amrex::grow(nextfab.box(), -kHalfLength);
    auto const* coeff = coeffdv.data();
    //std::cout << coeff[]
    for (int it = 0; it < num_iterations; ++it) {
        auto const& next = (it % 2 == 0) ? nextfab.array() : prevfab.array();
        auto const& prev = (it % 2 == 0) ? prevfab.const_array() : nextfab.const_array();
        auto const& vel = velfab.const_array();
        if (use_array4) {
            if (use_array4_hack) {
                if(it == 0)
                    std::cout << "Using array4 hack" << std::endl;
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    auto *pn = next.ptr(i,j,k);
                    auto const* pp = prev.ptr(i,j,k);
                    auto const* pv = vel.ptr(i,j,k);
                    float value = (*pp) * coeff[0];
#pragma unroll(kHalfLength)
                    for (int ir = 1; ir <= kHalfLength; ++ir) {
                        value += coeff[ir] * (pp[ ir] +
                                              pp[-ir] +
                                              pp[ ir*prev.jstride] +
                                              pp[-ir*prev.jstride] +
                                              pp[ ir*prev.kstride] +
                                              pp[-ir*prev.kstride]);
                    }
                    *pn = 2.0f * (*pp) - (*pn) + value*(*pv);
                });
            } else {
                if (it == 0)
                    std::cout << "Using array4" << std::endl;
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    float value = prev(i,j,k) * coeff[0];
                    //AMREX_DEVICE_PRINTF(" %f coeff_old:", coeff[0]);
                    //AMREX_DEVICE_PRINTF(" %f value_old:", value);
                    //AMREX_DEVICE_PRINTF(" %f prev_old:", prev(i,j,k));
                    //AMREX_DEVICE_PRINTF(" %f vel_old:", vel(i,j,k));

#pragma unroll(kHalfLength)
                    for (int ir = 1; ir <= kHalfLength; ++ir) {
                        value += coeff[ir] * (prev(i-ir,j   ,k   ) +
                                              prev(i+ir,j   ,k   ) +
                                              prev(i   ,j-ir,k   ) +
                                              prev(i   ,j+ir,k   ) +
                                              prev(i   ,j   ,k-ir) +
                                              prev(i   ,j   ,k+ir));
                        
                    }
                    //AMREX_DEVICE_PRINTF(" %f value_new:", value);
                    next(i,j,k) = 2.0f * prev(i,j,k) - next(i,j,k) + value*vel(i,j,k);
                });
            }
        } 
        
        
        else {
            if(it ==0)
            std::cout << "Using raw pointer" << std::endl;
            auto* pn = next.ptr(0,0,0);
            auto const* pp = prev.ptr(0,0,0);
            auto const* pv = vel.ptr(0,0,0);
            auto jstride = next.jstride;
            auto kstride = next.kstride;
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto offset = i + j*jstride + k*kstride;
                float value = pp[offset] * coeff[0];
#pragma unroll(kHalfLength)
                for (int ir = 1; ir <= kHalfLength; ++ir) {
                    value += coeff[ir] * (pp[offset+ir] +
                                          pp[offset-ir] +
                                          pp[offset+ir*jstride] +
                                          pp[offset-ir*jstride] +
                                          pp[offset+ir*kstride] +
                                          pp[offset-ir*kstride]);
                }
                pn[offset] = 2.0f * pp[offset] - pn[offset] + value*pv[offset];
            });
        }
    }
}



void main_main ()
{
    static_assert(std::is_same_v<float,Real>);

    std::array<int,3> grid_sizes{256,256,256};
    size_t n1 = grid_sizes[0];
    size_t n2 = grid_sizes[1];
    size_t n3 = grid_sizes[2];
    int n1_block = 32;
    int n2_block = 8;
    int n3_block = 64;
    int num_iterations = 10;
    int opt = 1;
    {
        ParmParse pp;
        pp.query("grid_sizes", grid_sizes);
        pp.query("iterations", num_iterations);
        pp.query("use_array4", use_array4);
        pp.query("use_array4_hack", use_array4_hack);
        pp.query("opt" , opt);
    }

     Box domain(IntVect(0),IntVect(grid_sizes[0]-1,
                                  grid_sizes[1]-1,
                                  grid_sizes[2]-1));
    //float* prev_cpu, next_cpu, vel_cpu;
    FArrayBox prev, next, vel;
    FArrayBox prev_cpu, next_cpu, vel_cpu;
    {
        Box fabbox = amrex::grow(domain,kHalfLength);
        prev.resize(fabbox,1);
        next.resize(fabbox,1);
        vel.resize(fabbox,1);
        prev_cpu.resize(fabbox,1,The_Pinned_Arena() );
        next_cpu.resize(fabbox,1, The_Pinned_Arena() );
        vel_cpu.resize(fabbox,1, The_Pinned_Arena() );

    }

    // Compute coefficients to be used in wavefield update
    Array<float,kHalfLength+1> coeff
        {-3.0548446f,   +1.7777778f,     -3.1111111e-1f,
         +7.572087e-2f, -1.76767677e-2f, +3.480962e-3f,
         -5.180005e-4f, +5.074287e-5f,   -2.42812e-6f};
    // Apply the DX DY and DZ to coefficients
    coeff[0] = (3.0f * coeff[0]) / (dxyz * dxyz);
    for (int i = 1; i <= kHalfLength; i++) {
        coeff[i] = coeff[i] / (dxyz * dxyz);
    }
    Gpu::DeviceVector<float> coeff_dv(coeff.size());
    Gpu::copyAsync(Gpu::hostToDevice, coeff.begin(), coeff.end(), coeff_dv.begin());

    std::cout << "Grid Sizes: " << grid_sizes[0] << " " << grid_sizes[1] << " "
              << grid_sizes[2] << "\n";
    std::cout << "Memory Usage: " << ((3*prev.nBytes()) / (1024 * 1024)) << " MB\n";


    Initialize(prev, next, vel);
    Gpu::streamSynchronize();

    if(opt){
        Iso3dfd_opt(next, prev, vel, coeff_dv, 2, n1, n2, n3, n1_block, n2_block, n3_block, n3 - kHalfLength); //warm up
        Gpu::streamSynchronize();

        auto t0 = amrex::second();
        Iso3dfd_opt(next, prev, vel, coeff_dv, num_iterations, n1, n2, n3, n1_block, n2_block, n3_block, n3 - kHalfLength);
        Gpu::streamSynchronize();
        auto t1 = amrex::second();
        printStats((t1-t0) * 1e-3, n1, n2, n3, num_iterations);
    }
    else{
    Iso3dfd(next, prev, vel, coeff_dv, 20); // warm up
    Gpu::streamSynchronize();

    auto t0 = amrex::second();
    Iso3dfd(next, prev, vel, coeff_dv, num_iterations);
    Gpu::streamSynchronize();
    auto t1 = amrex::second();
    printStats((t1-t0) * 1e3, n1, n2, n3, num_iterations);
    printStats(t1-t0, domain, num_iterations);

    amrex::Print() << "Final min, max, 1-norm, 2-norm, inf-norm, sum: "
                   << next.template min<RunOn::Device>() << ", "
                   << next.template max<RunOn::Device>() << ", "
                   << next.template norm<RunOn::Device>(1) << ", "
                   << next.template norm<RunOn::Device>(2) << ", "
                   << next.template norm<RunOn::Device>(0) << ", "
                   << next.template sum<RunOn::Device>(0) << "\n";

    
    
   // next.copyToHost();
    std::cout << next.array().size() << std::endl;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, next.dataPtr(), next.dataPtr() + next.size(), next_cpu.dataPtr());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, prev.dataPtr(), prev.dataPtr() + prev.size(), prev_cpu.dataPtr());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, vel.dataPtr(), vel.dataPtr() + vel.size(), vel_cpu.dataPtr());
    //Gpu::streamSynchronize();
    std::cout << prev_cpu.array()(67, 67, 119) << std::endl;
    std::cout << next_cpu.array()(67, 67, 119) << std::endl;
    std::cout << vel_cpu.array()(67, 67, 119) << std::endl;

    std::cout << "Starting verification " << std::endl;
    VerifyResult(prev_cpu.array().dataPtr(), next_cpu.array().dataPtr(), vel_cpu.array().dataPtr(), coeff.data(), n1 + 2*kHalfLength, n2 +  2*kHalfLength, n3 +  2*kHalfLength, num_iterations + 20);
}
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "\n\n";
        main_main();
        amrex::Print() << "\n\n";
    }
    amrex::Finalize();

     
}
