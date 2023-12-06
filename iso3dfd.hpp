constexpr int kHalfLength = 8;
constexpr float dxyz = 50.0f;
constexpr float dt = 0.002f;

#define STENCIL_LOOKUP(ir)                                          \
  (coeff[ir] * ((ptr_prev[ix + ir] + ptr_prev[ix - ir]) +           \
                (ptr_prev[ix + ir * n1] + ptr_prev[ix - ir * n1]) + \
                (ptr_prev[ix + ir * dimn1n2] + ptr_prev[ix - ir * dimn1n2])))