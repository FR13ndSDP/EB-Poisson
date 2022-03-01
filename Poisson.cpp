#include "Poisson.H"

using namespace amrex;

void InitData (MultiFab& State)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(State,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();
        const Array4<Real>& q = State.array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if (AMREX_SPACEDIM == 2)
            if (i==70 && j==70 && k==0) {
                q(i,j,k) = 10.0;
#elif (AMREX_SPACEDIM == 3)
	        if (i==70 && j==70 && k==70) {
	            q(i,j,k) = 10.0;
#endif
            } else {
                q(i,j,k) = 0.0;
            }
        });
    }
}
