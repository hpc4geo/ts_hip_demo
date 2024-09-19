
#include <petsc.h>
#include <hip/hip_runtime.h>

#define WARPS_PER_BLOCK 4

__global__ void __RHSFunction_hip(PetscInt len, PetscScalar f[])
{
  //int i, idx_in_warp = hipThreadIdx_x;//idx_in_warp = threadIdx.x % 32;
  int i, idx_in_warp = idx_in_warp = threadIdx.x % 32;


  if (idx_in_warp >= len) return;

  i = idx_in_warp;

    f[i] = (PetscScalar)i + 0.1;
}

extern "C" {

PetscErrorCode RHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscScalar           *f;
  const PetscScalar     *u;
  PetscInt              i,len;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));

  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));

  for (i=0; i<len; i++) {
    f[i] = (PetscScalar)i + 0.1;
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode xRHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  int ierr;
  PetscScalar           *f;
  const PetscScalar     *u;
  PetscInt              len;
  PetscMemType          mt;

  PetscCall(VecGetLocalSize(U,&len));

  PetscCall(VecGetArrayReadAndMemType(U, &u, &mt));
  PetscCall(VecGetArrayAndMemType(F, &f, &mt));

  //__RHSFunction_hip <<< len/WARPS_PER_BLOCK+1, WARPS_PER_BLOCK*32 >>> (len, f);

  PetscCall(VecRestoreArrayReadAndMemType(U, &u));
  PetscCall(VecRestoreArrayAndMemType(F, &f));

  //ierr = hipDeviceSynchronize();
  PetscFunctionReturn(PETSC_SUCCESS);
}
} /* extern C */

