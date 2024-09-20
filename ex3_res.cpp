
#include <petsc.h>
#include <hip/hip_runtime.h>

#define GRID_MAX 1024
#define THREAD_MAX 1024


void build_sizes(long int N,dim3 *b, dim3 *t)
{
  if (N < GRID_MAX * THREAD_MAX) {
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = (N + GRID_MAX-1)/GRID_MAX; b->y = 1; b->z = 1;
    return;
  }

  if (N < GRID_MAX * GRID_MAX * THREAD_MAX) {
    long int bJ = (N-1)/(GRID_MAX*GRID_MAX);
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = GRID_MAX; b->y = bJ; b->z = 1;

  }

}


//#define HCC_ENABLE_PRINTF

__global__ void __RHSFunction_hip(PetscInt len, PetscScalar f[])
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  //printf("block index %d %d %d | thread index %d,%d,%d --> i %d\n",(int)blockIdx.x,(int)blockIdx.y,(int)blockIdx.z,(int)threadIdx.x,(int)threadIdx.y,(int)threadIdx.z, i);
  if (i >= len) return;

    //printf("  i %d (b %d -> t %d)\n",i,(int)blockIdx.x,(int)threadIdx.x);

    f[i] = (PetscScalar)i + 0.1;
}

extern "C" {


PetscErrorCode RHSFunction_seq(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
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

PetscErrorCode RHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscScalar           *f;
  const PetscScalar     *u;
  PetscInt              i,len;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));

  //PetscCall(VecGetArrayRead(U, &u));
  //PetscCall(VecGetArray(F, &f));
  PetscCall(VecHIPGetArrayRead(U, &u));
  PetscCall(VecHIPGetArray(F, &f));

  for (i=0; i<len; i++) {
    f[i] = (PetscScalar)i + 0.1;
  }
  PetscCall(VecHIPRestoreArrayRead(U, &u));
  PetscCall(VecHIPRestoreArray(F, &f));
  //PetscCall(VecRestoreArrayRead(U, &u));
  //PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode xRHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  int ierr;
  PetscScalar           *f;
  const PetscScalar     *u;
  PetscInt              i,len;
  //PetscMemType          mt;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));

  //PetscCall(VecGetArrayReadAndMemType(U, &u, &mt));
  //PetscCall(VecGetArrayAndMemType(F, &f, &mt));
  PetscCall(VecHIPGetArrayRead(U, &u));
  PetscCall(VecHIPGetArray(F, &f));

  //__RHSFunction_hip <<< len/WARPS_PER_BLOCK+1, WARPS_PER_BLOCK*32 >>> (len, f);

  // Maximum Block Dimensions: 1024 x 1024 x 1024
  //Maximum Threads Per Block: 1024
  dim3 threads(1024,1,1);
  dim3 blocks((len+1024-1)/1024,1,1);
  __RHSFunction_hip <<< blocks, threads, 0, 0 >>> (len, f);

  //for (i=0; i<len; i++) {
  //  f[i] = (PetscScalar)i + 0.1;
  //}

  PetscCall(VecHIPRestoreArrayRead(U, &u));
  PetscCall(VecHIPRestoreArray(F, &f));
  //PetscCall(VecRestoreArrayReadAndMemType(U, &u));
  //PetscCall(VecRestoreArrayAndMemType(F, &f));

  //ierr = hipDeviceSynchronize();
  PetscFunctionReturn(PETSC_SUCCESS);
}
} /* extern C */

