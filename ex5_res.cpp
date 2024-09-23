
#include <petsc.h>
#if PetscDefined(HAVE_HIP)
  #include <hip/hip_runtime.h>
#endif
#include "ex5_ctx.h"

#define GRID_MAX 1024
#define THREAD_MAX 1024


#if PetscDefined(HAVE_HIP)
void build_sizes(long int N,dim3 *b, dim3 *t)
{
  b->x = b->y = b->z = 1;
  t->x = t->y = t->z = 1;

  if (N < GRID_MAX * THREAD_MAX) {
    printf("1d ->\n");

    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = (N + GRID_MAX-1)/GRID_MAX; b->y = 1; b->z = 1;

    return;
  }

  if (N < GRID_MAX * GRID_MAX * THREAD_MAX) {
    printf("2d ->\n");

    long int bJ = (N-1)/(GRID_MAX*GRID_MAX);
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = GRID_MAX; b->y = bJ+1; b->z = 1;

    return;
  }


  long int Nt = (long int)(N/THREAD_MAX);
  if ( Nt < GRID_MAX * GRID_MAX * GRID_MAX) {
    printf("3d ->\n");

    long int bK = (long int) ( Nt/(GRID_MAX*GRID_MAX) );
    long int N2d = Nt - GRID_MAX * bK;
    long int bJ = N2d/GRID_MAX;
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = GRID_MAX; b->y = GRID_MAX; b->z = bK+1;

    return;
  }
}

long int printd3(dim3 *b) {
  printf("(%ld %ld %ld) -> max items %ld\n",(long)b->x,(long)b->y,(long)b->z, (long)b->x*b->y*b->z);
  return (long int)b->x*b->y*b->z;
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

#endif

extern "C" {


PetscErrorCode RHSFunction_seq(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscScalar           *f;
  const PetscScalar     *u;
  PetscInt              i,len;
  Context               *data = (Context*)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));

  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));

  for (i=0; i<len; i++) {
    f[i] = data->host->elements[i] + 0.1;
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
  Context               *data = (Context*)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));

  PetscCall(VecHIPGetArrayRead(U, &u));
  PetscCall(VecHIPGetArray(F, &f));

  for (i=0; i<len; i++) {
    f[i] = data->host->elements[i] + 0.1;
  }
  PetscCall(VecHIPRestoreArrayRead(U, &u));
  PetscCall(VecHIPRestoreArray(F, &f));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode xRHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  int ierr;
  PetscScalar           *f;
  const PetscScalar     *u;
  PetscInt              i,len;
  Context               *data = (Context*)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));

  PetscCall(VecHIPGetArrayRead(U, &u));
  PetscCall(VecHIPGetArray(F, &f));

#if PetscDefined(HAVE_HIP)
  // Maximum Block Dimensions: 1024 x 1024 x 1024
  //Maximum Threads Per Block: 1024
  //dim3 threads(1024,1,1);
  //dim3 blocks((len+1024-1)/1024,1,1);
  dim3 blocks, threads;
  build_sizes(len, &blocks, &threads);
  {
    long int bm,tm;
    printf("blocks "); bm = printd3(&blocks);
    printf("threads "); tm = printd3(&threads);
    printf("max %ld | N %ld\n",bm * tm,(long int)len);

  }

  __RHSFunction_hip <<< blocks, threads, 0, 0 >>> (len, f, *(data->device));
#elseif
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only valid with HIP support");
#endif

  //for (i=0; i<len; i++) {
  //  f[i] = data->host->elements[i] + 0.1;
  //}

  PetscCall(VecHIPRestoreArrayRead(U, &u));
  PetscCall(VecHIPRestoreArray(F, &f));

#if PetscDefined(HAVE_HIP)
  ierr = hipDeviceSynchronize();
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
} /* extern C */
