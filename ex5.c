
#include <petsc.h>
#if PetscDefined(HAVE_HIP)
  #include <hip/hip_runtime.h>
#endif

#include "ex5_ctx.h"

PetscErrorCode RHSFunction_seq(TS ts, PetscReal t, Vec U, Vec F, void *ctx);
PetscErrorCode RHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx);
PetscErrorCode xRHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx);



PetscErrorCode ParamsCreate(int np,Params *p)
{
  int i;
  PetscFunctionBeginUser;
  p->npoints = np;
  PetscCall(PetscCalloc1(np,&p->elements));
  for (i=0; i<np; i++) {
    p->elements[i] = (double)(i) + 0.1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParamsCreate_Device(Params *p,Params *device_p)
{
  PetscFunctionBeginUser;
  device_p->npoints = p->npoints;
  device_p->elements = NULL;
  #if PetscDefined(HAVE_HIP)
    if (!device_p->elements) {
      hipMalloc((void**)&device_p->elements, sizeof(double)*device_p->npoints);
    }
    hipMemcpyHtoD(device_p->elements, p->elements, sizeof(double)*device_p->npoints);
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ex5(void)
{
  TS           ts; /* ODE integrator */
  Vec          U;  /* solution will be stored here */
  PetscMPIInt  commsize;
  PetscInt     i, len = 1024 * 1024 * 5;
  PetscScalar  *u = NULL;
  MPI_Comm     comm;
  Context      *ctx;

  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &commsize));
  if (commsize > 1) SETERRQ(comm,PETSC_ERR_SUP,"This is a serial example. MPI parallelism is not supported");

  /* initial condition */
  PetscCall(VecCreate(comm, &U));
  PetscCall(VecSetSizes(U, len, PETSC_DETERMINE)); // slip-rate + state
  PetscCall(VecSetType(U,VECSTANDARD));
  PetscCall(VecSetFromOptions(U));
  PetscCall(VecSetUp(U));


  PetscCall(VecGetLocalSize(U,&len));
  PetscCall(VecGetArray(U, &u));
  for (i=0; i<len; i++) {
    u[i] = 1.0;
  }
  PetscCall(VecRestoreArray(U, &u));

  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));

  PetscCalloc1(1,&ctx);
  PetscCalloc1(1,&ctx->host);
  PetscCalloc1(1,&ctx->device);

  PetscCall(ParamsCreate(len, ctx->host));
  PetscCall(ParamsCreate_Device(ctx->host, ctx->device));

  {
    PetscBool isseq;

    PetscCall(PetscObjectTypeCompare((PetscObject)U,"seq",&isseq));
    if (isseq) {
	     PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction_seq, (void*)ctx));
     } else {
       //PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction_hip, (void*)ctx));
       PetscCall(TSSetRHSFunction(ts, NULL, xRHSFunction_hip, (void*)ctx));
     }
  }

  PetscCall(TSSetSolution(ts, U));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, 1.0e-10));

  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSetUp(ts));

  PetscCall(TSSolve(ts, U));

  //VecView(U,PETSC_VIEWER_STDOUT_WORLD);

  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));

  PetscCall(ex5());

  PetscCall(PetscFinalize());
  return 0;
}
