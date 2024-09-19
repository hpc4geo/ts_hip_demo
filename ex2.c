
#include <petsc.h>

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx);

int main(int argc, char **argv)
{
  TS           ts; /* ODE integrator */
  Vec          U;  /* solution will be stored here */
  PetscMPIInt  commsize;
  PetscInt     i, len = 5;
  PetscScalar  *u = NULL;
  MPI_Comm     comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));

  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &commsize));
  if (commsize > 1) SETERRQ(comm,PETSC_ERR_SUP,"This is a serial example. MPI parallelism is not supported");

  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));

  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, NULL));

  /* initial condition */
  PetscCall(VecCreate(comm, &U));
  PetscCall(VecSetSizes(U, len, PETSC_DETERMINE)); // slip-rate + state
  PetscCall(VecSetFromOptions(U));
  PetscCall(VecSetUp(U));

  PetscCall(VecGetLocalSize(U,&len));
  PetscCall(VecGetArray(U, &u));
  for (i=0; i<len; i++) {
    u[i] = 0.0;
  }
  PetscCall(VecRestoreArray(U, &u));

  PetscCall(TSSetSolution(ts, U));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, 1.0e-10));

  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSetUp(ts));

  PetscCall(TSSolve(ts, U));

  VecView(U,PETSC_VIEWER_STDOUT_(comm));

  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());

  return 0;
}
