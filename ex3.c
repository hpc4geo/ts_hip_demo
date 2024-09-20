
#include <petsc.h>

PetscErrorCode RHSFunction_seq(TS ts, PetscReal t, Vec U, Vec F, void *ctx);
PetscErrorCode RHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx);
PetscErrorCode xRHSFunction_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx);


PetscErrorCode ex4_1(void)
{
  Vec          U;
  PetscInt     i, len = 5;
  MPI_Comm     comm;

  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;
  PetscCall(VecCreate(comm, &U));
  PetscCall(VecSetSizes(U, len, PETSC_DETERMINE));
  PetscCall(VecSetType(U,VECSTANDARD));
  PetscCall(VecSetFromOptions(U));
  PetscCall(VecSetUp(U));

  PetscCall(VecSet(U,2.0));
  PetscCall(VecView(U,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&U));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode xxx(void)
{
  TS           ts; /* ODE integrator */
  Vec          U;  /* solution will be stored here */
  PetscMPIInt  commsize;
  PetscInt     i, len = 1024*1024;
  PetscScalar  *u = NULL;
  MPI_Comm     comm;

  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &commsize));
  if (commsize > 1) SETERRQ(comm,PETSC_ERR_SUP,"This is a serial example. MPI parallelism is not supported");

  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));

  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction_seq, NULL));

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

  PetscCall(TSSetSolution(ts, U));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, 1.0e-10));

  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSetUp(ts));

  PetscCall(TSSolve(ts, U));

  Vec F;
  VecDuplicate(U,&F);
  VecSet(F,2.0);
  VecAXPY(U,1.0,F);
  //VecView(U,PETSC_VIEWER_STDOUT_(comm));

  //printf("single call\n");
  //PetscCall(xRHSFunction_hip(ts,0.0, U,F,NULL));


  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));

  //PetscCall(ex4_1());
  PetscCall(xxx());

  PetscCall(PetscFinalize());
  return 0;
}

