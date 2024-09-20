
#include <petsc.h>


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

PetscErrorCode ex4_2(void)
{
  DM           da;
  Vec          U;
  PetscInt     i, len = 5;
  MPI_Comm     comm;

  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;
  PetscCall(DMDACreate1d(comm, DM_BOUNDARY_NONE, len, 1, 1, NULL, &da));

  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da, &U));

  PetscCall(VecSet(U,2.0));
  PetscCall(VecView(U,PETSC_VIEWER_STDOUT_WORLD));


  PetscCall(VecDestroy(&U));
  PetscCall(DMDestroy(&da));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt idx=1;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));

  PetscOptionsGetInt(NULL,NULL,"-idx",&idx,NULL);
  switch (idx) {
case 1: PetscCall(ex4_1()); break;
case 2: PetscCall(ex4_2()); break;
  }

  PetscCall(PetscFinalize());
  return 0;
}
