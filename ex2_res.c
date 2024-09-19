
#include <petsc.h>

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
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
