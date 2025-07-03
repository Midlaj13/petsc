static char help[] = "Solves a 3D nonlinear Poisson-type PDE using PETSc.\n";


#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

// Custom application context
typedef struct {
  DM        da;    // DMDA object 
} AppCtx;

// Declare user-defined functions
extern PetscErrorCode FormFunctionLocal(SNES, Vec, Vec, void *);
extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormInitialGuess(AppCtx *, Vec);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);



int main(int argc, char **argv)
{
    SNES          snes;     // Nonlinear solver
    Vec           x, r;     // x: solution vector, r: residual vector
    Mat           J = NULL; // Jacobian matrix (optional here)
    AppCtx        user;     // User-defined context
    PetscInt      its;      // Iteration count
    MatFDColoring matfdcoloring = NULL;
    PetscBool     matrix_free = PETSC_FALSE, coloring = PETSC_FALSE, coloring_ds = PETSC_FALSE, local_coloring = PETSC_FALSE;
    PetscReal     fnorm;

  // Initialize PETSc
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
      // Create SNES solver
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));

  // Create 3D DMDA grid: 20x20x20 global, auto MPI partitioning
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_STAR, 20, 20, 20,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                         1, 1, NULL, NULL, NULL, &user.da));

  PetscCall(DMSetFromOptions(user.da)); // allow command-line overrides
  PetscCall(DMSetUp(user.da));          // finalize the DMDA setup
  PetscCall(DMCreateGlobalVector(user.da, &x)); // Create solution vector x
  PetscCall(VecDuplicate(x, &r));               // Duplicate structure for residual vector r
  PetscCall(SNESSetFunction(snes, r, FormFunction, (void *)&user));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-snes_mf", &matrix_free, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-fdcoloring", &coloring, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-fdcoloring_ds", &coloring_ds, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-fdcoloring_local", &local_coloring, NULL));

  if (!matrix_free) {
    PetscCall(DMSetMatType(user.da, MATAIJ));
    PetscCall(DMCreateMatrix(user.da, &J));
    if (coloring) {
      ISColoring iscoloring;
      if (!local_coloring) {
        PetscCall(DMCreateColoring(user.da, IS_COLORING_GLOBAL, &iscoloring));
        PetscCall(MatFDColoringCreate(J, iscoloring, &matfdcoloring));
        PetscCall(MatFDColoringSetFunction(matfdcoloring, (PetscErrorCode (*)(void))FormFunction, &user));
      } else {
        PetscCall(DMCreateColoring(user.da, IS_COLORING_LOCAL, &iscoloring));
        PetscCall(MatFDColoringCreate(J, iscoloring, &matfdcoloring));
        PetscCall(MatFDColoringUseDM(J, matfdcoloring));
        PetscCall(MatFDColoringSetFunction(matfdcoloring, (PetscErrorCode (*)(void))FormFunctionLocal, &user));
      }
      if (coloring_ds) PetscCall(MatFDColoringSetType(matfdcoloring, MATMFFD_DS));
      PetscCall(MatFDColoringSetFromOptions(matfdcoloring));
      PetscCall(MatFDColoringSetUp(J, iscoloring, matfdcoloring));
      PetscCall(SNESSetJacobian(snes, J, J, SNESComputeJacobianDefaultColor, matfdcoloring));
      PetscCall(ISColoringDestroy(&iscoloring));
    } else {
      PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, &user));
    }
  }
  PetscCall(SNESSetDM(snes, user.da));        // Attach DM to SNES
  PetscCall(SNESSetFromOptions(snes));        // Apply command-line options to SNES

  PetscCall(FormInitialGuess(&user, x));
  PetscCall(SNESSolve(snes, NULL, x));


  PetscInt       i, j, k, Mx, My, Mz, xs, ys, zs, xm, ym, zm;
PetscScalar  ***x_array;
PetscReal      hx, hy, hz;
FILE          *f;

PetscCall(DMDAGetInfo(user.da, PETSC_IGNORE, &Mx, &My, &Mz,
                      PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
                      PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
                      PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

hx = 1.0 / (Mx - 1);
hy = 1.0 / (My - 1);
hz = 1.0 / (Mz - 1);

PetscCall(DMDAVecGetArrayRead(user.da, x, &x_array));
PetscCall(DMDAGetCorners(user.da, &xs, &ys, &zs, &xm, &ym, &zm));

// Only let rank 0 write
PetscMPIInt rank;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
if (rank == 0) {
  f = fopen("solution_tecplot.dat", "w");

  fprintf(f, "TITLE = \"Solution Field\"\n");
  fprintf(f, "VARIABLES = \"X\", \"Y\", \"Z\", \"Phi\"\n");
  fprintf(f, "ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT\n", Mx, My, Mz);

  for (k = 0; k < Mz; k++) {
    for (j = 0; j < My; j++) {
      for (i = 0; i < Mx; i++) {
        PetscReal x = i * hx;
        PetscReal y = j * hy;
        PetscReal z = k * hz;

        PetscScalar phi = 0.0;

        // If current process owns this point, use its value
        if (i >= xs && i < xs + xm &&
            j >= ys && j < ys + ym &&
            k >= zs && k < zs + zm) {
          phi = x_array[k][j][i];
        }

        fprintf(f, "%g %g %g %g\n", x, y, z, PetscRealPart(phi));
      }
    }
  }
  fclose(f);
}

PetscCall(DMDAVecRestoreArrayRead(user.da, x, &x_array));





  PetscCall(SNESGetIterationNumber(snes, &its));
  PetscCall(FormFunction(snes, x, r, (void *)&user));
  PetscCall(VecNorm(r, NORM_2, &fnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %" PetscInt_FMT " fnorm %g\n", its, (double)fnorm));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&user.da));
  PetscCall(MatFDColoringDestroy(&matfdcoloring));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormInitialGuess(AppCtx *user, Vec X)
{
  PetscInt       i, j, k, Mx, My, Mz, xs, ys, zs, xm, ym, zm;
  PetscScalar ***x;

  PetscFunctionBeginUser;

  // Get global grid size
  PetscCall(DMDAGetInfo(user->da, PETSC_IGNORE, &Mx, &My, &Mz,
                        PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
                        PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
                        PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  // Get access to vector data
  PetscCall(DMDAVecGetArray(user->da, X, &x));

  // Get local grid portion
  PetscCall(DMDAGetCorners(user->da, &xs, &ys, &zs, &xm, &ym, &zm));

  // Set initial guess: small value inside, zero at boundary
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        if (i == 0 || j == 0 || k == 0 || i == Mx - 1 || j == My - 1 || k == Mz - 1) {
          x[k][j][i] = 0.0; // Dirichlet BC
        } else {
          x[k][j][i] = 0.01; // small nonzero initial value
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(user->da, X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode FormFunctionLocal(SNES snes, Vec localX, Vec F, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscInt    i, j, k, Mx, My, Mz, xs, ys, zs, xm, ym, zm;
  PetscReal   two = 2.0, hx, hy, hz, hxhzdhy, hyhzdhx, hxhydhz;
  PetscScalar u_north, u_south, u_east, u_west, u_up, u_down, u;
  PetscScalar u_xx, u_yy, u_zz, ***x, ***f;
  DM          da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz,
                        PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
                        PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
                        PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx      = 1.0 / (PetscReal)(Mx - 1);
  hy      = 1.0 / (PetscReal)(My - 1);
  hz      = 1.0 / (PetscReal)(Mz - 1);
  hxhzdhy = hx * hz / hy;
  hyhzdhx = hy * hz / hx;
  hxhydhz = hx * hy / hz;

  PetscCall(DMDAVecGetArrayRead(da, localX, &x));
  PetscCall(DMDAVecGetArray(da, F, &f));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        if (i == 0 ) {
          f[k][j][i] = x[k][j][i]-1.0; // Dirichlet BC: φ = 0
        }
        else if ( j == 0 || k == 0 || i == Mx - 1 || j == My - 1 || k == Mz - 1)
        {
          f[k][j][i] = 0.0;
        
        } else {
          u       = x[k][j][i];
          u_east  = x[k][j][i + 1];
          u_west  = x[k][j][i - 1];
          u_north = x[k][j + 1][i];
          u_south = x[k][j - 1][i];
          u_up    = x[k + 1][j][i];
          u_down  = x[k - 1][j][i];

          u_xx = (-u_east + two * u - u_west) * hyhzdhx;
          u_yy = (-u_north + two * u - u_south) * hxhzdhy;
          u_zz = (-u_up + two * u - u_down) * hxhydhz;

          f[k][j][i] = u_xx + u_yy + u_zz - (6.0 * u * u) + (2.0 * u) + (4.0 * u * u * u);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, localX, &x));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(PetscLogFlops(12.0 * xm * ym * zm)); // updated flop count
  PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *ptr)
{
  Vec localX;
  DM  da;

  PetscFunctionBeginUser;

  // 1. Get DM (distributed mesh)
  PetscCall(SNESGetDM(snes, &da));

  // 2. Create a local ghosted vector
  PetscCall(DMGetLocalVector(da, &localX));

  // 3. Scatter global X into localX (fills ghost cells)
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  // 4. Evaluate the function locally on each process
  PetscCall(FormFunctionLocal(snes, localX, F, ptr));

  // 5. Cleanup
  PetscCall(DMRestoreLocalVector(da, &localX));

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode FormJacobian(SNES snes, Vec X, Mat J, Mat jac, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  Vec         localX;
  PetscInt    i, j, k, Mx, My, Mz;
  MatStencil  col[7], row;
  PetscInt    xs, ys, zs, xm, ym, zm;
  PetscScalar v[7], hx, hy, hz, hxhzdhy, hyhzdhx, hxhydhz, ***x;
  DM          da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, &Mz, PETSC_IGNORE, PETSC_IGNORE,
                        PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE,
                        PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx      = 1.0 / (PetscReal)(Mx - 1);
  hy      = 1.0 / (PetscReal)(My - 1);
  hz      = 1.0 / (PetscReal)(Mz - 1);
  hxhzdhy = hx * hz / hy;
  hyhzdhx = hy * hz / hx;
  hxhydhz = hx * hy / hz;

  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  PetscCall(DMDAVecGetArrayRead(da, localX, &x));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        row.k = k; row.j = j; row.i = i;

        if (i == 0 || j == 0 || k == 0 || i == Mx - 1 || j == My - 1 || k == Mz - 1) {
          v[0] = 1.0;
          PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES));
        } else {
          PetscScalar phi = x[k][j][i];
          PetscScalar dfdphi = -12.0 * phi + 2.0 + 12.0 * phi * phi; // derivative of -6phi^2 + 2phi + 4phi^3

          v[0] = -hxhydhz;                     col[0].k = k - 1; col[0].j = j;     col[0].i = i;
          v[1] = -hxhzdhy;                     col[1].k = k;     col[1].j = j - 1; col[1].i = i;
          v[2] = -hyhzdhx;                     col[2].k = k;     col[2].j = j;     col[2].i = i - 1;
          v[3] = 2.0 * (hyhzdhx + hxhzdhy + hxhydhz) + dfdphi; // center
                                             col[3].k = k;     col[3].j = j;     col[3].i = i;
          v[4] = -hyhzdhx;                    col[4].k = k;     col[4].j = j;     col[4].i = i + 1;
          v[5] = -hxhzdhy;                    col[5].k = k;     col[5].j = j + 1; col[5].i = i;
          v[6] = -hxhydhz;                    col[6].k = k + 1; col[6].j = j;     col[6].i = i;

          PetscCall(MatSetValuesStencil(jac, 1, &row, 7, col, v, INSERT_VALUES));
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, localX, &x));
  PetscCall(DMRestoreLocalVector(da, &localX));

  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(jac, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}