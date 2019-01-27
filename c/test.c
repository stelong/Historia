#include <stdio.h>
#include <cvode/cvode.h>
#include <cvode/cvode_ls.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>
// #include <sundials/sundials_dense.h>
// #include <sundials/sundials_direct.h>
// #include <sundials/sundials_iterative.h>
// #include <sundials/sundials_linearsolver.h>
// #include <sundials/sundials_matrix.h>
// #include <sundials/sundials_nonlinearsolver.h>
// #include <sundials/sundials_nvector.h>
// #include <sundials/sundials_config.h>

// gcc -I/usr/include/ -L/usr/lib/ -o test test.c -lsundials_cvode -lsundials_nvecserial

#define Ith(v,i)    NV_Ith_S(v,i-1)
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1)

#define NEQ   3
#define Y1    RCONST(1.0)
#define Y2    RCONST(0.0)
#define Y3    RCONST(0.0)
#define RTOL  RCONST(1.0e-4)
#define ATOL1 RCONST(1.0e-8)
#define ATOL2 RCONST(1.0e-14)
#define ATOL3 RCONST(1.0e-6)
#define T0    RCONST(0.0)
#define T1    RCONST(0.4)
#define TMULT RCONST(10.0)
#define NOUT  12
#define ZERO  RCONST(0.0)

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static int g(realtype t, N_Vector y, realtype *gout, void *user_data);
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, 
	void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3);
static void PrintRootInfo(int root_f1, int root_f2);
static void PrintFinalStats(void *cvode_mem);
static int check_retval(void *returnvalue, const char *funcname, int opt);
static int check_ans(N_Vector y, realtype t, realtype rtol, N_Vector atol);

int main() {
	realtype reltol, t, tout;
	N_Vector y, abstol;
	SUNMatrix A;
	SUNLinearSolver LS;
	void *cvode_mem;
	int retval, retvalr, iout;
	int rootsfound[2];

	y = abstol = NULL;
	A = NULL;
	LS = NULL;
	cvode_mem = NULL;

	y = N_VNew_Serial(NEQ);
	if (check_retval((void *)y, "N_VNew_Serial", 0)) return(1);

	abstol = N_VNew_Serial(NEQ); 
	if (check_retval((void *)abstol, "N_VNew_Serial", 0)) return(1);

	Ith(y,1) = Y1;
	Ith(y,2) = Y2;
	Ith(y,3) = Y3;

	reltol = RTOL;

	Ith(abstol,1) = ATOL1;
	Ith(abstol,2) = ATOL2;
	Ith(abstol,3) = ATOL3;

	cvode_mem = CVodeCreate(CV_BDF);
	if (check_retval((void *)cvode_mem, "CVodeCreate", 0)) return(1);

	retval = CVodeInit(cvode_mem, f, T0, y);
	if (check_retval(&retval, "CVodeInit", 1)) return(1);

	retval = CVodeSVtolerances(cvode_mem, reltol, abstol);
	if (check_retval(&retval, "CVodeSVtolerances", 1)) return(1);

	retval = CVodeRootInit(cvode_mem, 2, g);
	if (check_retval(&retval, "CVodeRootInit", 1)) return(1);

	A = SUNDenseMatrix(NEQ, NEQ);
	if(check_retval((void *)A, "SUNDenseMatrix", 0)) return(1);

	// LS = SUNLinSol_Dense(y, A);
	// if(check_retval((void *)LS, "SUNLinSol_Dense", 0)) return(1);

	// retval = CVodeSetLinearSolver(cvode_mem, LS, A);
	// if(check_retval(&retval, "CVodeSetLinearSolver", 1)) return(1);

	// retval = CVodeSetJacFn(cvode_mem, Jac);
	// if(check_retval(&retval, "CVodeSetJacFn", 1)) return(1);

	printf(" \n3-species kinetics problem\n\n");

	return 0;
}

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype y1, y2, y3, yd1, yd3;

  y1 = Ith(y,1); y2 = Ith(y,2); y3 = Ith(y,3);

  yd1 = Ith(ydot,1) = RCONST(-0.04)*y1 + RCONST(1.0e4)*y2*y3;
  yd3 = Ith(ydot,3) = RCONST(3.0e7)*y2*y2;
        Ith(ydot,2) = -yd1 - yd3;

  return(0);
}

/*
 * g routine. Compute functions g_i(t,y) for i = 0,1. 
 */

static int g(realtype t, N_Vector y, realtype *gout, void *user_data)
{
  realtype y1, y3;

  y1 = Ith(y,1); y3 = Ith(y,3);
  gout[0] = y1 - RCONST(0.0001);
  gout[1] = y3 - RCONST(0.01);

  return(0);
}


static int check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *) returnvalue;
    if (*retval < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
	      funcname, *retval);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  return(0);
}

static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, 
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  realtype y2, y3;

  y2 = Ith(y,2); y3 = Ith(y,3);

  IJth(J,1,1) = RCONST(-0.04);
  IJth(J,1,2) = RCONST(1.0e4)*y3;
  IJth(J,1,3) = RCONST(1.0e4)*y2;

  IJth(J,2,1) = RCONST(0.04); 
  IJth(J,2,2) = RCONST(-1.0e4)*y3-RCONST(6.0e7)*y2;
  IJth(J,2,3) = RCONST(-1.0e4)*y2;

  IJth(J,3,1) = ZERO;
  IJth(J,3,2) = RCONST(6.0e7)*y2;
  IJth(J,3,3) = ZERO;

  return(0);
}