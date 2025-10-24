/*
 * pendulum.c
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "pendulum".
 *
 * Model version              : 2.12
 * Simulink Coder version : 24.2 (R2024b) 21-Jun-2024
 * C source code generated on : Fri Oct  3 15:40:17 2025
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "pendulum.h"
#include <math.h>
#include "rtwtypes.h"
#include "pendulum_private.h"
#include <string.h>

/* Block signals (default storage) */
B_pendulum_T pendulum_B;

/* Continuous states */
X_pendulum_T pendulum_X;

/* Disabled State Vector */
XDis_pendulum_T pendulum_XDis;

/* Block states (default storage) */
DW_pendulum_T pendulum_DW;

/* Real-time model */
static RT_MODEL_pendulum_T pendulum_M_;
RT_MODEL_pendulum_T *const pendulum_M = &pendulum_M_;

/*
 * This function updates continuous states using the ODE1 fixed-step
 * solver algorithm
 */
static void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
{
  time_T tnew = rtsiGetSolverStopTime(si);
  time_T h = rtsiGetStepSize(si);
  real_T *x = rtsiGetContStates(si);
  ODE1_IntgData *id = (ODE1_IntgData *)rtsiGetSolverData(si);
  real_T *f0 = id->f[0];
  int_T i;
  int_T nXc = 2;
  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);
  rtsiSetdX(si, f0);
  pendulum_derivatives();
  rtsiSetT(si, tnew);
  for (i = 0; i < nXc; ++i) {
    x[i] += h * f0[i];
  }

  rtsiSetSimTimeStep(si,MAJOR_TIME_STEP);
}

/* Model step function */
void pendulum_step(void)
{
  boolean_T tmp;
  if (rtmIsMajorTimeStep(pendulum_M)) {
    /* set solver stop time */
    if (!(pendulum_M->Timing.clockTick0+1)) {
      rtsiSetSolverStopTime(&pendulum_M->solverInfo,
                            ((pendulum_M->Timing.clockTickH0 + 1) *
        pendulum_M->Timing.stepSize0 * 4294967296.0));
    } else {
      rtsiSetSolverStopTime(&pendulum_M->solverInfo,
                            ((pendulum_M->Timing.clockTick0 + 1) *
        pendulum_M->Timing.stepSize0 + pendulum_M->Timing.clockTickH0 *
        pendulum_M->Timing.stepSize0 * 4294967296.0));
    }
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep(pendulum_M)) {
    pendulum_M->Timing.t[0] = rtsiGetT(&pendulum_M->solverInfo);
  }

  /* Integrator: '<S1>/Theta' */
  pendulum_B.Theta = pendulum_X.Theta_CSTATE;
  tmp = rtmIsMajorTimeStep(pendulum_M);
  if (tmp) {
    /* ToWorkspace: '<Root>/To Workspace' */
    if (rtmIsMajorTimeStep(pendulum_M)) {
      rt_UpdateLogVar((LogVar *)(LogVar*)
                      (pendulum_DW.ToWorkspace_PWORK.LoggedData),
                      &pendulum_B.Theta, 0);
    }
  }

  /* Integrator: '<S1>/Theta+' */
  pendulum_B.Theta_l = pendulum_X.Theta_CSTATE_a;
  if (tmp) {
    /* ToWorkspace: '<Root>/To Workspace1' */
    if (rtmIsMajorTimeStep(pendulum_M)) {
      rt_UpdateLogVar((LogVar *)(LogVar*)
                      (pendulum_DW.ToWorkspace1_PWORK.LoggedData),
                      &pendulum_B.Theta_l, 0);
    }
  }

  /* Logic: '<S1>/OR' incorporates:
   *  Constant: '<S1>/Constant'
   *  Constant: '<S1>/Constant1'
   *  RelationalOperator: '<S1>/Relational Operator'
   *  RelationalOperator: '<S1>/Relational Operator1'
   */
  pendulum_B.OR = ((pendulum_B.Theta >= pendulum_P.Constant_Value_i) ||
                   (pendulum_B.Theta <= pendulum_P.Constant1_Value));
  if (tmp) {
    /* Stop: '<S1>/Stop Simulation' */
    if (pendulum_B.OR) {
      rtmSetStopRequested(pendulum_M, 1);
    }

    /* End of Stop: '<S1>/Stop Simulation' */
  }

  /* Fcn: '<S1>/Theta*+' incorporates:
   *  Constant: '<Root>/Constant'
   */
  pendulum_B.Theta_j = (-0.29400000000000004 * sin(pendulum_B.Theta) +
                        pendulum_P.Constant_Value) / 0.0045;
  if (rtmIsMajorTimeStep(pendulum_M)) {
    /* Matfile logging */
    rt_UpdateTXYLogVars(pendulum_M->rtwLogInfo, (pendulum_M->Timing.t));
  }                                    /* end MajorTimeStep */

  if (rtmIsMajorTimeStep(pendulum_M)) {
    /* signal main to stop simulation */
    {                                  /* Sample time: [0.0s, 0.0s] */
      if ((rtmGetTFinal(pendulum_M)!=-1) &&
          !((rtmGetTFinal(pendulum_M)-(((pendulum_M->Timing.clockTick1+
               pendulum_M->Timing.clockTickH1* 4294967296.0)) * 0.01)) >
            (((pendulum_M->Timing.clockTick1+pendulum_M->Timing.clockTickH1*
               4294967296.0)) * 0.01) * (DBL_EPSILON))) {
        rtmSetErrorStatus(pendulum_M, "Simulation finished");
      }
    }

    rt_ertODEUpdateContinuousStates(&pendulum_M->solverInfo);

    /* Update absolute time for base rate */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick0 and the high bits
     * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
     */
    if (!(++pendulum_M->Timing.clockTick0)) {
      ++pendulum_M->Timing.clockTickH0;
    }

    pendulum_M->Timing.t[0] = rtsiGetSolverStopTime(&pendulum_M->solverInfo);

    {
      /* Update absolute timer for sample time: [0.01s, 0.0s] */
      /* The "clockTick1" counts the number of times the code of this task has
       * been executed. The resolution of this integer timer is 0.01, which is the step size
       * of the task. Size of "clockTick1" ensures timer will not overflow during the
       * application lifespan selected.
       * Timer of this task consists of two 32 bit unsigned integers.
       * The two integers represent the low bits Timing.clockTick1 and the high bits
       * Timing.clockTickH1. When the low bit overflows to 0, the high bits increment.
       */
      pendulum_M->Timing.clockTick1++;
      if (!pendulum_M->Timing.clockTick1) {
        pendulum_M->Timing.clockTickH1++;
      }
    }
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void pendulum_derivatives(void)
{
  XDot_pendulum_T *_rtXdot;
  _rtXdot = ((XDot_pendulum_T *) pendulum_M->derivs);

  /* Derivatives for Integrator: '<S1>/Theta' */
  _rtXdot->Theta_CSTATE = pendulum_B.Theta_l;

  /* Derivatives for Integrator: '<S1>/Theta+' */
  _rtXdot->Theta_CSTATE_a = pendulum_B.Theta_j;
}

/* Model initialize function */
void pendulum_initialize(void)
{
  /* Registration code */

  /* initialize real-time model */
  (void) memset((void *)pendulum_M, 0,
                sizeof(RT_MODEL_pendulum_T));

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&pendulum_M->solverInfo,
                          &pendulum_M->Timing.simTimeStep);
    rtsiSetTPtr(&pendulum_M->solverInfo, &rtmGetTPtr(pendulum_M));
    rtsiSetStepSizePtr(&pendulum_M->solverInfo, &pendulum_M->Timing.stepSize0);
    rtsiSetdXPtr(&pendulum_M->solverInfo, &pendulum_M->derivs);
    rtsiSetContStatesPtr(&pendulum_M->solverInfo, (real_T **)
                         &pendulum_M->contStates);
    rtsiSetNumContStatesPtr(&pendulum_M->solverInfo,
      &pendulum_M->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&pendulum_M->solverInfo,
      &pendulum_M->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&pendulum_M->solverInfo,
      &pendulum_M->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&pendulum_M->solverInfo,
      &pendulum_M->periodicContStateRanges);
    rtsiSetContStateDisabledPtr(&pendulum_M->solverInfo, (boolean_T**)
      &pendulum_M->contStateDisabled);
    rtsiSetErrorStatusPtr(&pendulum_M->solverInfo, (&rtmGetErrorStatus
      (pendulum_M)));
    rtsiSetRTModelPtr(&pendulum_M->solverInfo, pendulum_M);
  }

  rtsiSetSimTimeStep(&pendulum_M->solverInfo, MAJOR_TIME_STEP);
  rtsiSetIsMinorTimeStepWithModeChange(&pendulum_M->solverInfo, false);
  rtsiSetIsContModeFrozen(&pendulum_M->solverInfo, false);
  pendulum_M->intgData.f[0] = pendulum_M->odeF[0];
  pendulum_M->contStates = ((X_pendulum_T *) &pendulum_X);
  pendulum_M->contStateDisabled = ((XDis_pendulum_T *) &pendulum_XDis);
  pendulum_M->Timing.tStart = (0.0);
  rtsiSetSolverData(&pendulum_M->solverInfo, (void *)&pendulum_M->intgData);
  rtsiSetSolverName(&pendulum_M->solverInfo,"ode1");
  rtmSetTPtr(pendulum_M, &pendulum_M->Timing.tArray[0]);
  rtmSetTFinal(pendulum_M, 5.0);
  pendulum_M->Timing.stepSize0 = 0.01;

  /* Setup for data logging */
  {
    static RTWLogInfo rt_DataLoggingInfo;
    rt_DataLoggingInfo.loggingInterval = (NULL);
    pendulum_M->rtwLogInfo = &rt_DataLoggingInfo;
  }

  /* Setup for data logging */
  {
    rtliSetLogXSignalInfo(pendulum_M->rtwLogInfo, (NULL));
    rtliSetLogXSignalPtrs(pendulum_M->rtwLogInfo, (NULL));
    rtliSetLogT(pendulum_M->rtwLogInfo, "tout");
    rtliSetLogX(pendulum_M->rtwLogInfo, "");
    rtliSetLogXFinal(pendulum_M->rtwLogInfo, "");
    rtliSetLogVarNameModifier(pendulum_M->rtwLogInfo, "rt_");
    rtliSetLogFormat(pendulum_M->rtwLogInfo, 4);
    rtliSetLogMaxRows(pendulum_M->rtwLogInfo, 0);
    rtliSetLogDecimation(pendulum_M->rtwLogInfo, 1);
    rtliSetLogY(pendulum_M->rtwLogInfo, "");
    rtliSetLogYSignalInfo(pendulum_M->rtwLogInfo, (NULL));
    rtliSetLogYSignalPtrs(pendulum_M->rtwLogInfo, (NULL));
  }

  /* block I/O */
  (void) memset(((void *) &pendulum_B), 0,
                sizeof(B_pendulum_T));

  /* states (continuous) */
  {
    (void) memset((void *)&pendulum_X, 0,
                  sizeof(X_pendulum_T));
  }

  /* disabled states */
  {
    (void) memset((void *)&pendulum_XDis, 0,
                  sizeof(XDis_pendulum_T));
  }

  /* states (dwork) */
  (void) memset((void *)&pendulum_DW, 0,
                sizeof(DW_pendulum_T));

  /* Matfile logging */
  rt_StartDataLoggingWithStartTime(pendulum_M->rtwLogInfo, 0.0, rtmGetTFinal
    (pendulum_M), pendulum_M->Timing.stepSize0, (&rtmGetErrorStatus(pendulum_M)));

  /* SetupRuntimeResources for ToWorkspace: '<Root>/To Workspace' */
  {
    int_T dimensions[1] = { 1 };

    pendulum_DW.ToWorkspace_PWORK.LoggedData = rt_CreateLogVar(
      pendulum_M->rtwLogInfo,
      0.0,
      rtmGetTFinal(pendulum_M),
      pendulum_M->Timing.stepSize0,
      (&rtmGetErrorStatus(pendulum_M)),
      "angle",
      SS_DOUBLE,
      0,
      0,
      0,
      1,
      1,
      dimensions,
      NO_LOGVALDIMS,
      (NULL),
      (NULL),
      0,
      1,
      0.01,
      1);
    if (pendulum_DW.ToWorkspace_PWORK.LoggedData == (NULL))
      return;
  }

  /* SetupRuntimeResources for ToWorkspace: '<Root>/To Workspace1' */
  {
    int_T dimensions[1] = { 1 };

    pendulum_DW.ToWorkspace1_PWORK.LoggedData = rt_CreateLogVar(
      pendulum_M->rtwLogInfo,
      0.0,
      rtmGetTFinal(pendulum_M),
      pendulum_M->Timing.stepSize0,
      (&rtmGetErrorStatus(pendulum_M)),
      "angle_v",
      SS_DOUBLE,
      0,
      0,
      0,
      1,
      1,
      dimensions,
      NO_LOGVALDIMS,
      (NULL),
      (NULL),
      0,
      1,
      0.01,
      1);
    if (pendulum_DW.ToWorkspace1_PWORK.LoggedData == (NULL))
      return;
  }

  /* InitializeConditions for Integrator: '<S1>/Theta' */
  pendulum_X.Theta_CSTATE = pendulum_P.PendulumandCart_init;

  /* InitializeConditions for Integrator: '<S1>/Theta+' */
  pendulum_X.Theta_CSTATE_a = pendulum_P.Theta_IC;
}

/* Model terminate function */
void pendulum_terminate(void)
{
  /* (no terminate code required) */
}
