/*
 * pendulum_data.c
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

/* Block parameters (default storage) */
P_pendulum_T pendulum_P = {
  /* Mask Parameter: PendulumandCart_init
   * Referenced by: '<S1>/Theta'
   */
  0.07,

  /* Expression: 0
   * Referenced by: '<S1>/Theta+'
   */
  0.0,

  /* Expression: -0.6123256683349609
   * Referenced by: '<Root>/Constant'
   */
  -0.61232566833496094,

  /* Expression: pi/2
   * Referenced by: '<S1>/Constant'
   */
  1.5707963267948966,

  /* Expression: -pi/2
   * Referenced by: '<S1>/Constant1'
   */
  -1.5707963267948966
};
