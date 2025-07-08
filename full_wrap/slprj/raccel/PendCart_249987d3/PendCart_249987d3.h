#ifndef PendCart_249987d3_h_
#define PendCart_249987d3_h_
#ifndef PendCart_249987d3_COMMON_INCLUDES_
#define PendCart_249987d3_COMMON_INCLUDES_
#include <stdlib.h>
#include "sl_AsyncioQueue/AsyncioQueueCAPI.h"
#include "rtwtypes.h"
#include "sigstream_rtw.h"
#include "simtarget/slSimTgtSigstreamRTW.h"
#include "simtarget/slSimTgtSlioCoreRTW.h"
#include "simtarget/slSimTgtSlioClientsRTW.h"
#include "simtarget/slSimTgtSlioSdiRTW.h"
#include "simstruc.h"
#include "fixedpoint.h"
#include "raccel.h"
#include "slsv_diagnostic_codegen_c_api.h"
#include "rt_logging_simtarget.h"
#include "rt_nonfinite.h"
#include "math.h"
#include "dt_info.h"
#include "ext_work.h"
#endif
#include "PendCart_249987d3_types.h"
#include <stddef.h>
#include "rtw_modelmap_simtarget.h"
#include "rt_defines.h"
#include <string.h>
#define MODEL_NAME PendCart_249987d3
#define NSAMPLE_TIMES (3) 
#define NINPUTS (0)       
#define NOUTPUTS (0)     
#define NBLOCKIO (4) 
#define NUM_ZC_EVENTS (0) 
#ifndef NCSTATES
#define NCSTATES (2)   
#elif NCSTATES != 2
#error Invalid specification of NCSTATES defined in compiler command
#endif
#ifndef rtmGetDataMapInfo
#define rtmGetDataMapInfo(rtm) (*rt_dataMapInfoPtr)
#endif
#ifndef rtmSetDataMapInfo
#define rtmSetDataMapInfo(rtm, val) (rt_dataMapInfoPtr = &val)
#endif
#ifndef IN_RACCEL_MAIN
#endif
typedef struct { real_T gjzy5tfd3w ; real_T bnoc2r3zb1 ; real_T ookh2etl03 ;
boolean_T b2kznrnnft ; } B ; typedef struct { struct { void * AQHandles ; }
kaxu24zx3x ; struct { void * AQHandles ; } ddkprtk3xc ; } DW ; typedef struct
{ real_T jy10erohzi ; real_T daj4zgvcxt ; } X ; typedef struct { real_T
jy10erohzi ; real_T daj4zgvcxt ; } XDot ; typedef struct { boolean_T
jy10erohzi ; boolean_T daj4zgvcxt ; } XDis ; typedef struct {
rtwCAPI_ModelMappingInfo mmi ; } DataMapInfo ; struct P_ { real_T u ; real_T
PendulumandCart_init ; real_T Theta_IC ; real_T Constant_Value ; real_T
Constant1_Value ; } ; extern const char_T * RT_MEMORY_ALLOCATION_ERROR ;
extern B rtB ; extern X rtX ; extern DW rtDW ; extern P rtP ; extern mxArray
* mr_PendCart_249987d3_GetDWork ( ) ; extern void
mr_PendCart_249987d3_SetDWork ( const mxArray * ssDW ) ; extern mxArray *
mr_PendCart_249987d3_GetSimStateDisallowedBlocks ( ) ; extern const
rtwCAPI_ModelMappingStaticInfo * PendCart_249987d3_GetCAPIStaticMap ( void )
; extern SimStruct * const rtS ; extern DataMapInfo * rt_dataMapInfoPtr ;
extern rtwCAPI_ModelMappingInfo * rt_modelMapInfoPtr ; void MdlOutputs ( int_T
tid ) ; void MdlOutputsParameterSampleTime ( int_T tid ) ; void MdlUpdate ( int_T tid ) ; void MdlTerminate ( void ) ; void MdlInitializeSizes ( void ) ; void MdlInitializeSampleTimes ( void ) ; SimStruct * raccel_register_model ( ssExecutionInfo * executionInfo ) ;
#endif
