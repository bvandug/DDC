#ifndef PendCart_249987d3_cap_host_h__
#define PendCart_249987d3_cap_host_h__
#ifdef HOST_CAPI_BUILD
#include "rtw_capi.h"
#include "rtw_modelmap_simtarget.h"
typedef struct { rtwCAPI_ModelMappingInfo mmi ; }
PendCart_249987d3_host_DataMapInfo_T ;
#ifdef __cplusplus
extern "C" {
#endif
void PendCart_249987d3_host_InitializeDataMapInfo ( PendCart_249987d3_host_DataMapInfo_T * dataMap , const char * path ) ;
#ifdef __cplusplus
}
#endif
#endif
#endif
