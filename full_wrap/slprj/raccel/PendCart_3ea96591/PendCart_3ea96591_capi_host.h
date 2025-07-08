#ifndef PendCart_3ea96591_cap_host_h__
#define PendCart_3ea96591_cap_host_h__
#ifdef HOST_CAPI_BUILD
#include "rtw_capi.h"
#include "rtw_modelmap_simtarget.h"
typedef struct { rtwCAPI_ModelMappingInfo mmi ; }
PendCart_3ea96591_host_DataMapInfo_T ;
#ifdef __cplusplus
extern "C" {
#endif
void PendCart_3ea96591_host_InitializeDataMapInfo ( PendCart_3ea96591_host_DataMapInfo_T * dataMap , const char * path ) ;
#ifdef __cplusplus
}
#endif
#endif
#endif
