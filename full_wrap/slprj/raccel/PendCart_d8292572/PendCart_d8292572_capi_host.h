#ifndef PendCart_d8292572_cap_host_h__
#define PendCart_d8292572_cap_host_h__
#ifdef HOST_CAPI_BUILD
#include "rtw_capi.h"
#include "rtw_modelmap_simtarget.h"
typedef struct { rtwCAPI_ModelMappingInfo mmi ; }
PendCart_d8292572_host_DataMapInfo_T ;
#ifdef __cplusplus
extern "C" {
#endif
void PendCart_d8292572_host_InitializeDataMapInfo ( PendCart_d8292572_host_DataMapInfo_T * dataMap , const char * path ) ;
#ifdef __cplusplus
}
#endif
#endif
#endif
