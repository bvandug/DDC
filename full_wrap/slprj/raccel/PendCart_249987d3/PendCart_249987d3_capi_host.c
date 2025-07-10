#include "PendCart_249987d3_capi_host.h"
static PendCart_249987d3_host_DataMapInfo_T root;
static int initialized = 0;
__declspec( dllexport ) rtwCAPI_ModelMappingInfo *getRootMappingInfo()
{
    if (initialized == 0) {
        initialized = 1;
        PendCart_249987d3_host_InitializeDataMapInfo(&(root), "PendCart_249987d3");
    }
    return &root.mmi;
}

rtwCAPI_ModelMappingInfo *mexFunction(){return(getRootMappingInfo());}
