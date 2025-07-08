#include "PendCart_3ea96591_capi_host.h"
static PendCart_3ea96591_host_DataMapInfo_T root;
static int initialized = 0;
__declspec( dllexport ) rtwCAPI_ModelMappingInfo *getRootMappingInfo()
{
    if (initialized == 0) {
        initialized = 1;
        PendCart_3ea96591_host_InitializeDataMapInfo(&(root), "PendCart_3ea96591");
    }
    return &root.mmi;
}

rtwCAPI_ModelMappingInfo *mexFunction(){return(getRootMappingInfo());}
