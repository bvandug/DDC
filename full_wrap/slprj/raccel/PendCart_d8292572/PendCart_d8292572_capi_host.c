#include "PendCart_d8292572_capi_host.h"
static PendCart_d8292572_host_DataMapInfo_T root;
static int initialized = 0;
__declspec( dllexport ) rtwCAPI_ModelMappingInfo *getRootMappingInfo()
{
    if (initialized == 0) {
        initialized = 1;
        PendCart_d8292572_host_InitializeDataMapInfo(&(root), "PendCart_d8292572");
    }
    return &root.mmi;
}

rtwCAPI_ModelMappingInfo *mexFunction(){return(getRootMappingInfo());}
