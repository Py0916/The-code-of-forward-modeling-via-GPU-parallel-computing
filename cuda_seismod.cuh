
#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 

__global__ void addKernel(int* c, const int* a, const int* b);

__global__ void Kernel_Get_dR_dH_dT(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* R, float* H, float* T, float* Cn, int nx, int nz, int N);

__global__ void Kernel_Get_dU_dV(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* U, float* V, float* Cn, int nx, int nz, int N);

__global__ void Kernel_Get_U_V(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* ddx, float* ddz, float* U_x, float* U_z, float* V_x, float* V_z, float* U, float* V, float* rou, float dx, float dz, float dt, int nx, int nz, int N);

__global__ void  Kernel_Get_R_T_H(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* ddx, float* ddz, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z, float* R, float* T, float* H, float* mu, float* numd, float dx, float dz, float dt, int nx, int nz, int N);


__global__ void Group_Kernel_Get_dR_dH_dT(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* R, float* H, float* T, float* Cn, int nx_real, int nz_real, int N);

__global__ void Group_Kernel_Get_dU_dV(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* U, float* V, float* Cn, int nx_real, int nz_real, int N);

__global__ void Group_Kernel_Get_U_V(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* ddx, float* ddz, float* U_x, float* U_z, float* V_x, float* V_z, float* U, float* V, float* rou, float dx, float dz, float dt, int nx_real, int nz_real, int N);

__global__ void  Group_Kernel_Get_R_T_H(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* ddx, float* ddz, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z, float* R, float* T, float* H, float* mu, float* numd, float dx, float dz, float dt, int nx_real, int nz_real, int N);

__global__ void Group_SetInitialValue(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* U, float* V, float* U_x, float* U_z, float* V_x, float* V_z, float* R, float* T, float* H, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z);

__global__ void GetRSM(float* field, float* mod);