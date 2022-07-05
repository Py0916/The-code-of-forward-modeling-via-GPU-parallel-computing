//#pragma once
#include<string.h>
#include "cuda_seismod.cuh"
#include <math.h>
#include <algorithm>

extern "C" __declspec(dllexport)
int _stdcall vectorAdd(int c[], int a[], int b[], int size);

extern "C" __declspec(dllexport)
int _stdcall SeisMod(float* Vp_In, float* Vs_In, float* rou_In, float* mu_In, float* numd_In, float* Wavefield_In);

extern "C" __declspec(dllexport)
int _stdcall SeisMod_CUDA(float* Vp_In, float* rou_In, float* mu_In, float* numd_In, float* Wavefield_In);

extern "C" __declspec(dllexport)
int _stdcall SeisMod_Group_CUDA(float* Vp_In, float* rou_In, float* mu_In, float* numd_In, int n_Parallel, int n_Group, float* fieldRecord, int n_Trace, float Offset, float Step, float* f_Cost);

extern "C" __declspec(dllexport)
bool _stdcall Mod_To_Grid(float* Vp, float* Vs, float* rou, float* h, int nx, int nz, int n_layer, int point_layer, float dx, float dz, float* Vp_Out, float* Vs_Out, float* rou_Out, float* mu_Out, float* numd_Out);

extern "C" __declspec(dllexport)
void _stdcall SetModPara(int nx_In, int nz_In, int nt_In, int N_In, int pml_In, int n_layer_In, int point_layer_In, int fm_In, float dx_In, float dz_In, float dt_In, float t_delay_In);


void _stdcall Initial(float* Vp, float* Vs, float* rou, float* mu, float* numd, float* U, float* V, float* U_x, float* U_z, float* V_x, float* V_z, float* R, float* T, float* H, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z, float* ddx, float* ddz, float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* Cn);

float* GetCn(int n_C);
void _stdcall SetValue_Zero(float* U, float* V, float* U_x, float* U_z, float* V_x, float* V_z, float* R, float* T, float* H, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z, float* ddx, float* ddz, float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv);
float Rickerwavelet(float t);
void Group_Set_ddx_ddz(float* Vp, float* ddx, float* ddz, int nx_real, int nz_real, int n_p2);
void _stdcall Set_ddx_ddz(float* Vp, float* ddx, float* ddz);
float d_pml(float x, float Vp, float pml_width);
int SearchIndex(float f, float*& Array, int point_layer);

class  Para_CUDA
{
public:
	//device
	float* rou, * mu, * numd;//* Vp, * Vs,不参与device端运算，删除了
	float* U, * V, * U_x, * U_z, * V_x, * V_z, * R, * H, * T, * R_x, * R_z, * T_x, * T_z, * H_x, * H_z;
	float* ddx, * ddz;
	float* Ux_Rx, * Uz_Tz, * Vx_Hu, * Vz_Hv;
	float* Cn;
	//Host
	float* Vp_Host, * Vs_Host, * rou_Host, * mu_Host, * numd_Host;
	float* ddx_Host, * ddz_Host;
	float* U_Host, * V_Host, * U_x_Host, * U_z_Host, * V_x_Host, * V_z_Host, * R_Host, * H_Host, * T_Host, * R_x_Host, * R_z_Host, * T_x_Host, * T_z_Host, * H_x_Host, * H_z_Host;	
	float* Ux_Rx_Host, * Uz_Tz_Host, * Vx_Hu_Host, * Vz_Hv_Host;
	float* Cn_Host;
	float* SourceVulue_Host;
};

bool Alloc_Storage(Para_CUDA& P_CUDA, int nx, int nz, int nt, int N);
void FreeCudaPara(Para_CUDA& P_CUDA);

