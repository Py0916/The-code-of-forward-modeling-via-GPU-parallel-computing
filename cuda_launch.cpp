// cuda.cpp : ���� DLL Ӧ�ó���ĵ���������
//

#include "cuda_launch.h"
#include<malloc.h>
#include <string>
#include <numeric>
#include <time.h>
//�������  
int _stdcall vectorAdd(int c[], int a[], int b[], int size)
{
	int result = -1;
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// ѡ���������е�GPU  
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		result = 1;
		goto Error;
	}

	// ��GPU��Ϊ����dev_a��dev_b��dev_c�����ڴ�ռ�.  
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		result = 2;
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		result = 3;
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		result = 4;
		goto Error;
	}

	// �������ڴ渴�����ݵ�GPU�ڴ���.  
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		result = 5;
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		result = 6;
		goto Error;
	}

	// ����GPU�ں˺���  
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// ����cudaDeviceSynchronize�ȴ�GPU�ں˺���ִ����ɲ��ҷ����������κδ�����Ϣ  
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		result = 7;
		goto Error;
	}

	// ��GPU�ڴ��и������ݵ������ڴ���  
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		result = 8;
		goto Error;
	}

	result = 0;

	// ����CUDA�豸�����˳�֮ǰ�������cudaDeviceReset  
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		return 9;
	}
Error:
	//�ͷ��豸�б�����ռ�ڴ�  
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return result;

}
int N = 4;
int nx = 256;
int nz = 256;
int pml = 30;
float dx = 0.5f;
float dz = 0.5f;
int n_layer = 3;
float dt = 0.0002f;
int nt = 2048;
int point_layer = 2;
//const int SourceX = N;
//const int SourceZ = N;
const float coefficientR = 0.0001f;
const float PI = 3.1415926f;
int fm = 30;
float t0 = 0.015f;


void SetModPara(int nx_In, int nz_In, int nt_In, int N_In, int pml_In, int n_layer_In, int point_layer_In, int fm_In, float dx_In, float dz_In, float dt_In, float t_delay_In)
{
	nx = nx_In;
	nz = nz_In;
	nt = nt_In;
	N = N_In;
	pml = pml_In;
	n_layer = n_layer_In;
	point_layer = point_layer_In;
	fm = fm_In;
	dx = dx_In;
	dz = dz_In;
	dt = dt_In;
	t0 = t_delay_In;
}

void _stdcall Initial(float* Vp, float* Vs, float* rou, float* mu, float* numd, float* U, float* V, float* U_x, float* U_z, float* V_x, float* V_z, float* R, float* T, float* H, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z, float* ddx, float* ddz, float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* Cn)
{
	//�����ڴ�ռ�
	int size;
	size = nt * nx * nz * sizeof(float);
	//cudaMallocManaged((void**)&(Wavefield), size);
	cudaError cudaE;
	size = nx * nz * sizeof(float);
	//cudaMalloc((void**)(&Vp), size);

	cudaE = cudaMallocManaged((void**)(&Vp), size);
	cudaMallocManaged((void**)(&Vs), size);
	cudaMallocManaged((void**)(&rou), size);
	cudaMallocManaged((void**)(&mu), size);
	cudaMallocManaged((void**)(&numd), size);

	cudaMallocManaged((void**)(&U), size);
	cudaMallocManaged((void**)(&V), size);
	cudaMallocManaged((void**)(&U_x), size);
	cudaMallocManaged((void**)(&U_z), size);
	cudaMallocManaged((void**)(&V_x), size);
	cudaMallocManaged((void**)(&V_z), size);
	cudaMallocManaged((void**)(&T), size);
	cudaMallocManaged((void**)(&H), size);
	cudaMallocManaged((void**)(&R), size);
	cudaMallocManaged((void**)(&T_x), size);
	cudaMallocManaged((void**)(&T_z), size);
	cudaMallocManaged((void**)(&H_x), size);
	cudaMallocManaged((void**)(&H_z), size);
	cudaMallocManaged((void**)(&R_x), size);
	cudaMallocManaged((void**)(&R_z), size);

	cudaMallocManaged((void**)(&ddx), size);
	cudaMallocManaged((void**)(&ddz), size);

	cudaMallocManaged((void**)(&Ux_Rx), size);
	cudaMallocManaged((void**)(&Uz_Tz), size);
	cudaMallocManaged((void**)(&Vx_Hu), size);
	cudaMallocManaged((void**)(&Vz_Hv), size);


	size = N * sizeof(float);
	cudaMallocManaged((void**)(&Cn), size);
	float* CnHost = GetCn(N);
	for (int i = 0; i < N; i++)
	{
		Cn[i] = CnHost[i];
	}
	delete[] CnHost;
}
float* GetCn(int n_C)
{
	//float* Cn = (float*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, n_C * sizeof(float));
	float* Cn = new float[n_C];
	for (int i = 0; i < n_C; i++)
	{
		Cn[i] = (float)pow(-1, i + 2) / (2 * i + 1);//ע��ʽi��1��ʼ���˴�i��0��ʼ
		for (int j = 0; j < n_C; j++)
		{
			if (i != j)
				Cn[i] *= (float)(2 * j + 1) * (2 * j + 1) / abs((2 * i + 1) * (2 * i + 1) - (2 * j + 1) * (2 * j + 1));
		}
	}
	return Cn;
}
void _stdcall SetValue_Zero(float* U, float* V, float* U_x, float* U_z, float* V_x, float* V_z, float* R, float* T, float* H, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z, float* ddx, float* ddz, float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv)
{
	int size = nx * nz;
	for (int i = 0; i < size; i++)
	{
		U[i] = 0;
		V[i] = 0;
		U_x[i] = 0;
		U_z[i] = 0;
		V_x[i] = 0;
		V_z[i] = 0;
		R[i] = 0;
		T[i] = 0;
		H[i] = 0;
		R_x[i] = 0;
		R_z[i] = 0;
		T_x[i] = 0;
		T_z[i] = 0;
		H_x[i] = 0;
		H_z[i] = 0;

		ddx[i] = 0;
		ddz[i] = 0;
		Ux_Rx[i] = 0;
		Uz_Tz[i] = 0;
		Vx_Hu[i] = 0;
		Vz_Hv[i] = 0;
	}
}
float Rickerwavelet(float t)
{
	float wavelet, tmp, A = 100;
	tmp = PI * fm * (t - t0);
	wavelet = A * (float)(exp(-1 * tmp * tmp) * (1 - 2 * tmp * tmp));
	return wavelet;
}
void Group_Set_ddx_ddz(float* Vp, float* ddx, float* ddz, int nx_real, int nz_real, int n_p2)
{
	float pml_width_x = dx * pml;
	float pml_width_z = dz * pml;
	int pml_x = nx_real - pml;
	int pml_z = nz_real - pml;
	float Px, Pz;
	int i, j, m;
	int n_Parallel = n_p2 * n_p2;
	for (m = 0; m < n_Parallel; m++)
	{
		for (i = pml_x - N; i < nx_real - N; i++)
		{
			for (j = N; j < nz_real - N; j++)
			{
				Px = (i - pml_x + N + 1) * dx;
				ddx[m * nx_real * nz_real + i * nz_real + j] = d_pml(Px, Vp[m * nx_real * nz_real + i * nz_real + j], pml_width_x);
			}
		}
		for (i = N; i < nx_real - N; i++)
		{
			for (j = pml_z - N; j < nz_real - N; j++)
			{
				Pz = (j - pml_z + 1 + N) * dz;
				ddz[m * nx_real * nz_real + i * nz_real + j] = d_pml(Pz, Vp[m * nx_real * nz_real + i * nz_real + j], pml_width_z);
			}
		}
	}

}
void Set_ddx_ddz(float* Vp, float* ddx, float* ddz)
{
	float pml_width_x = dx * pml;
	float pml_width_z = dz * pml;
	int pml_x = nx - pml;
	int pml_z = nz - pml;
	float Px, Pz;
	int i, j;
	//�ұ߽�
	for (i = pml_x - N; i < nx - N; i++)
	{
		for (j = N; j < nz - N; j++)
		{
			Px = (i - pml_x + N + 1) * dx;
			ddx[i * nz + j] = d_pml(Px, Vp[i * nz + j], pml_width_x);
		}
	}
	//�±߽�
	for (i = N; i < nx - N; i++)
	{
		for (j = pml_z - N; j < nz - N; j++)
		{
			Pz = (j - pml_z + 1 + N) * dz;
			ddz[i * nz + j] = d_pml(Pz, Vp[i * nz + j], pml_width_z);
		}
	}
}
float d_pml(float x, float Vp, float pml_width)//attenuation˥��,pml�����ձ߽�
{
	float attenuation;
	attenuation = (float)(2 * Vp * log(1 / coefficientR) * pow(x / pml_width, 4) / (pml_width));//Xia����

	return attenuation;
}
bool Alloc_Storage(Para_CUDA& P_CUDA, int nx, int nz, int nt, int N)
{
	int size;

	cudaError cudaStatus;

	size = nx * nz * sizeof(float);
	cudaMalloc((void**)(&P_CUDA.rou), size);
	cudaMalloc((void**)(&P_CUDA.mu), size);
	cudaMalloc((void**)(&P_CUDA.numd), size);

	cudaMalloc((void**)(&P_CUDA.U), size);
	cudaMalloc((void**)(&P_CUDA.V), size);
	cudaMalloc((void**)(&P_CUDA.U_x), size);
	cudaMalloc((void**)(&P_CUDA.U_z), size);
	cudaMalloc((void**)(&P_CUDA.V_x), size);
	cudaMalloc((void**)(&P_CUDA.V_z), size);
	cudaMalloc((void**)(&P_CUDA.T), size);
	cudaMalloc((void**)(&P_CUDA.H), size);
	cudaMalloc((void**)(&P_CUDA.R), size);
	cudaMalloc((void**)(&P_CUDA.T_x), size);
	cudaMalloc((void**)(&P_CUDA.T_z), size);
	cudaMalloc((void**)(&P_CUDA.H_x), size);
	cudaMalloc((void**)(&P_CUDA.H_z), size);
	cudaMalloc((void**)(&P_CUDA.R_x), size);
	cudaMalloc((void**)(&P_CUDA.R_z), size);

	cudaMalloc((void**)(&P_CUDA.ddx), size);
	cudaMalloc((void**)(&P_CUDA.ddz), size);

	cudaMalloc((void**)(&P_CUDA.Ux_Rx), size);
	cudaMalloc((void**)(&P_CUDA.Uz_Tz), size);
	cudaMalloc((void**)(&P_CUDA.Vx_Hu), size);
	cudaMalloc((void**)(&P_CUDA.Vz_Hv), size);

	size = N * sizeof(float);
	cudaMalloc((void**)(&P_CUDA.Cn), size);

	size = nx * nz;

	P_CUDA.U_Host = new float[size];
	P_CUDA.V_Host = new float[size];
	P_CUDA.U_x_Host = new float[size];
	P_CUDA.U_z_Host = new float[size];
	P_CUDA.V_x_Host = new float[size];
	P_CUDA.V_z_Host = new float[size];
	P_CUDA.R_Host = new float[size];
	P_CUDA.T_Host = new float[size];
	P_CUDA.H_Host = new float[size];
	P_CUDA.R_x_Host = new float[size];
	P_CUDA.R_z_Host = new float[size];
	P_CUDA.T_x_Host = new float[size];
	P_CUDA.T_z_Host = new float[size];
	P_CUDA.H_x_Host = new float[size];
	P_CUDA.H_z_Host = new float[size];
	P_CUDA.ddx_Host = new float[size];
	P_CUDA.ddz_Host = new float[size];

	P_CUDA.Ux_Rx_Host = new float[size];
	P_CUDA.Uz_Tz_Host = new float[size];
	P_CUDA.Vx_Hu_Host = new float[size];
	P_CUDA.Vz_Hv_Host = new float[size];

	P_CUDA.SourceVulue_Host = new float[nt];
	return true;
}
bool SetInitialValue(Para_CUDA& P_CUDA, int nx, int nz, int nt, int N)
{
	int size = nx * nz;
	for (int i = 0; i < size; i++)
	{
		P_CUDA.U_Host[i] = 0;
		P_CUDA.V_Host[i] = 0;
		P_CUDA.U_x_Host[i] = 0;
		P_CUDA.U_z_Host[i] = 0;
		P_CUDA.V_x_Host[i] = 0;
		P_CUDA.V_z_Host[i] = 0;
		P_CUDA.R_Host[i] = 0;
		P_CUDA.T_Host[i] = 0;
		P_CUDA.H_Host[i] = 0;
		P_CUDA.R_x_Host[i] = 0;
		P_CUDA.R_z_Host[i] = 0;
		P_CUDA.T_x_Host[i] = 0;
		P_CUDA.T_z_Host[i] = 0;
		P_CUDA.H_x_Host[i] = 0;
		P_CUDA.H_z_Host[i] = 0;
		P_CUDA.Ux_Rx_Host[i] = 0;
		P_CUDA.Uz_Tz_Host[i] = 0;
		P_CUDA.Vx_Hu_Host[i] = 0;
		P_CUDA.Vz_Hv_Host[i] = 0;
	}
	P_CUDA.Cn_Host = GetCn(N);

	for (int i = 0; i < nt; i++)
	{
		P_CUDA.SourceVulue_Host[i] = Rickerwavelet(i * dt + dt);
	}
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(P_CUDA.U, P_CUDA.U_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.V, P_CUDA.V_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.U_x, P_CUDA.U_x_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.U_z, P_CUDA.U_z_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.V_x, P_CUDA.V_x_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.V_z, P_CUDA.V_z_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.T, P_CUDA.T_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.H, P_CUDA.H_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.R, P_CUDA.R_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.T_x, P_CUDA.T_x_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.T_z, P_CUDA.T_z_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.H_x, P_CUDA.H_x_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.H_z, P_CUDA.H_z_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.R_x, P_CUDA.R_x_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.R_z, P_CUDA.R_z_Host, size * sizeof(float), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(P_CUDA.Ux_Rx, P_CUDA.Ux_Rx_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.Uz_Tz, P_CUDA.Uz_Tz_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.Vx_Hu, P_CUDA.Vx_Hu_Host, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.Vz_Hv, P_CUDA.Vz_Hv_Host, size * sizeof(float), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(P_CUDA.Cn, P_CUDA.Cn_Host, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		return false;
	}
	return true;
}
void FreeCudaPara(Para_CUDA& P_CUDA)
{
	cudaFree(P_CUDA.rou);
	cudaFree(P_CUDA.mu);
	cudaFree(P_CUDA.numd);

	cudaFree(P_CUDA.U);
	cudaFree(P_CUDA.V);
	cudaFree(P_CUDA.U_x);
	cudaFree(P_CUDA.U_z);
	cudaFree(P_CUDA.V_x);
	cudaFree(P_CUDA.V_z);
	cudaFree(P_CUDA.T);
	cudaFree(P_CUDA.H);
	cudaFree(P_CUDA.R);
	cudaFree(P_CUDA.T_x);
	cudaFree(P_CUDA.T_z);
	cudaFree(P_CUDA.H_x);
	cudaFree(P_CUDA.H_z);
	cudaFree(P_CUDA.R_x);
	cudaFree(P_CUDA.R_z);
	cudaFree(P_CUDA.ddx);
	cudaFree(P_CUDA.ddz);
	cudaFree(P_CUDA.Ux_Rx);
	cudaFree(P_CUDA.Uz_Tz);
	cudaFree(P_CUDA.Vx_Hu);
	cudaFree(P_CUDA.Vz_Hv);

	cudaFree(P_CUDA.Cn);
	delete[] P_CUDA.ddx_Host;
	delete[] P_CUDA.ddz_Host;
	delete[] P_CUDA.Cn_Host;
	delete[] P_CUDA.SourceVulue_Host;
	P_CUDA.ddx_Host = NULL;
	P_CUDA.ddz_Host = NULL;
	P_CUDA.Cn_Host = NULL;
	P_CUDA.SourceVulue_Host = NULL;
}
int _stdcall SeisMod_CUDA(float* Vp_In, float* rou_In, float* mu_In, float* numd_In, float* Wavefield_In)
{
	Para_CUDA P_CUDA;
	cudaError cudaStatus;
	int result = -1;
	int size, i, j, k;
	int threads = 256;
	int blocks = nx * nz / threads;
	Alloc_Storage(P_CUDA, nx, nz, nt, N);
	SetInitialValue(P_CUDA, nx, nz, nt, N);

	size = nx * nz;
	for (i = 0; i < size; i++)
	{
		P_CUDA.ddx_Host[i] = 0;
		P_CUDA.ddz_Host[i] = 0;
		P_CUDA.V_Host[i] = 0;
		P_CUDA.T_Host[i] = 0;
		P_CUDA.H_Host[i] = 0;
	}


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		result = -1;
		FreeCudaPara(P_CUDA);
		return result;
	}



	//���ɱ߽�1��numd=0��rou=0.5*rou
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j <= N; j++)
		{
			numd_In[i * nz + j] = 0;
			mu_In[i * nz + j] /= 2;
			rou_In[i * nz + j] /= 2;
		}
	}
	size = nx * nz * sizeof(float);
	cudaStatus = cudaMemcpy(P_CUDA.rou, rou_In, size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.mu, mu_In, size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.numd, numd_In, size, cudaMemcpyHostToDevice);

	Set_ddx_ddz(Vp_In, P_CUDA.ddx_Host, P_CUDA.ddz_Host);//Get ddx_Host
	cudaStatus = cudaMemcpy(P_CUDA.ddx, P_CUDA.ddx_Host, size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(P_CUDA.ddz, P_CUDA.ddz_Host, size, cudaMemcpyHostToDevice);

	for (k = 0; k < nt; k++)
	{
		if (k == 0)
		{
			Group_SetInitialValue << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.U, P_CUDA.V, P_CUDA.U_x, P_CUDA.U_z, P_CUDA.V_x, P_CUDA.V_z, P_CUDA.R, P_CUDA.T, P_CUDA.H, P_CUDA.R_x, P_CUDA.R_z, P_CUDA.T_x, P_CUDA.T_z, P_CUDA.H_x, P_CUDA.H_z);
			cudaStatus = cudaDeviceSynchronize();
		}
		else
		{
		cudaStatus = cudaMemcpy(P_CUDA.T_Host, P_CUDA.T, size, cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(P_CUDA.H_Host, P_CUDA.H, size, cudaMemcpyDeviceToHost);
		for (i = 0; i < nx; i++)
		{
			P_CUDA.T_Host[i * nz + N] = 0;
			P_CUDA.H_Host[i * nz + N] = 0;
		}
		cudaStatus = cudaMemcpy(P_CUDA.T, P_CUDA.T_Host, size, cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(P_CUDA.H, P_CUDA.H_Host, size, cudaMemcpyHostToDevice);
	}
		Kernel_Get_dR_dH_dT << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.R, P_CUDA.H, P_CUDA.T, P_CUDA.Cn, nx, nz, N);//����дGPU���м��㣬��ȡUx_Rx,  Uz_Tz,  Vx_Hu,  Vz_Hv
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			result = -2;
			FreeCudaPara(P_CUDA);
			return result;
		}
		Kernel_Get_U_V << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.ddx, P_CUDA.ddz, P_CUDA.U_x, P_CUDA.U_z, P_CUDA.V_x, P_CUDA.V_z, P_CUDA.U, P_CUDA.V, P_CUDA.rou, dx, dz, dt, nx, nz, N);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			result = -3;
			FreeCudaPara(P_CUDA);
			return result;
		}
		cudaStatus = cudaMemcpy(P_CUDA.V_Host, P_CUDA.V, size, cudaMemcpyDeviceToHost);
		P_CUDA.V_Host[N * nz + N] = P_CUDA.SourceVulue_Host[k];//������Դ����Դ��Initial()�������Ѿ���ֵ�ˡ�
		cudaStatus = cudaMemcpy(P_CUDA.V, P_CUDA.V_Host, size, cudaMemcpyHostToDevice);

		Kernel_Get_dU_dV << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.U, P_CUDA.V, P_CUDA.Cn, nx, nz, N);//����дGPU���м��㣬��ȡUx_Rx,  Uz_Tz,  Vx_Hu,  Vz_Hv
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			result = -4;
			FreeCudaPara(P_CUDA);
			return result;
		}
		Kernel_Get_R_T_H << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.ddx, P_CUDA.ddz, P_CUDA.R_x, P_CUDA.R_z, P_CUDA.T_x, P_CUDA.T_z, P_CUDA.H_x, P_CUDA.H_z, P_CUDA.R, P_CUDA.T, P_CUDA.H, P_CUDA.mu, P_CUDA.numd, dx, dz, dt, nx, nz, N);//GPU���м��㣬ͨ��Rx,Hu,Hv,Tz��U_x,U_z,V_x,V_z������º��U_x,U_z,V_x,V_z,U,V
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			result = -5;
			FreeCudaPara(P_CUDA);
			return result;
		}
		memcpy(Wavefield_In + size / sizeof(float) * k, P_CUDA.V_Host, size);
	}
	FreeCudaPara(P_CUDA);
	return result;
}
int _stdcall SeisMod_Group_CUDA(float* Vp_In, float* rou_In, float* mu_In, float* numd_In, int n_Parallel, int n_Group, float* fieldRecord, int n_Trace, float Offset, float Step, float* f_Cost)
{
	clock_t t1, t2;
	t1 = clock();
	int ii, n_PosRecord = n_Parallel * n_Trace;
	int size_RecordOnce = n_Trace * nt;
	float* ModRecord = new float[n_Group * size_RecordOnce];//ÿ�����ݵ����¼�Ĵ�С*n_Parallel
	float* ModRecordOnce = new float[size_RecordOnce];
	int* PosRecord = new int[n_PosRecord];
	float* fieldRecord_device, * ModRecord_device;
	cudaMalloc((void**)(&fieldRecord_device), size_RecordOnce * sizeof(float));
	cudaMalloc((void**)(&ModRecord_device), size_RecordOnce * sizeof(float));

	cudaMemcpy(fieldRecord_device, fieldRecord, size_RecordOnce * sizeof(float), cudaMemcpyHostToDevice);
	int n_p2 = (int)pow(n_Parallel, 0.5);
	for (int m = 0; m < n_Parallel; m++)
	{
		for (int k = 0; k < n_Trace; k++)
		{
			ii = (int)(N + (Offset + Step * k) / dx);
			PosRecord[m * n_Trace + k] = m * nx * nz / n_Parallel + ii * nz / n_p2 + N;
		}
	}

	Para_CUDA P_CUDA;
	cudaError cudaStatus;
	int result = -1;
	int size, i, j, k, m;
	int threads = 256;
	int blocks = nx * nz / threads;
	Alloc_Storage(P_CUDA, nx, nz, nt, N);
	P_CUDA.Cn_Host = GetCn(N);
	cudaStatus = cudaMemcpy(P_CUDA.Cn, P_CUDA.Cn_Host, N * sizeof(float), cudaMemcpyHostToDevice);
	for (int i = 0; i < nt; i++)
	{
		P_CUDA.SourceVulue_Host[i] = Rickerwavelet(i * dt + dt);
	}
	size = nx * nz;
	for (i = 0; i < size; i++)
	{
		P_CUDA.ddx_Host[i] = 0;
		P_CUDA.ddz_Host[i] = 0;
		P_CUDA.V_Host[i] = 0;
		P_CUDA.T_Host[i] = 0;
		P_CUDA.H_Host[i] = 0;
	}

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		result = -1;
		FreeCudaPara(P_CUDA);
		return result;
	}
	size = nx * nz;
	float* Vp_Once = new float[size];
	float* rou_Once = new float[size];
	float* mu_Once = new float[size];
	float* numd_Once = new float[size];
	int g, size_Group;
	size_Group = n_Group / n_Parallel;//����Ⱥ��128��ÿ�β���4*4����Ҫ8��
	int nx_real, nz_real, size_real;
	nx_real = nx / (int)pow(n_Parallel, 0.5);
	nz_real = nz / (int)pow(n_Parallel, 0.5);
	size_real = nx * nz / n_Parallel;//��ʵģ�͵Ĵ�С,�˴���nx*nz/16


	t2 = clock();
	printf("nx*nz*nt:256*256*2048,GPU:%f seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
	for (g = 0; g < size_Group; g++)
	{
		std::copy(Vp_In + g * size, Vp_In + (g + 1) * size + 1, Vp_Once);
		std::copy(rou_In + g * size, rou_In + (g + 1) * size + 1, rou_Once);
		std::copy(mu_In + g * size, mu_In + (g + 1) * size + 1, mu_Once);
		std::copy(numd_In + g * size, numd_In + (g + 1) * size + 1, numd_Once);

		for (m = 0; m < n_Parallel; m++)
		{
			for (i = 0; i < nx_real; i++)
			{
				for (j = 0; j <= N; j++)
				{
					numd_Once[m * size_real + i * nz_real + j] = 0;
					mu_Once[m * size_real + i * nz_real + j] /= 2;
					rou_Once[m * size_real + i * nz_real + j] /= 2;
				}
			}
		}

		cudaStatus = cudaMemcpy(P_CUDA.rou, rou_Once, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(P_CUDA.mu, mu_Once, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(P_CUDA.numd, numd_Once, size * sizeof(float), cudaMemcpyHostToDevice);

		Group_Set_ddx_ddz(Vp_Once, P_CUDA.ddx_Host, P_CUDA.ddz_Host, nx_real, nz_real, n_p2);//Get ddx_Host
		cudaStatus = cudaMemcpy(P_CUDA.ddx, P_CUDA.ddx_Host, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(P_CUDA.ddz, P_CUDA.ddz_Host, size * sizeof(float), cudaMemcpyHostToDevice);


		//��ʼʱ��ѭ��
		for (k = 0; k < nt; k++)
		{
			if (k == 0)
			{
				Group_SetInitialValue << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.U, P_CUDA.V, P_CUDA.U_x, P_CUDA.U_z, P_CUDA.V_x, P_CUDA.V_z, P_CUDA.R, P_CUDA.T, P_CUDA.H, P_CUDA.R_x, P_CUDA.R_z, P_CUDA.T_x, P_CUDA.T_z, P_CUDA.H_x, P_CUDA.H_z);
				cudaStatus = cudaDeviceSynchronize();
			}
			else
			{
				cudaStatus = cudaMemcpy(P_CUDA.T_Host, P_CUDA.T, size * sizeof(float), cudaMemcpyDeviceToHost);
				cudaStatus = cudaMemcpy(P_CUDA.H_Host, P_CUDA.H, size * sizeof(float), cudaMemcpyDeviceToHost);
				for (m = 0; m < n_Parallel; m++)
				{
					for (i = 0; i < nx_real; i++)
					{
						P_CUDA.T_Host[m * size_real + i * nz_real + N] = 0;
						P_CUDA.H_Host[m * size_real + i * nz_real + N] = 0;
					}
				}
				cudaStatus = cudaMemcpy(P_CUDA.T, P_CUDA.T_Host, size * sizeof(float), cudaMemcpyHostToDevice);
				cudaStatus = cudaMemcpy(P_CUDA.H, P_CUDA.H_Host, size * sizeof(float), cudaMemcpyHostToDevice);
			}

			Group_Kernel_Get_dR_dH_dT << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.R, P_CUDA.H, P_CUDA.T, P_CUDA.Cn, nx_real, nz_real, N);//����дGPU���м��㣬��ȡUx_Rx,  Uz_Tz,  Vx_Hu,  Vz_Hv
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				result = -2;
				FreeCudaPara(P_CUDA);
				return result;
			}
			Group_Kernel_Get_U_V << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.ddx, P_CUDA.ddz, P_CUDA.U_x, P_CUDA.U_z, P_CUDA.V_x, P_CUDA.V_z, P_CUDA.U, P_CUDA.V, P_CUDA.rou, dx, dz, dt, nx_real, nz_real, N);//GPU���м��㣬ͨ��Rx,Hu,Hv,Tz��U_x,U_z,V_x,V_z������º��U_x,U_z,V_x,V_z,U,V
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				result = -3;
				FreeCudaPara(P_CUDA);
				return result;
			}

			cudaStatus = cudaMemcpy(P_CUDA.V_Host, P_CUDA.V, size * sizeof(float), cudaMemcpyDeviceToHost);
			for (m = 0; m < n_Parallel; m++)
			{
				P_CUDA.V_Host[m * size_real + N * nz_real + N] = P_CUDA.SourceVulue_Host[k];//������Դ����Դ��Initial()�������Ѿ���ֵ�ˡ�
			}
			cudaStatus = cudaMemcpy(P_CUDA.V, P_CUDA.V_Host, size * sizeof(float), cudaMemcpyHostToDevice);

			Group_Kernel_Get_dU_dV << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.U, P_CUDA.V, P_CUDA.Cn, nx_real, nz_real, N);//����дGPU���м��㣬��ȡUx_Rx,  Uz_Tz,  Vx_Hu,  Vz_Hv
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				result = -4;
				FreeCudaPara(P_CUDA);
				return result;
			}
			Group_Kernel_Get_R_T_H << <blocks, threads >> > (P_CUDA.Ux_Rx, P_CUDA.Uz_Tz, P_CUDA.Vx_Hu, P_CUDA.Vz_Hv, P_CUDA.ddx, P_CUDA.ddz, P_CUDA.R_x, P_CUDA.R_z, P_CUDA.T_x, P_CUDA.T_z, P_CUDA.H_x, P_CUDA.H_z, P_CUDA.R, P_CUDA.T, P_CUDA.H, P_CUDA.mu, P_CUDA.numd, dx, dz, dt, nx_real, nz_real, N);//GPU���м��㣬ͨ��Rx,Hu,Hv,Tz��U_x,U_z,V_x,V_z������º��U_x,U_z,V_x,V_z,U,V
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				result = -5;
				FreeCudaPara(P_CUDA);
				return result;
			}
			for (m = 0; m < n_Parallel; m++)
			{
				for (j = 0; j < n_Trace; j++)
				{
					ModRecord[(n_Parallel * g + m) * size_RecordOnce + j * nt + k] = P_CUDA.V_Host[PosRecord[m * n_Trace + j]];
				}
			}
		}

	}
	float tt1, tt2, tt3;
	for (i = 0; i < n_Group; i++)
	{

		for (j = 0; j < n_Trace; j++)
		{
			tt1 = *std::max_element(ModRecord + size_RecordOnce * i + nt * j, ModRecord + size_RecordOnce * i + nt * j + nt);
			tt2 = -*std::min_element(ModRecord + size_RecordOnce * i + nt * j, ModRecord + size_RecordOnce * i + nt * j + nt);
			tt3 = std::max(tt1, tt2);
			for (k = 0; k < nt; k++)
			{
				ModRecord[size_RecordOnce * i + nt * j + k] /= tt3;
			}
		}
	}
	threads = 256;
	blocks = nt / threads * n_Trace;
	for (i = 0; i < n_Group; i++)
	{
		memcpy(ModRecordOnce, ModRecord + size_RecordOnce * i, size_RecordOnce * sizeof(float));
		cudaMemcpy(ModRecord_device, ModRecordOnce, size_RecordOnce * sizeof(float), cudaMemcpyHostToDevice);

		GetRSM << <blocks, threads >> > (fieldRecord_device, ModRecord_device);
		cudaStatus = cudaDeviceSynchronize();
		cudaMemcpy(ModRecordOnce, ModRecord_device, size_RecordOnce * sizeof(float), cudaMemcpyDeviceToHost);//ModRecordOnce�ﱣ����ǲ�ֵ��ƽ��
		f_Cost[i] = std::accumulate(ModRecordOnce, ModRecordOnce + size_RecordOnce, 0.0f);//���
		f_Cost[i] = powf((f_Cost[i] / size_RecordOnce), 0.5f);
	}
	printf("nx*nz*nt:256*256*2048,GPU:%f seconds\n", (double)(clock() - t2) / CLOCKS_PER_SEC);
	t2 = clock();
	delete[] ModRecord;
	delete[] ModRecordOnce;
	delete[] PosRecord;
	cudaFree(fieldRecord_device);
	cudaFree(ModRecord_device);

	delete[] Vp_Once;
	delete[] rou_Once;
	delete[] mu_Once;
	delete[] numd_Once;
	FreeCudaPara(P_CUDA);
	printf("nx*nz*nt:256*256*2048,GPU:%f seconds\n", (double)(clock() - t2) / CLOCKS_PER_SEC);
	return result;
}
int _stdcall SeisMod(float* Vp_In, float* Vs_In, float* rou_In, float* mu_In, float* numd_In, float* Wavefield_In)
{
	int result = -1;

	float* Vp, * Vs, * rou, * mu, * numd;
	float* U, * V, * U_x, * U_z, * V_x, * V_z, * R, * H, * T, * R_x, * R_z, * T_x, * T_z, * H_x, * H_z;
	float* ddx, * ddz;
	float* Ux_Rx, * Uz_Tz, * Vx_Hu, * Vz_Hv;
	float* Cn;
	float* SourceVulue_Host;

	int size, i, j, k;

	cudaError cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		result = 1;
		goto Error;
	}
	size = nx * nz * sizeof(float);
	cudaStatus = cudaMallocManaged((void**)(&Vp), size);
	cudaMallocManaged((void**)(&Vs), size);
	cudaMallocManaged((void**)(&rou), size);
	cudaMallocManaged((void**)(&mu), size);
	cudaMallocManaged((void**)(&numd), size);

	cudaMallocManaged((void**)(&U), size);
	cudaMallocManaged((void**)(&V), size);
	cudaMallocManaged((void**)(&U_x), size);
	cudaMallocManaged((void**)(&U_z), size);
	cudaMallocManaged((void**)(&V_x), size);
	cudaMallocManaged((void**)(&V_z), size);
	cudaMallocManaged((void**)(&T), size);
	cudaMallocManaged((void**)(&H), size);
	cudaMallocManaged((void**)(&R), size);
	cudaMallocManaged((void**)(&T_x), size);
	cudaMallocManaged((void**)(&T_z), size);
	cudaMallocManaged((void**)(&H_x), size);
	cudaMallocManaged((void**)(&H_z), size);
	cudaMallocManaged((void**)(&R_x), size);
	cudaMallocManaged((void**)(&R_z), size);

	cudaMallocManaged((void**)(&ddx), size);
	cudaMallocManaged((void**)(&ddz), size);

	cudaMallocManaged((void**)(&Ux_Rx), size);
	cudaMallocManaged((void**)(&Uz_Tz), size);
	cudaMallocManaged((void**)(&Vx_Hu), size);
	cudaMallocManaged((void**)(&Vz_Hv), size);

	cudaMallocManaged((void**)(&Cn), N * sizeof(float));
	Cn = GetCn(N);

	SetValue_Zero(U, V, U_x, U_z, V_x, V_z, R, T, H, R_x, R_z, T_x, T_z, H_x, H_z, ddx, ddz, Ux_Rx, Uz_Tz, Vx_Hu, Vz_Hv);

	SourceVulue_Host = new float[nt];
	for (i = 0; i < nt; i++)
	{
		SourceVulue_Host[i] = Rickerwavelet(i * dt);
	}

	size = nx * nz * sizeof(float);

	memcpy(Vp, Vp_In, size);
	memcpy(Vs, Vs_In, size);
	memcpy(rou, rou_In, size);
	memcpy(mu, mu_In, size);
	memcpy(numd, numd_In, size);

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < N; j++)
		{
			numd[i * nz + j] = 0;
			rou[i * nz + j] /= 2;
		}
	}
	for (k = 0; k < nt; k++)
	{
		for (i = 0; i < nx; i++)
		{
			T[i * nz + N] = 0;
		}
		Kernel_Get_dR_dH_dT << <nx, nz >> > (Ux_Rx, Uz_Tz, Vx_Hu, Vz_Hv, R, H, T, Cn, nx, nz, N);//����дGPU���м��㣬��ȡUx_Rx,  Uz_Tz,  Vx_Hu,  Vz_Hv
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			result = 1;
			goto Error;
		}
		Kernel_Get_U_V << <nx, nz >> > (Ux_Rx, Uz_Tz, Vx_Hu, Vz_Hv, ddx, ddz, U_x, U_z, V_x, V_z, U, V, rou, dx, dz, dt, nx, nz, N);//GPU���м��㣬ͨ��Rx,Hu,Hv,Tz��U_x,U_z,V_x,V_z������º��U_x,U_z,V_x,V_z,U,V
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			result = 2;
			goto Error;
		}
		V[nx * nz / 2 + N] = SourceVulue_Host[k];
		Kernel_Get_dU_dV << <nx, nz >> > (Ux_Rx, Uz_Tz, Vx_Hu, Vz_Hv, U, V, Cn, nx, nz, N);//����дGPU���м��㣬��ȡUx_Rx,  Uz_Tz,  Vx_Hu,  Vz_Hv
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			result = 3;
			goto Error;
		}
		Kernel_Get_R_T_H << <nx, nz >> > (Ux_Rx, Uz_Tz, Vx_Hu, Vz_Hv, ddx, ddz, R_x, R_z, T_x, T_z, H_x, H_z, R, T, H, mu, numd, dx, dz, dt, nx, nz, N);//GPU���м��㣬ͨ��Rx,Hu,Hv,Tz��U_x,U_z,V_x,V_z������º��U_x,U_z,V_x,V_z,U,V
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			result = 4;
			goto Error;
		}
		memcpy(Wavefield_In + size / 4 * k, V, size);
	}

Error:
	cudaFree(Vp);
	cudaFree(Vs);
	cudaFree(rou);
	cudaFree(mu);
	cudaFree(numd);

	cudaFree(U);
	cudaFree(V);
	cudaFree(U_x);
	cudaFree(U_z);
	cudaFree(V_x);
	cudaFree(V_z);
	cudaFree(R);
	cudaFree(T);
	cudaFree(H);
	cudaFree(R_x);
	cudaFree(R_z);
	cudaFree(T_x);
	cudaFree(T_z);
	cudaFree(H_x);
	cudaFree(H_z);

	cudaFree(ddx);
	cudaFree(ddz);

	cudaFree(Ux_Rx);
	cudaFree(Uz_Tz);
	cudaFree(Vx_Hu);
	cudaFree(Vz_Hv);

	cudaFree(Cn);
	delete[] SourceVulue_Host;
	SourceVulue_Host = NULL;
	return result;
}


bool _stdcall Mod_To_Grid(float* Vp, float* Vs, float* rou, float* h, int nx, int nz, int n_layer, int point_layer, float dx, float dz, float* Vp_Out, float* Vs_Out, float* rou_Out, float* mu_Out, float* numd_Out)
{
	int i, j, k, index, m;
	float temp;
	//float* ArrayTmp = (float*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, point_layer * sizeof(float));
	float* ArrayTmp = new float[point_layer];
	if (n_layer == 1)
	{
		for (i = 0; i < nx; i++)
		{
			for (j = 0; j < nz; j++)
			{
				Vp_Out[i * nz + j] = Vp[0];
				Vs_Out[i * nz + j] = Vs[0];
				rou_Out[i * nz + j] = rou[0];
			}
		}
	}
	else
	{

		for (i = 0; i < nx; i++)
		{
			for (j = 0; j < nz; j++)
			{
				bool Islastlayer = true;
				for (k = 0; k < n_layer - 1; k++)
				{
					for (m = 0; m < point_layer; m++)
					{
						ArrayTmp[m] = h[k * 2 * point_layer + m];
					}
					index = SearchIndex(i * dx, ArrayTmp, point_layer);
					temp = (h[k * 2 * point_layer + point_layer + index] - h[k * 2 * point_layer + point_layer + index - 1]) / (h[k * 2 * point_layer + index] - h[k * 2 * point_layer + point_layer + index - 1]) * (i * dx - h[k * 2 * point_layer + point_layer + index - 1]) + h[k * 2 * point_layer + point_layer + index - 1];
					if ((float)(j * dz) <= temp)
					{
						Vp_Out[i * nz + j] = Vp[k];
						Vs_Out[i * nz + j] = Vs[k];
						rou_Out[i * nz + j] = rou[k];
						Islastlayer = false;
						break;
					}
				}
				if (Islastlayer)
				{
					Vp_Out[i * nz + j] = Vp[n_layer - 1];
					Vs_Out[i * nz + j] = Vs[n_layer - 1];
					rou_Out[i * nz + j] = rou[n_layer - 1];
				}
			}
		}
	}

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < nz; j++)
		{
			mu_Out[i * nz + j] = rou_Out[i * nz + j] * (float)(pow(Vs_Out[i * nz + j], 2)); 
			numd_Out[i * nz + j] = rou_Out[i * nz + j] * (float)(pow(Vp_Out[i * nz + j], 2) - 2.0 * pow(Vs_Out[i * nz + j], 2));
		}
	}
	delete[] ArrayTmp;
	ArrayTmp = NULL;
	return true;
}
int SearchIndex(float f, float*& Array, int point_layer)
{
	int M = point_layer;
	if (f <= Array[0])
	{
		return 1;
	}
	else
	{
		for (int i = 1; i < M; i++)
		{
			if (f <= Array[i] && f > Array[i - 1])
				return i;
		}
	}
	return -1;
}