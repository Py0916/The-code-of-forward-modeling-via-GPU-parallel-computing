#include "cuda_seismod.cuh"
__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void Kernel_Get_dR_dH_dT(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* R, float* H, float* T, float* Cn, int nx, int nz, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Ux_Rx[i] = 0;
	Vx_Hu[i] = 0;
	Vz_Hv[i] = 0;
	Uz_Tz[i] = 0;
	int m = i / nz; 
	int n = i % nz; 
	if (m >= N && m < nx - N && n >= N && n < nz - N)
	{
		for (int ii = 0; ii < N; ii++)
		{
			Ux_Rx[i] += Cn[ii] * (R[i + (ii + 1) * nz] - R[i - ii * nz]);
			Vx_Hu[i] += Cn[ii] * (H[i + ii * nz] - H[i - (ii + 1) * nz]);
			Vz_Hv[i] += Cn[ii] * (T[i + ii + 1] - T[i - ii]);
			Uz_Tz[i] += Cn[ii] * (H[i + ii] - H[i - ii - 1]);
		}
	}
}
__global__ void Kernel_Get_dU_dV(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* U, float* V, float* Cn, int nx, int nz, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Ux_Rx[i] = 0;
	Vx_Hu[i] = 0;
	Vz_Hv[i] = 0;
	Uz_Tz[i] = 0;
	int m = i / nz; 
	int n = i % nz; 
	if (m >= N && m < nx - N && n >= N && n < nz - N)
	{
		for (int ii = 0; ii < N; ii++)
		{
			Ux_Rx[i] += Cn[ii] * (U[i + ii * nz] - U[i - (ii + 1) * nz]);
			Vx_Hu[i] += Cn[ii] * (V[i + (ii + 1) * nz] - V[i - ii * nz]);
			Vz_Hv[i] += Cn[ii] * (V[i + ii] - V[i - ii - 1]);
			Uz_Tz[i] += Cn[ii] * (U[i + ii + 1] - U[i - ii]);
		}
	}
}

__global__ void Kernel_Get_U_V(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* ddx, float* ddz, float* U_x, float* U_z, float* V_x, float* V_z, float* U, float* V, float* rou, float dx, float dz, float dt, int nx, int nz, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int m = i / nz; 
	int n = i % nz; 
	if (m >= N && m < nx - N && n >= N && n < nz - N)
	{
		U_x[i] = U_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt / (rou[i] * dx * (2 + dt * ddx[i])) * Ux_Rx[i];
		U_z[i] = U_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt / (rou[i] * dz * (2 + dt * ddz[i])) * Uz_Tz[i];

		V_x[i] = V_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt / (rou[i] * dx * (2 + dt * ddx[i])) * Vx_Hu[i];
		V_z[i] = V_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt / (rou[i] * dz * (2 + dt * ddz[i])) * Vz_Hv[i];

		U[i] = U_x[i] + U_z[i];
		V[i] = V_x[i] + V_z[i];
	}
}

__global__ void Kernel_Get_R_T_H(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* ddx, float* ddz, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z, float* R, float* T, float* H, float* mu, float* numd, float dx, float dz, float dt, int nx, int nz, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int m = i / nz; 
	int n = i % nz; 
	if (m >= N && m < nx - N && n >= N && n < nz - N)
	{
		R_x[i] = R_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt * (numd[i] + 2 * mu[i]) / ((2 + dt * ddx[i]) * dx) * Ux_Rx[i];
		R_z[i] = R_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt * numd[i] / ((2 + dt * ddz[i]) * dz) * Vz_Hv[i];
		T_x[i] = T_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt * numd[i] / ((2 + dt * ddx[i]) * dx) * Ux_Rx[i];
		T_z[i] = T_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt * (numd[i] + 2 * mu[i]) / ((2 + dt * ddz[i]) * dz) * Vz_Hv[i];
		H_x[i] = H_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt * mu[i] / ((2 + dt * ddx[i]) * dx) * Vx_Hu[i];
		H_z[i] = H_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt * mu[i] / ((2 + dt * ddz[i]) * dz) * Uz_Tz[i];


		R[i] = R_x[i] + R_z[i];
		T[i] = T_x[i] + T_z[i];
		H[i] = H_x[i] + H_z[i];
	}
}

__global__ void Group_Kernel_Get_dR_dH_dT(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* R, float* H, float* T, float* Cn, int nx_real, int nz_real, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Ux_Rx[i] = 0;
	Vx_Hu[i] = 0;
	Vz_Hv[i] = 0;
	Uz_Tz[i] = 0;

	int m = (i % (nx_real * nz_real)) / nz_real; 
	int n = (i % (nx_real * nz_real)) % nz_real; 
	if (m >= N && m < nx_real - N && n >= N && n < nz_real - N)
	{
		for (int ii = 0; ii < N; ii++)
		{
			Ux_Rx[i] += Cn[ii] * (R[i + (ii + 1) * nz_real] - R[i - ii * nz_real]);
			Vx_Hu[i] += Cn[ii] * (H[i + ii * nz_real] - H[i - (ii + 1) * nz_real]);
			Vz_Hv[i] += Cn[ii] * (T[i + ii + 1] - T[i - ii]);
			Uz_Tz[i] += Cn[ii] * (H[i + ii] - H[i - ii - 1]);
		}
	}
}
__global__ void Group_Kernel_Get_dU_dV(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* U, float* V, float* Cn, int nx_real, int nz_real, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Ux_Rx[i] = 0;
	Vx_Hu[i] = 0;
	Vz_Hv[i] = 0;
	Uz_Tz[i] = 0;
	int m = (i % (nx_real * nz_real)) / nz_real; 
	int n = (i % (nx_real * nz_real)) % nz_real; 
	if (m >= N && m < nx_real - N && n >= N && n < nz_real - N)
	{
		for (int ii = 0; ii < N; ii++)
		{
			Ux_Rx[i] += Cn[ii] * (U[i + ii * nz_real] - U[i - (ii + 1) * nz_real]);
			Vx_Hu[i] += Cn[ii] * (V[i + (ii + 1) * nz_real] - V[i - ii * nz_real]);
			Vz_Hv[i] += Cn[ii] * (V[i + ii] - V[i - ii - 1]);
			Uz_Tz[i] += Cn[ii] * (U[i + ii + 1] - U[i - ii]);
		}
	}
}

__global__ void Group_Kernel_Get_U_V(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* ddx, float* ddz, float* U_x, float* U_z, float* V_x, float* V_z, float* U, float* V, float* rou, float dx, float dz, float dt, int nx_real, int nz_real, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int m = (i % (nx_real * nz_real)) / nz_real; 
	int n = (i % (nx_real * nz_real)) % nz_real; 
	if (m >= N && m < nx_real - N && n >= N && n < nz_real - N)
	{
		U_x[i] = U_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt / (rou[i] * dx * (2 + dt * ddx[i])) * Ux_Rx[i];
		U_z[i] = U_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt / (rou[i] * dz * (2 + dt * ddz[i])) * Uz_Tz[i];

		V_x[i] = V_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt / (rou[i] * dx * (2 + dt * ddx[i])) * Vx_Hu[i];
		V_z[i] = V_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt / (rou[i] * dz * (2 + dt * ddz[i])) * Vz_Hv[i];

		U[i] = U_x[i] + U_z[i];
		V[i] = V_x[i] + V_z[i];
	}
}

__global__ void Group_Kernel_Get_R_T_H(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* ddx, float* ddz, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z, float* R, float* T, float* H, float* mu, float* numd, float dx, float dz, float dt, int nx_real, int nz_real, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int m = (i % (nx_real * nz_real)) / nz_real; 
	int n = (i % (nx_real * nz_real)) % nz_real; 
	if (m >= N && m < nx_real - N && n >= N && n < nz_real - N)
	{
		R_x[i] = R_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt * (numd[i] + 2 * mu[i]) / ((2 + dt * ddx[i]) * dx) * Ux_Rx[i];
		R_z[i] = R_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt * numd[i] / ((2 + dt * ddz[i]) * dz) * Vz_Hv[i];
		T_x[i] = T_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt * numd[i] / ((2 + dt * ddx[i]) * dx) * Ux_Rx[i];
		T_z[i] = T_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt * (numd[i] + 2 * mu[i]) / ((2 + dt * ddz[i]) * dz) * Vz_Hv[i];
		H_x[i] = H_x[i] * (2 - dt * ddx[i]) / (2 + dt * ddx[i]) + 2 * dt * mu[i] / ((2 + dt * ddx[i]) * dx) * Vx_Hu[i];
		H_z[i] = H_z[i] * (2 - dt * ddz[i]) / (2 + dt * ddz[i]) + 2 * dt * mu[i] / ((2 + dt * ddz[i]) * dz) * Uz_Tz[i];

		R[i] = R_x[i] + R_z[i];
		T[i] = T_x[i] + T_z[i];
		H[i] = H_x[i] + H_z[i];
	}
}

__global__ void Group_SetInitialValue(float* Ux_Rx, float* Uz_Tz, float* Vx_Hu, float* Vz_Hv, float* U, float* V, float* U_x, float* U_z, float* V_x, float* V_z, float* R, float* T, float* H, float* R_x, float* R_z, float* T_x, float* T_z, float* H_x, float* H_z)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Ux_Rx[i] = 0;
	Uz_Tz[i] = 0;
	Vx_Hu[i] = 0;
	Vz_Hv[i] = 0;
	U[i] = 0;
	V[i] = 0;
	U_x[i] = 0;
	U_z[i] = 0;
	V_x[i] = 0;
	V_z[i] = 0;
	T[i] = 0;
	H[i] = 0;
	R[i] = 0;
	T_x[i] = 0;
	T_z[i] = 0;
	H_x[i] = 0;
	H_z[i] = 0;
	R_x[i] = 0;
	R_z[i] = 0;
}

__global__ void GetRSM(float* field, float* mod)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float tmp = field[i] - mod[i];
	mod[i] = tmp*tmp;
}