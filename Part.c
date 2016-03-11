#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAXN 512

int i, j;

typedef struct { float r; float i; } complex;
static complex ctmp;

complex data1[MAXN][MAXN];
complex data2[MAXN][MAXN];
complex data3[MAXN][MAXN];

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

void fft1d(complex *r, int n, int isign)
{
	int     m, i, i1, j, k, i2, l, l1, l2;
	float   c1, c2, z;
	complex t, u;

	if (isign == 0) return;

	/* Do the bit reversal */
	i2 = n >> 1;
	j = 0;
	for (i = 0; i<n - 1; i++) {
		if (i < j)
			C_SWAP(r[i], r[j]);
		k = i2;
		while (k <= j) {
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	/* m = (int) log2((double)n); */
	for (i = n, m = 0; i>1; m++, i /= 2);

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l = 0; l<m; l++) {
		l1 = l2;
		l2 <<= 1;
		u.r = 1.0;
		u.i = 0.0;
		for (j = 0; j<l1; j++) {
			for (i = j; i<n; i += l2) {
				i1 = i + l1;

				/* t = u * r[i1] */
				t.r = u.r * r[i1].r - u.i * r[i1].i;
				t.i = u.r * r[i1].i + u.i * r[i1].r;

				/* r[i1] = r[i] - t */
				r[i1].r = r[i].r - t.r;
				r[i1].i = r[i].i - t.i;

				/* r[i] = r[i] + t */
				r[i].r += t.r;
				r[i].i += t.i;
			}
			z = u.r * c1 - u.i * c2;

			u.i = u.r * c2 + u.i * c1;
			u.r = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (isign == -1) /* FWD FFT */
			c2 = -c2;
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for inverse transform */
	if (isign == 1) {       /* IFFT*/
		for (i = 0; i<n; i++) {
			r[i].r /= n;
			r[i].i /= n;
		}
	}
}

void file_read()
{
	FILE *f1, *f2; /*open file descriptor */
	f1 = fopen("1_im1", "r");
	f2 = fopen("1_im2", "r");
	for (i = 0; i<MAXN; i++)
	{
		for (j = 0; j<MAXN; j++)
		{
			fscanf(f1, "%g", &data1[i][j].r);
			fscanf(f2, "%g", &data2[i][j].r);
			data1[i][j].i = 0;
			data2[i][j].i = 0;
		}
	}
	fclose(f1);
	fclose(f2);
}

void file_write()
{
	FILE *f1; /*open file descriptor */
	f1 = fopen("output", "w+");
	for (i = 0; i<MAXN; i++)
	{
		for (j = 0; j<MAXN; j++)
		{
			fprintf(f1, "%e  ", data3[i][j].r);
		}
		fprintf(f1, "\n");
	}
	fclose(f1);
}


void convo2d()
{
	/*ID FFT Row wise*/
	for (i = 0; i<MAXN; i++)
	{
		fft1d(data1[i], MAXN, -1);
		fft1d(data2[i], MAXN, -1);
	}

	/*Transpose*/
	for (i = 0; i < MAXN; i++)
	{
		for (j = i; j < MAXN; j++)
		{
			C_SWAP(data1[i][j], data1[j][i]);
			C_SWAP(data2[i][j], data2[j][i]);
		}
	}

	/*1D FFT Row wise (Transposed Column)*/
	for (i = 0; i<MAXN; i++)
	{
		fft1d(data1[i], MAXN, -1);
		fft1d(data2[i], MAXN, -1);
	}
	
	/*Point Multiplication*/
	for (i = 0; i < MAXN; i++)
	{
		for (j = 0; j < MAXN; j++)
		{
			data3[i][j].r = data1[i][j].r * data2[i][j].r - data1[i][j].i * data2[i][j].i;
			data3[i][j].i = data1[i][j].r * data2[i][j].i + data1[i][j].i * data2[i][j].r;
		}
	}
	
	/*1D IFFT Row wise*/
	for (i = 0; i<MAXN; i++)
	{
		fft1d(data3[i], MAXN, 1);
	}
	
	/*Transpose*/
	for (j = 0; j < MAXN; j++)
	{
		for (i = j; i < MAXN; i++)
		{
			C_SWAP(data3[i][j], data3[j][i]);
		}
	}

	/*1D IFFT Row wise (Transposed Column)*/
	for (i = 0; i<MAXN; i++)
	{
		fft1d(data3[i], MAXN, 1);
	}
}

int main(int argc, char* argv[])
{
	clock_t etstart, etstop;  /* Elapsed times using MPI */
	file_read();
	etstart = clock(); 
	convo2d();
	etstop = clock();
	file_write();
	printf("Time: %f secs\n",(((double) (etstop - etstart))/2000));
}