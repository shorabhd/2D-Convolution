#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#define MAXN 512
int i,j;

typedef struct { float r; float i; } complex;
static complex ctmp;

complex data1[MAXN][MAXN];
complex data2[MAXN][MAXN];
complex data3[MAXN][MAXN];

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

int nprocs, rank, chunk;

void fft1d(complex *r, int n, int isign)
{
   int     m,i,i1,j,k,i2,l,l1,l2;
   float   c1,c2,z;
   complex t, u;

   if (isign == 0) return;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
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
   for (i=n,m=0; i>1; m++,i/=2);

   /* Compute the FFT */
   c1 = -1.0;
   c2 =  0.0;
   l2 =  1;
   for (l=0;l<m;l++) {
      l1   = l2;
      l2 <<= 1;
      u.r = 1.0;
      u.i = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
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
         z =  u.r * c1 - u.i * c2;

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
      for (i=0;i<n;i++) {
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
			fscanf(f1, "%e", &data1[i][j].r);
			fscanf(f2, "%e", &data2[i][j].r);
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
			fprintf(f1, "%1.6e  ", data3[i][j].r);
		}
		fprintf(f1, "\n");
	}
	fclose(f1);
}

int main (int argc, char **argv) 
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	chunk = MAXN / nprocs; /* number of rows for each process */
	complex tmp;
	double time_init, time_end, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14;
	MPI_Status status;

	file_read();

	if ( rank == 0 )
	time_init = MPI_Wtime();

	/* Divide the processors in 4 groups */ 
	int size = nprocs / 2;
	int color_rank;
	int P1_array[size], P2_array[size], P3_array[size], P4_array[size];

	for(i=0; i<nprocs; i++) 
	{
		int processor_group = i / size;
		switch(processor_group)
		{
			case 0:
			P1_array[ i%size ] = i;
			break;
			case 1:
			P2_array[ i%size ] = i;
			break;
			case 2:
			P3_array[ i%size ] = i;
			break;
			case 3:
			P4_array[ i%size ] = i;
			break;
		}
	}

	MPI_Group world_group, P1, P2, P3, P4; 
	MPI_Comm P1_comm, P2_comm, P3_comm, P4_comm;

	MPI_Comm_group(MPI_COMM_WORLD, &world_group); 

	int color = rank / size;

	if (color == 0)      
	{
		MPI_Group_incl(world_group, nprocs/4, P1_array, &P1);
		MPI_Comm_create( MPI_COMM_WORLD, P1, &P1_comm);
		MPI_Group_rank(P1, &color_rank);
	} 
	else if (color == 1) 
	{ 
		MPI_Group_incl(world_group, nprocs/4, P2_array, &P2); 
		MPI_Comm_create( MPI_COMM_WORLD, P2, &P2_comm);
		MPI_Group_rank(P2, &color_rank);
	} 
	else if (color == 2) 
	{ 
		MPI_Group_incl(world_group, nprocs/4, P3_array, &P3); 
		MPI_Comm_create( MPI_COMM_WORLD, P3, &P3_comm);
		MPI_Group_rank(P3, &color_rank);
	} 
	else if (color == 3) 
	{ 
		MPI_Group_incl(world_group, nprocs/4, P4_array, &P4); 
		MPI_Comm_create( MPI_COMM_WORLD, P4, &P4_comm);
		MPI_Group_rank(P4, &color_rank);
	} 

	

	chunk = MAXN / size;

	/*Distribution of data1 and data2*/
	if ( rank == 0 )
	{
		// Send data1 to the P1 processors
		for ( i=0; i<size; i++ ) 
		{
			if ( P1_array[i]==0 ) continue;
			MPI_Send( &data1[chunk*i][0], chunk*MAXN, MPI_COMPLEX, P1_array[i], 0, MPI_COMM_WORLD );
		}
		// Send data2 to the P2 processors
		for ( i=0; i<size; i++ ) 
		{
			if ( P2_array[i]==0 ) continue;
			MPI_Send( &data2[chunk*i][0], chunk*MAXN, MPI_COMPLEX, P2_array[i], 0, MPI_COMM_WORLD );
		}
	}
	else 
	{
		// Receive data1 because this is group P1
		if (color == 0)
			MPI_Recv(&data1[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status );
		// Receive data2 because this is group P2
		if (color == 1)
			MPI_Recv( &data2[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status );
	}

	MPI_Barrier(MPI_COMM_WORLD); 
	if(rank == 0) t1 = MPI_Wtime();

	/*1D FFT Row wise in Group P1*/
	if (color == 0)
	for ( i=chunk*color_rank; i<chunk*(color_rank+1); i++ )
	fft1d(data1[i], MAXN, -1);

	/*1D FFT Row wise in Group P2 */
	if (color == 1)
		for( i=chunk*color_rank; i<chunk*(color_rank+1); i++ )
			fft1d(data2[i], MAXN, -1);

	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t2 = MPI_Wtime();

	/*Collecting data1 in root processor of P1*/
	if (color == 0) {
		if (color_rank == 0) {
			for (i=1; i<size; i++) {
				MPI_Recv(&data1[chunk*i][0], chunk*MAXN, MPI_COMPLEX, i, 0, P1_comm, &status);
			}
		}
		else 
			MPI_Send( &data1[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, P1_comm );
	}
   
    /*Collecting data2 in root processor of P2*/
	if (color == 1) {
		if ( color_rank == 0 ) {
			for ( i=1; i<size; i++ ) {
				MPI_Recv(&data2[chunk*i][0], chunk*MAXN, MPI_COMPLEX, i, 0, P2_comm, &status );
			}
		}
		else 
			MPI_Send(&data2[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, P2_comm );
	}
   
	MPI_Barrier(MPI_COMM_WORLD); 
	if (rank == 0) t3 = MPI_Wtime();

	/* Transpose data1 in root processor of P1*/
	if ( color == 0 && color_rank == 0 ) {
		for (i=0;i<MAXN;i++) {
			for (j=i;j<MAXN;j++) {
				C_SWAP(data1[i][j], data1[j][i]);
			}
		}
	}

	/* Transpose data2 in root processor of P2*/
	if ( color == 1 && color_rank == 0 ) {
		for (i=0;i<MAXN;i++) {
			for (j=i;j<MAXN;j++) {
				C_SWAP(data2[i][j], data2[j][i]);
			}
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t4 = MPI_Wtime();

	/*Again Distribution of data1 in the Group P1*/
	if (color == 0) {
		if ( color_rank == 0 ) {
			for ( i=1; i<size; i++ ) {
				MPI_Send( &data1[chunk*i][0], chunk*MAXN, MPI_COMPLEX, i, 0, P1_comm );
			}
		}
		else 
			MPI_Recv( &data1[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, P1_comm, &status );
	}

	/*Again Distribution of data2 in the Group P2*/
	if ( color == 1 ) {
		if ( color_rank == 0 ) {
			for ( i=1; i<size; i++ ) {
				MPI_Send( &data2[chunk*i][0], chunk*MAXN, MPI_COMPLEX, i, 0, P2_comm );
			}
		}
		else 
			MPI_Recv( &data2[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, P2_comm, &status );
	}
	
	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t5 = MPI_Wtime();


	/*1D FFT Row wise (Transposed Column) in Group P1*/
	if ( color == 0 )
		for ( i=chunk*color_rank; i<chunk*(color_rank+1); i++ )
			fft1d(data1[i], MAXN, -1);

	/*1D FFT Row wise (Transposed Column) in Group P2*/
	if ( color == 1 )
		for ( i=chunk*color_rank; i<chunk*(color_rank+1); i++ )
			fft1d(data2[i], MAXN, -1);

	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t6 = MPI_Wtime();

	/*Gather data1 and data2 into the P3 processors via root processors*/

	if ( color == 0 )
		MPI_Send ( &data1[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, P3_array[color_rank], 0, MPI_COMM_WORLD );
	
	else if ( color == 1 )
		MPI_Send ( &data2[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, P3_array[color_rank], 0, MPI_COMM_WORLD );

	else if ( color == 2 ) 
	{
		MPI_Recv( &data1[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, P1_array[color_rank], 0, MPI_COMM_WORLD, &status );
		MPI_Recv( &data2[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, P2_array[color_rank], 0, MPI_COMM_WORLD, &status );
	}

	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t7 = MPI_Wtime();

	/* Point Multiplication */
	if ( color == 2 ) 
	{
		for (i= chunk*color_rank ;i< chunk*(color_rank+1);i++) 
		{
			for (j=0;j<MAXN;j++) 
			{
				data3[i][j].r = data1[i][j].r*data2[i][j].r - data1[i][j].i*data2[i][j].i;
				data3[i][j].i = data1[i][j].r*data2[i][j].i + data1[i][j].i*data2[i][j].r;
			}
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD); 
	if (rank == 0) t8 = MPI_Wtime();

	/* Sending the Output Matrix to P4 */
	if (color == 2) 
	{
		MPI_Send ( &data3[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, P4_array[color_rank], 0, MPI_COMM_WORLD );
	}
	else if (color == 3) 
	{
		MPI_Recv( &data3[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, P3_array[color_rank], 0, MPI_COMM_WORLD, &status );
	}
	
	MPI_Barrier(MPI_COMM_WORLD); 
	if (rank == 0) t9 = MPI_Wtime();

	/*1D IFFT Row wise on Output Matrix*/
	if (color == 3)
		for ( i=chunk*color_rank; i<chunk*(color_rank+1); i++ )
			fft1d(data3[i], MAXN, 1);

	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t10 = MPI_Wtime();

	/*Collecting data3 in root processor of P4*/
	if (color == 3) 
	{
		if (color_rank == 0) 
		{
			for ( i=1; i<size; i++ ) {
				MPI_Recv( &data3[chunk*i][0], chunk*MAXN, MPI_COMPLEX, i, 0, P4_comm, &status );
			}
		}
		else 
			MPI_Send( &data3[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, P4_comm );
	}
	
	MPI_Barrier(MPI_COMM_WORLD); 
	if (rank == 0) t11 = MPI_Wtime();

	/* Transpose data3*/
	if (color == 3 && color_rank == 0) 
	{
		for (i=0;i<MAXN;i++) 
		{
			for (j=i;j<MAXN;j++) 
			{
				C_SWAP(data3[i][j], data3[j][i]);
			}	
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t12 = MPI_Wtime();


	/*Distribution of data3 in the Group P4 */
	if (color == 3) 
	{
		if (color_rank == 0) 
		{
			for ( i=1; i<size; i++ ) 
			{
				MPI_Send(&data3[chunk*i][0], chunk*MAXN, MPI_COMPLEX, i, 0, P4_comm );
			}
		}
		else 
			MPI_Recv(&data3[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, P4_comm, &status );
	}
	
	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t13 = MPI_Wtime();

	/*1D IFFT Row wise(Transposed Column) on Output Matrix*/
	if ( color == 3 )
		for ( i=chunk*color_rank; i<chunk*(color_rank+1); i++ )
			fft1d(data3[i], MAXN, 1);

	MPI_Barrier(MPI_COMM_WORLD); 
	if ( rank == 0 ) t14 = MPI_Wtime();

	/*Collecting data3 in root processor of Group P4*/
	if (rank == 0)
	{
		for ( i=0; i<size; i++ ) 
		{
			if (P4_array[i]==0) continue;
			MPI_Recv(&data3[chunk*i][0], chunk*MAXN, MPI_COMPLEX, P4_array[i], 0, MPI_COMM_WORLD, &status);
		}
	}
	else if (color == 3)
		MPI_Send(&data3[chunk*color_rank][0], chunk*MAXN, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD); 

	/* Final time */
	if ( rank == 0 )
	time_end = MPI_Wtime();

	/* Write output file */
	file_write();

	if (rank==0) 
	{
		double tcomputation = (t2-t1) + (t4-t3) + (t6-t5) + (t8-t7) + (t10-t9) + (t12-t11) + (t14-t13);
		double tcommunication = (t1-time_init) + (t3-t2) + (t5-t4) + (t7-t6) + (t9-t8) + (t11-t10) + (t13-t12) + (time_end-t14);
		printf("Computation Time is  %f ms\n", tcomputation * 1000 );
		printf("Communication Time is  %f ms\n", tcommunication * 1000 );
	}
	MPI_Finalize();
}