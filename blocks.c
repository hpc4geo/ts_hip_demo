
#include "stdio.h"

#define GRID_MAX 1024
#define THREAD_MAX 1024

typedef struct {
  long int x;
  long int y;
  long int z;
} dim3;

/*
blockIdx is the index within the grid.
blockDim indicates the number of threads in each block.
*/
long int get_global_index(dim3 gridDim, dim3 blockIdx, dim3 blockDim, dim3 threadIdx)
{
  long int bijk = blockIdx.x + blockIdx.y * (gridDim.x) + blockIdx.z * (gridDim.x * gridDim.y);
  long int tijk = threadIdx.x + threadIdx.y * (blockDim.x) + threadIdx.z * (blockDim.x * blockDim.y);
  long int i = bijk * (blockDim.x * blockDim.y * blockDim.z) + tijk;
  return i;
}

void build_sizes(long int N,dim3 *b, dim3 *t)
{
  b->x = b->y = b->z = 1;
  t->x = t->y = t->z = 1;

  if (N < GRID_MAX * THREAD_MAX) {
    printf("1d ->\n");

    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = (N + GRID_MAX-1)/GRID_MAX; b->y = 1; b->z = 1;

    return;
  }

  if (N < GRID_MAX * GRID_MAX * THREAD_MAX) {
    printf("2d ->\n");

    long int bJ = (N-1)/(GRID_MAX*GRID_MAX);
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = GRID_MAX; b->y = bJ+1; b->z = 1;

    return;
  }


  long int Nt = (long int)(N/THREAD_MAX);
  if ( Nt < GRID_MAX * GRID_MAX * GRID_MAX) {
    printf("3d ->\n");

    long int bK = (long int) ( Nt/(GRID_MAX*GRID_MAX) );
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = GRID_MAX; b->y = GRID_MAX; b->z = bK+1;

    return;
  }
}

void build_sizes_blocksize(long int N,dim3 *b, dim3 *t)
{
  b->x = b->y = b->z = 1;
  t->x = t->y = t->z = 1;

  if (N >= GRID_MAX*GRID_MAX*GRID_MAX) { printf("Error - N too large\n"); }
  if (N < GRID_MAX) {
    printf("1d ->\n");

    b->x = (N + GRID_MAX-1)/GRID_MAX; b->y = 1; b->z = 1;

    return;
  }

  if (N < GRID_MAX * GRID_MAX) {
    printf("2d ->\n");

    long int bJ = (N-1)/(GRID_MAX*GRID_MAX);
    b->x = GRID_MAX; b->y = bJ+1; b->z = 1;

    return;
  }


  if ( N < GRID_MAX * GRID_MAX * GRID_MAX) {
    printf("3d ->\n");

    long int bK = (long int) ( N/(GRID_MAX*GRID_MAX) );
    b->x = GRID_MAX; b->y = GRID_MAX; b->z = bK+1;

    return;
  }
}


long int printd3(dim3 *b) {
  printf("(%ld %ld %ld) -> max items %ld\n",b->x,b->y,b->z, b->x*b->y*b->z);
  return b->x*b->y*b->z;
}

int main(int a,char *b[])
{
  dim3 blk,th;
  long int N, sum;

  printf("1d\n");
  N = 1000; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N);

  N = 1025; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N);


  printf("2d\n");
  N = 1024 * 1024; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N); else { printf("error N=%ld\n",N);}

  N = 1024 * 1024+1; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N); else { printf("error N=%ld\n",N);}


  N = 1024 * 1024 * 1024; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N); else { printf("error N=%ld\n",N);}

  N = 1024 * 1024 * 1024 + 1; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N); else { printf("error N=%ld\n",N);}

  N = 1024 * 1024 * 1024 - 1; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N); else { printf("error N=%ld\n",N);}

  N = 1024 * 1024 * 1024 + 1024; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N); else { printf("error N=%ld\n",N);}

  N = 1099511627775 ; build_sizes(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N); else { printf("error N=%ld\n",N);}


  N = 1024 * 1024 * 4; build_sizes_blocksize(N,&blk,&th);
  sum = printd3(&blk)*printd3(&th);
  printf("max %ld -- ",sum); if (N <= sum) printf("ok (N=%ld)\n",N); else { printf("error N=%ld\n",N);}


  return 0;
}
