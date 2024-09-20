
#include "stdio.h"

#define GRID_MAX 1024
#define THREAD_MAX 1024

typedef struct {
  long int x;
  long int y;
  long int z;
} dim3;

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
    long int N2d = Nt - GRID_MAX * bK;
    long int bJ = N2d/GRID_MAX;
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
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
  
  
  return 0;
}



