
#ifndef __ctx_h__
#define __ctx_h__

struct _params {
  int npoints;
  double *elements;
};

typedef struct _params Params;
typedef struct _params* PParams;


typedef struct {
  Params *host;
  Params *device;
} Context;

#endif
