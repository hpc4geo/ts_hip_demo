
## Outline of example codes

blocks.c
- Demo creating a grid/thread plan for a kernel given a size `N`.

ex4.c
- Example creating petsc Vec's of type hip via VecCreate() and DMCreateGlobalVector() 

ex1.c
- TS example (cpu onlu) reference code.

ex2.c ex2_res.c
- Same as ex1.c expect the main and RHSFunction are in different files.

ex3.c ex3_res.cpp
- Example containing a hip kernel defining the residual for TS. Same as ex2.c just using hip.




