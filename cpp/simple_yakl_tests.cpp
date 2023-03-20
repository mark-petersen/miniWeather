//////////////////////////////////////////////////////////////////////////////////////////
// simple yakl tests
// Author: Mark Petersen
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include "const.h" // This contains the yakl definitions

typedef yakl::Array<real  ,3,yakl::memDevice> real3d;
typedef yakl::Array<real  ,3,yakl::memHost> real3dHost;

int main() {
  yakl::init();

  int nx = 3;
  int ny = 3;
  int nz = 3;
  real3d     workArray    ( "workArray" , nx,ny,nz ); // work array on the device (gpu)
  real3dHost workArray_cpu( "workArray" , nx,ny,nz ); // work array on the host (cpu)

  // initialize array on device
  // Note that the first argument is option, may be a string or
  //   parallel_for( YAKL_AUTO_LABEL(), ...
  // Then the loops are labeled in the nvidia diagnostics output here with:
  //   srun -n 1 -G4 nsys nvprof ./simple_yakl_tests  // on perlmutter
  //   jsrun -n 1 -a 1 -c 1 -g 1 nvprof ./simple_yakl_tests  // on summit
  parallel_for( "init array 1", SimpleBounds<3>(nx,ny,nz) , YAKL_LAMBDA (int i, int j, int k) {
    workArray(i,j,k) = i*100.0 + j*10.0 + k;
  });
  yakl::fence();

  std::cout << "workArray before copy" << std::endl;
  std::cout << workArray << std::endl;

  std::cout << "workArray_cpu before copy" << std::endl;
  std::cout << workArray_cpu << std::endl;

  // Copy from GPU to host
  workArray.deep_copy_to(workArray_cpu);
  yakl::fence();

  std::cout << "workArray_cpu after copy" << std::endl;
  std::cout << workArray_cpu << std::endl;

  // alter array on cpu
  for (int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      for (int k=0; k<nz; k++) {
          workArray_cpu(i,j,k) += 0.4;
      }
    }
  }

  // Copy from host to GPU
  workArray_cpu.deep_copy_to(workArray);
  yakl::fence();

  std::cout << "workArray after copy back" << std::endl;
  std::cout << workArray << std::endl;

  std::cout << "workArray_cpu after copy back" << std::endl;
  std::cout << workArray_cpu << std::endl;
}

