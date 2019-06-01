#include <math.h>
#include <stdio.h>
#include <iostream>
#include <random>

#define DIMS (2)
#define NUM_THREADS (1024)
#define SCALE (0.1)

// This is supposed to be dynamically compiled with these options
// #define NUM_BLOCKS (5000)
// #define NUM_FRAMES (600)
// #define DT (0.05)
// #define WIDTH (284)
// #define HEIGHT (160)
// #define PUSH (0.3)
// #define INITIAL_DISTRIBUTION (1)
// #define FORCE_TYPE (1)

__device__
void force_barrier(float *pos, float *vel) {
  float x = pos[0];
  float y = pos[1];
  float ax = pow(x, 3) * exp(-pow(x, 4));
  float ay = -1e-4*pow(y, 3);
  vel[0] += ax * DT;
  vel[1] += ay * DT;
}

__device__
void force_double_slit(float *pos, float *vel) {
  float x = pos[0];
  float y = pos[1];
  float wall = exp(-pow(4*(x+1), 4));
  float holes = 1 - exp(-pow(4*(y+1), 4)) - exp(-pow(4*(y-1), 4));
  float ax = 1024 * pow(x+1, 3) * wall * holes;
  float ay = -(pow(y+1, 3) * exp(-pow(4*(y+1), 4)) + pow(y-1, 3) * exp(-pow(4*(y-1), 4))) * wall * holes * 1024;
  vel[0] += ax * DT;
  vel[1] += ay * DT;
}

__global__
void step(float *pos, float *vel, int *counts)
{
  int index = DIMS * (threadIdx.x + NUM_THREADS * blockIdx.x);
  pos[index] += vel[index] * DT;
  pos[index + 1] += vel[index + 1] * DT;
  #if (FORCE_TYPE == 1)
    force_barrier(pos+index, vel+index);
  #elif (FORCE_TYPE == 2)
    force_double_slit(pos+index, vel+index);
  #endif
  int i = (int)floor(pos[index] * HEIGHT * SCALE + WIDTH * 0.5);
  int j = (int)floor(pos[index+1] * HEIGHT * SCALE + HEIGHT * 0.5);
  if (i >= 0 && i < WIDTH && j >= 0 && j < HEIGHT) {
    atomicAdd(counts + (i + j * WIDTH), 1);
  }
}

int main(void)
{
  int N = NUM_THREADS * NUM_BLOCKS;
  float *pos, *vel;

  int *counts;

  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<float> dist(0, 1);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&pos, N*DIMS*sizeof(float));
  cudaMallocManaged(&vel, N*DIMS*sizeof(float));
  cudaMallocManaged(&counts, WIDTH*HEIGHT*sizeof(int));

  for (int i = 0; i < N; i++) {
    #if (INITIAL_DISTRIBUTION == 1)
      pos[2*i] = dist(rng) * 0.5 - 5;
      pos[2*i+1] = dist(rng) * 0.5;
      vel[2*i] = dist(rng) * 0.1 + PUSH;
      vel[2*i+1] = dist(rng) * 0.1;
    #endif
  }

  for (int i = 0; i < NUM_FRAMES; ++i) {
    step<<<N / NUM_THREADS, NUM_THREADS>>>(pos, vel, counts);
    cudaDeviceSynchronize();
    fwrite(counts, sizeof(int), WIDTH * HEIGHT, stdout);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
      counts[i] = 0;
    }
  }

  // Free memory
  cudaFree(pos);
  cudaFree(vel);
  cudaFree(counts);

  return 0;
}
