#include <math.h>
#include <stdio.h>
#include <iostream>
#include <random>

#define DIMS (2)
#define NUM_THREADS (1024)
#define SCALE (0.1)
#define VEL_SCALE (1.0)

// This is supposed to be dynamically compiled with these options
// #define NUM_BLOCKS (5000)
// #define NUM_FRAMES (600)
// #define DT (0.05)
// #define WIDTH (284)
// #define HEIGHT (160)
// #define SHIFT (-5)
// #define PUSH (0.3)
// #define INITIAL_DISTRIBUTION (1)
// #define FORCE_TYPE (1)
// #define EXPOSURE (1)
// #define SHOW_MOMENTUM (0)
// #define MEASUREMENT_TYPE (0)

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
  float ay = -(pow(y+1, 3) * exp(-pow(4*(y+1), 4)) + pow(y-1, 3) * exp(-pow(4*(y-1), 4))) * wall * 1024;
  vel[0] += ax * DT;
  vel[1] += ay * DT;
}

__device__
void force_mirror(float *pos, float *vel) {
  float x = pos[0];
  float y = pos[1];
  float curve = x - 5.5 + 0.07 * pow(y, 2);
  float potential = exp(-pow(curve, 4));
  float ax = 4 * pow(curve, 3) * potential;
  float ay = 0.07 * 2 * y * ax;
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
  #elif (FORCE_TYPE == 3)
    force_mirror(pos + index, vel + index);
  #endif
  #if SHOW_MOMENTUM
    int i = (int)floor(vel[index] * HEIGHT * VEL_SCALE + WIDTH * 0.5);
    int j = (int)floor(vel[index+1] * HEIGHT * VEL_SCALE + HEIGHT * 0.5);
  #else
    int i = (int)floor(pos[index] * HEIGHT * SCALE + WIDTH * 0.5);
    int j = (int)floor(pos[index+1] * HEIGHT * SCALE + HEIGHT * 0.5);
  #endif
  if (i >= 0 && i < WIDTH && j >= 0 && j < HEIGHT) {
    atomicAdd(counts + (i + j * WIDTH), 1);
  }
}

int main(void)
{
  int N = NUM_THREADS * NUM_BLOCKS;
  float *pos, *vel;
  double t = 0;
  #if (MEASUREMENT_TYPE)
    int measurement_done = false;
  #endif

  int *counts;

  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<float> dist(0, 1);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&pos, N*DIMS*sizeof(float));
  cudaMallocManaged(&vel, N*DIMS*sizeof(float));
  cudaMallocManaged(&counts, WIDTH*HEIGHT*sizeof(int));

  #if (INITIAL_DISTRIBUTION == 1)
    for (int i = 0; i < N; i++) {
      pos[2*i] = dist(rng) * 0.5 + SHIFT;
      pos[2*i+1] = dist(rng) * 0.5;
      vel[2*i] = dist(rng) * 0.1 + PUSH;
      vel[2*i+1] = dist(rng) * 0.1;
    }
  #elif (INITIAL_DISTRIBUTION == 2)
    for (int i = 0; 2*i < N; ++i) {
      pos[4*i] = dist(rng) * 0.1 - 3;
      pos[4*i+1] = dist(rng) * 0.1 + 1;
      pos[4*i+2] = dist(rng) * 0.5 + 3;
      pos[4*i+3] = dist(rng) * 0.5 - 0.5;
      vel[4*i] = dist(rng) * 0.1 + PUSH;
      vel[4*i+1] = dist(rng) * 0.1;
      vel[4*i+2] = dist(rng) * 0.1 - PUSH;
      vel[4*i+3] = dist(rng) * 0.1;
    }
  #endif

  for (int i = 0; i < NUM_FRAMES; ++i) {
    for (int j = 0; j < EXPOSURE; ++j) {
      t += DT;
      step<<<N / NUM_THREADS, NUM_THREADS>>>(pos, vel, counts);
    }
    cudaDeviceSynchronize();
    #if (MEASUREMENT_TYPE)
      if (t >= 1.0 and !measurement_done) {
        for (int i = 0; i < N; ++i) {
          float x = pos[2*i];
          float y = pos[2*i+1];
          int condition = 0.3 < x && x < 1.8 && -0.1 > y && y > -1.6;
          #if (MEASUREMENT_TYPE == 1)
            if (!condition) {
          #else
            if (condition) {
          #endif
            pos[2*i] = 5;
            pos[2*i+1] = 5;
            vel[2*i] = 0;
            vel[2*i + 1] = 0;
          }
        }
        measurement_done = true;
      }
    #endif
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
