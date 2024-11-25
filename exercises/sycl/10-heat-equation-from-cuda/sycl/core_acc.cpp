/* Main solver routines for heat equation solver */

#include <sycl/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat.h"

/* Update the temperature values using five-point stencil */
void evolve_kernel(double *currdata, double *prevdata, double a, double dt, int nx, int ny,
                       double dx2, double dy2, const sycl::nd_item<3> &item_ct1)
{

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    int ind, ip, im, jp, jm;

    // CUDA threads are arranged in column major order; thus j index from x, i from y
    int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

    if (i > 0 && j > 0 && i < nx+1 && j < ny+1) {
        ind = i * (ny + 2) + j;
        ip = (i + 1) * (ny + 2) + j;
        im = (i - 1) * (ny + 2) + j;
        jp = i * (ny + 2) + j + 1;
        jm = i * (ny + 2) + j - 1;
        currdata[ind] = prevdata[ind] + a * dt *
          ((prevdata[ip] -2.0 * prevdata[ind] + prevdata[im]) / dx2 +
          (prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm]) / dy2);

    }

}

void evolve(field *curr, field *prev, double a, double dt)
{
    int nx, ny;
    double dx2, dy2;
    nx = prev->nx;
    ny = prev->ny;
    dx2 = prev->dx * prev->dx;
    dy2 = prev->dy * prev->dy;

    /* CUDA thread settings */
    const int blocksize = 16;  //!< CUDA thread block dimension
    sycl::range<3> dimBlock(1, blocksize, blocksize);
    // CUDA threads are arranged in column major order; thus make ny x nx grid
    sycl::range<3> dimGrid(1, (nx + 2 + blocksize - 1) / blocksize,
                           (ny + 2 + blocksize - 1) / blocksize);

    {
        global_queue.submit([&](sycl::handler &cgh) {
            auto *curr_devdata_ct0 = curr->devdata;
            auto *prev_devdata_ct1 = prev->devdata;

            cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                             [=](sycl::nd_item<3> item_ct1) {
                                 evolve_kernel(curr_devdata_ct0,
                                               prev_devdata_ct1, a, dt, nx, ny,
                                               dx2, dy2, item_ct1);
                             });
        });
    }
    global_queue.wait_and_throw();
}

void enter_data(field *temperature1, field *temperature2)
{
    size_t datasize;

    datasize = (temperature1->nx + 2) * (temperature1->ny + 2) * sizeof(double);

    temperature1->devdata =
        (double *)sycl::malloc_device(datasize, global_queue);
    temperature2->devdata =
        (double *)sycl::malloc_device(datasize, global_queue);

    global_queue.memcpy(temperature1->devdata, temperature1->data, datasize);
    global_queue.memcpy(temperature2->devdata, temperature2->data, datasize);
    global_queue.wait();
}

/* Copy a temperature field from the device to the host */
void update_host(field *temperature)
{
    size_t datasize;

    datasize = (temperature->nx + 2) * (temperature->ny + 2) * sizeof(double);
    global_queue.memcpy(temperature->data, temperature->devdata, datasize);
    global_queue.wait();
}

/* Copy a temperature field from the host to the device */
void update_device(field *temperature)
{
    size_t datasize;

    datasize = (temperature->nx + 2) * (temperature->ny + 2) * sizeof(double);
    global_queue.memcpy(temperature->devdata, temperature->data, datasize);
}

