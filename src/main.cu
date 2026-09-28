/**
 * @file main.cu
 * @brief Counting Sort algorithm parallelized with CUDA.
 * @author Marco Plaitano
 * @date 19 Jan 2022
 *
 * COUNTING SORT CUDA
 * Parallelize and Evaluate Performances of "Counting Sort" Algorithm, by using
 * CUDA.
 *
 * Copyright (C) 2022 Plaitano Marco
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "util.h"


/**
 * @brief Count the number of occurrences of each element in the array.
 * @param array_device: Device copy of the input array.
 * @param array_size:   Size of the input array.
 * @param count_device: Device copy of count array.
 * @param count_size:   Size of the count array.
 * @param min:          Minimum value in the input array.
 *
 * Kernel call.
 */
__global__ void kernel_count(const int *array_device, long long array_size,
                             int *count_device, int count_size, int min);



int main(int argc, char **argv) {
    /* Check for the correct amount of command line arguments. */
    if (argc != 3) {
        fprintf(stderr, "ERROR! usage: bin/global.out array_size block_size\n");
        exit(EXIT_FAILURE);
    }

    /* Check for the correctness of the parameters. */
    if (RANGE_MAX <= RANGE_MIN) {
        fprintf(stderr, "ERROR! can't have RANGE_MAX <= RANGE_MIN.\n");
        exit(EXIT_FAILURE);
    }

    float time_kernel = 0, time_sort = 0;
    long long i = 0, j = 0, k = 0;
    int min = 0, max = 0;

    /*INISIALISASI DATA DI HOST (CPU RAM)*/
    const long long size = atoll(argv[1]);
    int *array = (int *)safe_alloc(size * sizeof(int));
    array_init_random(array, size, RANGE_MIN, RANGE_MAX);

    /*ALOKASI MEMORI DI DEVICE (GPU VRAM) & TRANSFER DATA */
    int *array_device;
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&array_device, size * sizeof(int))
    );
    CUDA_ERROR_CHECK(
        cudaMemcpy(array_device, array, size * sizeof(int),
                   cudaMemcpyHostToDevice)
    );

    /* Cari nilai minimum dan maksimum dari array */
    array_min_max(array, size, &min, &max);
    const int count_size = max - min + 1;

    /* Inisialisasi array frekuensi (count) di Host dan Device */
    int *count = (int *)safe_alloc(sizeof(int) * count_size);
    int *count_device;
    CUDA_ERROR_CHECK(
        cudaMalloc((void **)&count_device, count_size * sizeof(int))
    );

    /*KONFIGURASI GRID DAN BLOCK CUDA*/
    dim3 dimBlock(atoi(argv[2]));
    dim3 dimGrid((size - 1) / dimBlock.x + 1);

    /*EKSEKUSI KERNEL CUDA (PENGHITUNGAN FREKUENSI)*/
    START_TIME(time_sort);

    START_CUDA_TIME(time_kernel);
    kernel_count<<<dimGrid, dimBlock>>>(array_device, size, count_device,
                                        count_size, min);
    cudaDeviceSynchronize();
    CUDA_KERNEL_ERROR_CHECK;
    END_CUDA_TIME(time_kernel);

    /*TRANSFER HASIL DARI GPU KE CPU & REKONSTRUKSI ARRAY */
    CUDA_ERROR_CHECK(
        cudaMemcpy(count, count_device, count_size * sizeof(int),
                   cudaMemcpyDeviceToHost)
    );

    /* Rekonstruksi array berdasar frekuensi count (Serial di Host) */
    for (i = min; i < max + 1; i++)
        for (j = 0; j < count[i - min]; j++)
            array[k++] = i;

    END_TIME(time_sort);

    /* VALIDASI KETERURUTAN & TAMPILKAN HASIL DENGAN LABEL */
    bool is_sorted = array_is_sorted(array, size);

    /* Bebaskan alokasi memori */
    CUDA_ERROR_CHECK( cudaFree(array_device) );
    CUDA_ERROR_CHECK( cudaFree(count_device) );
    free(array);
    free(count);

    if (!is_sorted) {
        fprintf(stderr, "\n[ERROR] Array TIDAK terurut dengan benar!\n");
        return EXIT_FAILURE;
    } else {
        printf("           HASIL COUNTING SORT CUDA (PARALLEL)         \n");
        printf(" Jumlah Elemen (N)          : %lld data\n", size);
        printf(" Rentang Nilai (Range)      : [%d, %d]\n", min, max);
        printf(" Block Size (Threads/Block) : %d threads\n", dimBlock.x);
        printf(" Grid Size (Jumlah Block)   : %d blocks\n", dimGrid.x);
        printf("------------------------------------------------------\n");
        printf(" Waktu Eksekusi Kernel GPU  : %.5f detik\n", time_kernel);
        printf(" Total Waktu Sorting        : %.5f detik\n", time_sort);
        printf(" Status Pengurutan          : VALID (Array Terurut Sempurna)\n");
        printf("------------------------------------------------------\n");
        printf(" Format Asli (CSV)          : %lld;%d;%d;%.5f;%.5f\n",
               size, dimGrid.x, dimBlock.x, time_kernel, time_sort);
        printf("======================================================\n\n");
        return EXIT_SUCCESS;
    }
}




__global__ void kernel_count(const int *array_device, long long array_size,
                             int *count_device, int count_size, int min)
{
    /* Global index to identify current thread. */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    /* Value by which the index has to be shifted to cover the next position. */
    int offset = blockDim.x;

    /*
     * Copy of the count array stored in the block's shared memory. This array
     * is shared by all - and only - the threads of a block.
     * Since its allocation has to be static, the default range is used to
     * calculate its size though the actual size (count_size given as argument)
     * might be smaller; for all types of calculations, only the first
     * count_size elements will be taken into account.
     */
    __shared__ int count_shared[RANGE_MAX - RANGE_MIN + 1];

    /* Each thread sets some of the elements of count at 0. */
    for (int i = threadIdx.x; i < count_size; i += offset)
        count_shared[i] = 0;

    __syncthreads();

    /*
     * Based on how the grid and block dimensions are calculated, there are
     * always enough threads to cover the input array. Sometimes - when the
     * size is not divisible by the block size - the number of threads exceeds
     * the array size; that explains the need for this if clause.
     */
    if (index < array_size)
        /* Access the elements of the input array via texture memory. */
        atomicAdd(&count_shared[__ldg(&array_device[index]) - min], 1);

    __syncthreads();

    /* Merge all the shared versions of count into the global one. */
    for (int i = threadIdx.x; i < count_size; i += offset)
        atomicAdd(&count_device[i], count_shared[i]);
}
