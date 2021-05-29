#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array>
#include <random>
#include <set>
#include <numeric>

#include "TracerCUDA/ParallelPartition.cuh"

struct CopyFunctor
{
    __device__ __host__ __forceinline__
    uint32_t operator()(const int keys) const
    {
        return keys;
    }
};

TEST(ParallelPartition, Generic)
{
    static constexpr int IterationCount = 19;
    static constexpr int ElementCount = 75'000;
    static constexpr int PARTITION_MAX = 1000;
    static constexpr int PARTITION_MIN = 500;
    std::uniform_int_distribution<int> partitionAmountSelector(PARTITION_MIN, PARTITION_MAX);
    std::vector<int> partitionCounts;
    partitionCounts.reserve(PARTITION_MAX);

    std::mt19937 rng;
    rng.seed(0);
   
    DeviceMemory partitionIds(ElementCount * sizeof(int));
    std::vector<int> hPartitionKeyList(ElementCount);
    std::vector<uint32_t> hPartitionIndexList(ElementCount);

    // Start testing the function
    for(int iterCount = 0; iterCount < IterationCount; iterCount++)
    {
        // Randomly select partition amount
        int totalPartitionCount = partitionAmountSelector(rng);
        partitionCounts.resize(totalPartitionCount);
        
        // Allocate a random partition list
        std::uniform_int_distribution<int> idSelector(0, totalPartitionCount - 1);
        for(int i = 0; i < ElementCount; i++)
        {
            int key = idSelector(rng);
            hPartitionKeyList[i] = key;
        }

        // Find partitionSizes
        std::for_each(partitionCounts.begin(), partitionCounts.end(),
                      [](int& c){c = 0;});
        for(int i = 0; i < ElementCount; i++)
        {            
            int key = hPartitionKeyList[i];
            partitionCounts[key]++;
        }

        // Copy the data to GPU
        CUDA_CHECK(cudaMemcpy(static_cast<int*>(partitionIds),
                              hPartitionKeyList.data(),
                              sizeof(int) * ElementCount,
                              cudaMemcpyHostToDevice));

        // Partitionized the id randomly
        // Now call partition
        DeviceMemory sortedIndexBuffer;
        std::set<ArrayPortion<int>> partitionSet;
        PartitionGPU(partitionSet,
                     sortedIndexBuffer,
                     static_cast<const int*>(partitionIds),
                     ElementCount,
                     CopyFunctor(),
                     totalPartitionCount);

        // Copy Data back to CPU
        CUDA_CHECK(cudaMemcpy(hPartitionIndexList.data(),
                              static_cast<const uint32_t*>(sortedIndexBuffer),
                              sizeof(uint32_t) * ElementCount,
                              cudaMemcpyDeviceToHost));

        // Now Check
        for(const ArrayPortion<int>& p : partitionSet)
        {
            EXPECT_EQ(p.count, partitionCounts[p.portionId]);
            for(size_t i = 0; i < p.count; i++)
            {                
                EXPECT_EQ(hPartitionKeyList[hPartitionIndexList[p.offset + i]], p.portionId);
            }
        }
    }
}