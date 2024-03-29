// Copyright 2019 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DAWNNATIVE_BUDDYMEMORYALLOCATOR_H_
#define DAWNNATIVE_BUDDYMEMORYALLOCATOR_H_

#include <vector>

#include "dawn_native/BuddyAllocator.h"
#include "dawn_native/MemoryAllocator.h"
#include "dawn_native/ResourceMemoryAllocation.h"

namespace dawn_native {
    // BuddyMemoryAllocator uses the buddy allocator to sub-allocate blocks of device
    // memory created by MemoryAllocator clients. It creates a very large buddy system
    // where backing device memory blocks equal a specified level in the system.
    //
    // Upon sub-allocating, the offset gets mapped to device memory by computing the corresponding
    // memory index and should the memory not exist, it is created. If two sub-allocations share the
    // same memory index, the memory refcount is incremented to ensure de-allocating one doesn't
    // release the other prematurely.
    //
    // The device will only create up to Log2(kMaxResourceSize) allocators and can prefer speed
    // over memory footprint by selecting an allocator with a higher memory threshold which results
    // in pre-allocating more memory.
    //
    // The resource allocation is guaranteed by the device to have compatible memory flags.
    class BuddyMemoryAllocator {
      public:
        BuddyMemoryAllocator(uint64_t maxBlockSize,
                             uint64_t memorySize,
                             std::unique_ptr<MemoryAllocator> client);
        ~BuddyMemoryAllocator() = default;

        ResultOrError<ResourceMemoryAllocation> Allocate(uint64_t allocationSize,
                                                         uint64_t alignment,
                                                         int memoryFlags = 0);
        void Deallocate(const ResourceMemoryAllocation& allocation);

        uint64_t GetMemorySize() const;

        // For testing purposes.
        uint64_t ComputeTotalNumOfHeapsForTesting() const;

      private:
        uint64_t GetMemoryIndex(uint64_t offset) const;

        uint64_t mMemorySize = 0;

        BuddyAllocator mBuddyBlockAllocator;
        std::unique_ptr<MemoryAllocator> mClient;

        struct TrackedSubAllocations {
            size_t refcount = 0;
            std::unique_ptr<ResourceHeapBase> mMemoryAllocation;
        };

        std::vector<TrackedSubAllocations> mTrackedSubAllocations;
    };
}  // namespace dawn_native

#endif  // DAWNNATIVE_BUDDYMEMORYALLOCATOR_H_
