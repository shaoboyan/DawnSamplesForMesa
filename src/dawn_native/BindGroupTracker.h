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

#ifndef DAWNNATIVE_BINDGROUPTRACKER_H_
#define DAWNNATIVE_BINDGROUPTRACKER_H_

#include "common/Constants.h"
#include "dawn_native/BindGroupLayout.h"
#include "dawn_native/Pipeline.h"
#include "dawn_native/PipelineLayout.h"

#include <array>
#include <bitset>

namespace dawn_native {

    // Keeps track of the dirty bind groups so they can be lazily applied when we know the
    // pipeline state or it changes.
    // |BindGroup| is a template parameter so a backend may provide its backend-specific
    // type or native handle.
    // |DynamicOffset| is a template parameter because offsets in Vulkan are uint32_t but uint64_t
    // in other backends.
    template <typename BindGroup, bool CanInheritBindGroups, typename DynamicOffset = uint64_t>
    class BindGroupTrackerBase {
      public:
        void OnSetBindGroup(uint32_t index,
                            BindGroup bindGroup,
                            uint32_t dynamicOffsetCount,
                            uint64_t* dynamicOffsets) {
            ASSERT(index < kMaxBindGroups);

            if (mBindGroupLayoutsMask[index]) {
                // It is okay to only dirty bind groups that are used by the current pipeline
                // layout. If the pipeline layout changes, then the bind groups it uses will
                // become dirty.

                if (mBindGroups[index] != bindGroup) {
                    mDirtyBindGroups.set(index);
                    mDirtyBindGroupsObjectChangedOrIsDynamic.set(index);
                }

                if (dynamicOffsetCount > 0) {
                    mDirtyBindGroupsObjectChangedOrIsDynamic.set(index);
                }
            }

            mBindGroups[index] = bindGroup;
            mDynamicOffsetCounts[index] = dynamicOffsetCount;
            SetDynamicOffsets(mDynamicOffsets[index].data(), dynamicOffsetCount, dynamicOffsets);
        }

        void OnSetPipeline(PipelineBase* pipeline) {
            mPipelineLayout = pipeline->GetLayout();
            if (mLastAppliedPipelineLayout == mPipelineLayout) {
                return;
            }

            // Keep track of the bind group layout mask to avoid marking unused bind groups as
            // dirty. This also allows us to avoid computing the intersection of the dirty bind
            // groups and bind group layout mask in Draw or Dispatch which is very hot code.
            mBindGroupLayoutsMask = mPipelineLayout->GetBindGroupLayoutsMask();

            // Changing the pipeline layout sets bind groups as dirty. If CanInheritBindGroups,
            // the first |k| matching bind groups may be inherited.
            if (CanInheritBindGroups && mLastAppliedPipelineLayout != nullptr) {
                // Dirty bind groups that cannot be inherited.
                std::bitset<kMaxBindGroups> dirtiedGroups =
                    ~mPipelineLayout->InheritedGroupsMask(mLastAppliedPipelineLayout);

                mDirtyBindGroups |= dirtiedGroups;
                mDirtyBindGroupsObjectChangedOrIsDynamic |= dirtiedGroups;

                // Clear any bind groups not in the mask.
                mDirtyBindGroups &= mBindGroupLayoutsMask;
                mDirtyBindGroupsObjectChangedOrIsDynamic &= mBindGroupLayoutsMask;
            } else {
                mDirtyBindGroups = mBindGroupLayoutsMask;
                mDirtyBindGroupsObjectChangedOrIsDynamic = mBindGroupLayoutsMask;
            }
        }

      protected:
        // The Derived class should call this when it applies bind groups.
        void DidApply() {
            // Reset all dirty bind groups. Dirty bind groups not in the bind group layout mask
            // will be dirtied again by the next pipeline change.
            mDirtyBindGroups.reset();
            mDirtyBindGroupsObjectChangedOrIsDynamic.reset();
            mLastAppliedPipelineLayout = mPipelineLayout;
        }

        std::bitset<kMaxBindGroups> mDirtyBindGroups = 0;
        std::bitset<kMaxBindGroups> mDirtyBindGroupsObjectChangedOrIsDynamic = 0;
        std::bitset<kMaxBindGroups> mBindGroupLayoutsMask = 0;
        std::array<BindGroup, kMaxBindGroups> mBindGroups = {};
        std::array<uint32_t, kMaxBindGroups> mDynamicOffsetCounts = {};
        std::array<std::array<DynamicOffset, kMaxBindingsPerGroup>, kMaxBindGroups>
            mDynamicOffsets = {};

        // |mPipelineLayout| is the current pipeline layout set on the command buffer.
        // |mLastAppliedPipelineLayout| is the last pipeline layout for which we applied changes
        // to the bind group bindings.
        PipelineLayoutBase* mPipelineLayout = nullptr;
        PipelineLayoutBase* mLastAppliedPipelineLayout = nullptr;

      private:
        // Vulkan backend use uint32_t as dynamic offsets type, it is not correct.
        // Vulkan should use VkDeviceSize. Dawn vulkan backend has to handle this.
        static void SetDynamicOffsets(uint32_t* data,
                                      uint32_t dynamicOffsetCount,
                                      uint64_t* dynamicOffsets) {
            for (uint32_t i = 0; i < dynamicOffsetCount; ++i) {
                ASSERT(dynamicOffsets[i] <= std::numeric_limits<uint32_t>::max());
                data[i] = static_cast<uint32_t>(dynamicOffsets[i]);
            }
        }

        static void SetDynamicOffsets(uint64_t* data,
                                      uint32_t dynamicOffsetCount,
                                      uint64_t* dynamicOffsets) {
            memcpy(data, dynamicOffsets, sizeof(uint64_t) * dynamicOffsetCount);
        }
    };

}  // namespace dawn_native

#endif  // DAWNNATIVE_BINDGROUPTRACKER_H_
