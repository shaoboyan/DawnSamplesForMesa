// Copyright 2018 The Dawn Authors
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

#include "dawn_native/vulkan/SwapChainVk.h"

#include "dawn_native/vulkan/DeviceVk.h"
#include "dawn_native/vulkan/TextureVk.h"

namespace dawn_native { namespace vulkan {

    // static
    SwapChain* SwapChain::Create(Device* device, const SwapChainDescriptor* descriptor) {
        return new SwapChain(device, descriptor);
    }

    SwapChain::SwapChain(Device* device, const SwapChainDescriptor* descriptor)
        : SwapChainBase(device, descriptor) {
        const auto& im = GetImplementation();
        DawnWSIContextVulkan wsiContext = {};
        im.Init(im.userData, &wsiContext);

        ASSERT(im.textureUsage != DAWN_TEXTURE_USAGE_NONE);
        mTextureUsage = static_cast<dawn::TextureUsage>(im.textureUsage);
    }

    SwapChain::~SwapChain() {
    }

    TextureBase* SwapChain::GetNextTextureImpl(const TextureDescriptor* descriptor) {
        const auto& im = GetImplementation();
        DawnSwapChainNextTexture next = {};
        DawnSwapChainError error = im.GetNextTexture(im.userData, &next);

        if (error) {
            GetDevice()->HandleError(dawn::ErrorType::Unknown, error);
            return nullptr;
        }

        VkImage nativeTexture = VkImage::CreateFromU64(next.texture.u64);
        return new Texture(ToBackend(GetDevice()), descriptor, nativeTexture);
    }

    MaybeError SwapChain::OnBeforePresent(TextureBase* texture) {
        Device* device = ToBackend(GetDevice());

        // Perform the necessary pipeline barriers for the texture to be used with the usage
        // requested by the implementation.
        CommandRecordingContext* recordingContext = device->GetPendingRecordingContext();
        ToBackend(texture)->TransitionUsageNow(recordingContext, mTextureUsage);

        DAWN_TRY(device->SubmitPendingCommands());

        return {};
    }

}}  // namespace dawn_native::vulkan
