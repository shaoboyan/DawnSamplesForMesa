// Copyright 2017 The Dawn Authors
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

#include "SampleUtils.h"

#include "utils/ComboRenderPipelineDescriptor.h"
#include "utils/DawnHelpers.h"
#include "utils/SystemUtils.h"

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>

dawn::Device device;
dawn::Queue queue;
dawn::SwapChain swapchain;
dawn::RenderPipeline pipeline[2];
dawn::BindGroup bindGroupFrame;
dawn::BindGroup bindGroupInstance[2];
dawn::Buffer ubo[2];

float RandomFloat(float min, float max) {
    float zeroOne = rand() / float(RAND_MAX);
    return zeroOne * (max - min) + min;
}

size_t kNumTriangles = 10000;
size_t kNumFrames = 600;

struct alignas(kMinDynamicBufferOffsetAlignment) ShaderData {
    float scale;
    float offsetX;
    float offsetY;
    float scalar;
    float scalarOffset;
};

struct FrameAnimData {
    float time;
};

static std::vector<ShaderData> shaderData;

dawn::Texture texture;
dawn::Sampler sampler;

void initTextures() {
    dawn::TextureDescriptor descriptor;
    descriptor.dimension = dawn::TextureDimension::e2D;
    descriptor.size.width = 64;
    descriptor.size.height = 64;
    descriptor.size.depth = 1;
    descriptor.arrayLayerCount = 1;
    descriptor.sampleCount = 1;
    descriptor.format = dawn::TextureFormat::RGBA8Unorm;
    descriptor.mipLevelCount = 1;
    descriptor.usage = dawn::TextureUsage::CopyDst | dawn::TextureUsage::Sampled;
    texture = device.CreateTexture(&descriptor);

    dawn::SamplerDescriptor samplerDesc = utils::GetDefaultSamplerDescriptor();
    sampler = device.CreateSampler(&samplerDesc);

    // Initialize the texture with arbitrary data until we can load images
    std::vector<uint8_t> data(4 * 64 * 64, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i % 253);
    }

    dawn::Buffer stagingBuffer = utils::CreateBufferFromData(
        device, data.data(), static_cast<uint32_t>(data.size()), dawn::BufferUsage::CopySrc);
    dawn::BufferCopyView bufferCopyView = utils::CreateBufferCopyView(stagingBuffer, 0, 0, 0);
    dawn::TextureCopyView textureCopyView = utils::CreateTextureCopyView(texture, 0, 0, {0, 0, 0});
    dawn::Extent3D copySize = {64, 64, 1};

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToTexture(&bufferCopyView, &textureCopyView, &copySize);

    dawn::CommandBuffer copy = encoder.Finish();
    queue.Submit(1, &copy);
}

dawn::PipelineLayout MakeBasicPipelineLayout(
    dawn::Device device,
    std::vector<dawn::BindGroupLayout> bindingInitializer) {
    dawn::PipelineLayoutDescriptor descriptor;

    descriptor.bindGroupLayoutCount = bindingInitializer.size();
    descriptor.bindGroupLayouts = bindingInitializer.data();

    return device.CreatePipelineLayout(&descriptor);
}

void init() {
    device = CreateCppDawnDevice();

    queue = device.CreateQueue();
    swapchain = GetSwapChain(device);
    swapchain.Configure(GetPreferredSwapChainTextureFormat(), dawn::TextureUsage::OutputAttachment,
                        640, 480);

    initTextures();

    dawn::ShaderModule vsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Vertex, R"(
        #version 450

        layout(std140, set = 0, binding = 0) uniform FrameAnimation {
            float time;
        } a;

        layout(std140, set = 1, binding = 0) uniform Constants {
            float scale;
            float offsetX;
            float offsetY;
            float scalar;
            float scalarOffset;
        } c;

        layout(location = 0) out vec4 v_color;

        const vec4 positions[3] = vec4[3](
            vec4( 0.0f,  0.1f, 0.0f, 1.0f),
            vec4(-0.1f, -0.1f, 0.0f, 1.0f),
            vec4( 0.1f, -0.1f, 0.0f, 1.0f)
        );

        const vec4 colors[3] = vec4[3](
            vec4(1.0f, 0.0f, 0.0f, 1.0f),
            vec4(0.0f, 1.0f, 0.0f, 1.0f),
            vec4(0.0f, 0.0f, 1.0f, 1.0f)
        );

        void main() {
            vec4 position = positions[gl_VertexIndex];
            vec4 color = colors[gl_VertexIndex];

            float fade = mod(c.scalarOffset + a.time * c.scalar / 10.0, 1.0);
            if (fade < 0.5) {
                fade = fade * 2.0;
            } else {
                fade = (1.0 - fade) * 2.0;
            }
            float xpos = position.x * c.scale;
            float ypos = position.y * c.scale;
            float angle = 3.14159 * 2.0 * fade;
            float xrot = xpos * cos(angle) - ypos * sin(angle);
            float yrot = xpos * sin(angle) + ypos * cos(angle);
            xpos = xrot + c.offsetX;
            ypos = yrot + c.offsetY;
            v_color = vec4(fade, 1.0 - fade, 0.0, 1.0) + color;
            gl_Position = vec4(xpos, ypos, 0.0, 1.0);
        })");

    dawn::ShaderModule fsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Fragment, R"(
        #version 450
        layout(location = 0) out vec4 fragColor;
        layout(location = 0) in vec4 v_color;

        layout(set = 1, binding = 1) uniform sampler mySampler;
        layout(set = 1, binding = 2) uniform texture2D myTexture;

        void main() {
            //fragColor = v_color;
            fragColor = v_color * texture(sampler2D(myTexture, mySampler), fragColor.xy);
        })");

    dawn::BindGroupLayout bgl_frame = utils::MakeBindGroupLayout(device, {
        {0, dawn::ShaderStage::Vertex, dawn::BindingType::UniformBuffer, false}
    });

    dawn::BindGroupLayout bgl0 = utils::MakeBindGroupLayout(device, {
        {0, dawn::ShaderStage::Vertex, dawn::BindingType::UniformBuffer, true},
    });

    auto bgl1 = utils::MakeBindGroupLayout(device, {
        {0, dawn::ShaderStage::Vertex, dawn::BindingType::UniformBuffer, true},
        {1, dawn::ShaderStage::Fragment, dawn::BindingType::Sampler},
        {2, dawn::ShaderStage::Fragment, dawn::BindingType::SampledTexture},
    });

    //dawn::PipelineLayout pipelineLayout = MakeBasicPipelineLayout(device, {bgl0, bgl1});
    dawn::PipelineLayout pipelineLayout = MakeBasicPipelineLayout(device, {bgl_frame, bgl1});

    utils::ComboRenderPipelineDescriptor descriptor(device);
    descriptor.layout = pipelineLayout;
    descriptor.vertexStage.module = vsModule;
    descriptor.cFragmentStage.module = fsModule;
    descriptor.cColorStates[0].format = GetPreferredSwapChainTextureFormat();

    pipeline[0] = device.CreateRenderPipeline(&descriptor);
    pipeline[1] = device.CreateRenderPipeline(&descriptor);

    shaderData.resize(kNumTriangles);
    for (auto& data : shaderData) {
        data.scale = RandomFloat(0.2f, 0.4f);
        data.offsetX = RandomFloat(-0.9f, 0.9f);
        data.offsetY = RandomFloat(-0.9f, 0.9f);
        data.scalar = RandomFloat(0.5f, 2.0f);
        data.scalarOffset = RandomFloat(0.0f, 10.0f);
    }

    dawn::BufferDescriptor bufferDesc0;
    bufferDesc0.size = sizeof(FrameAnimData);
    bufferDesc0.usage = dawn::BufferUsage::CopyDst | dawn::BufferUsage::Uniform;
    ubo[0] = device.CreateBuffer(&bufferDesc0);

    dawn::BufferDescriptor bufferDesc1;
    bufferDesc1.size = kNumTriangles * sizeof(ShaderData);
    bufferDesc1.usage = dawn::BufferUsage::CopyDst | dawn::BufferUsage::Uniform;
    ubo[1] = device.CreateBuffer(&bufferDesc1);
    ubo[1].SetSubData(0, kNumTriangles * sizeof(ShaderData), shaderData.data());

    bindGroupFrame =
        utils::MakeBindGroup(device, bgl_frame, {
            {0, ubo[0], 0, sizeof(FrameAnimData)}
        });

    dawn::TextureView view1 = texture.CreateView();
    dawn::TextureView view2 = texture.CreateView();
    
    bindGroupInstance[0] = utils::MakeBindGroup(device, bgl1, {
        {0, ubo[1], 0, sizeof(ShaderData)},
        {1, sampler},
        {2, view1},
    });
    
    bindGroupInstance[1] = utils::MakeBindGroup(device, bgl1, {
        {0, ubo[1], 0, sizeof(ShaderData)},
        {1, sampler},
        {2, view2}
    });
}

void frame() {
    dawn::Texture backbuffer = swapchain.GetNextTexture();

    static int f = 0;
    f++;

    float time = f / 60.0f;
    ubo[0].SetSubData(0, sizeof(time), &time);

    utils::ComboRenderPassDescriptor renderPass({backbuffer.CreateView()});
    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    {
        dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass);
        

        for (size_t i = 0; i < kNumTriangles; i++) {
            pass.SetPipeline(pipeline[i&1]);
            pass.SetBindGroup(0, bindGroupFrame, 0, nullptr);
            uint64_t offset = i * sizeof(ShaderData);
            pass.SetBindGroup(1, bindGroupInstance[i&1], 1, &offset);
            pass.Draw(3, 1, 0, 0);
        }

        pass.EndPass();
    }

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
    swapchain.Present(backbuffer);
    DoFlush();
}

int main(int argc, const char* argv[]) {
    if (!InitSample(argc, argv)) {
        return 1;
    }
    char* pNext;
    for (int i = 1; i < argc; i++) {
        if (std::string("-t") == argv[i]) {
	    kNumTriangles = strtol(argv[i++ + 1], &pNext, 10);
	    printf("kNumTriangles = %lu\n", kNumTriangles);
	    continue;
	}
        if (std::string("-f") == argv[i]) {
	    kNumFrames = strtol(argv[ i++ + 1], &pNext, 10);
	    printf("kNumFrames = %lu\n", kNumFrames);
	    continue;
	}
    }

    init();

    for (size_t f = 0; f < kNumFrames && !ShouldQuit(); f++) {
        frame();
    }

    // TODO release stuff
}
