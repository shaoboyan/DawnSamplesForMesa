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

#include "common/Assert.h"
#include "common/Constants.h"
#include "common/Math.h"
#include "tests/DawnTest.h"
#include "utils/ComboRenderPipelineDescriptor.h"
#include "utils/DawnHelpers.h"

constexpr static unsigned int kRTSize = 8;

class BindGroupTests : public DawnTest {
protected:
    dawn::CommandBuffer CreateSimpleComputeCommandBuffer(
            const dawn::ComputePipeline& pipeline, const dawn::BindGroup& bindGroup) {
        dawn::CommandEncoder encoder = device.CreateCommandEncoder();
        dawn::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bindGroup);
        pass.Dispatch(1, 1, 1);
        pass.EndPass();
        return encoder.Finish();
    }

    dawn::PipelineLayout MakeBasicPipelineLayout(
        dawn::Device device,
        std::vector<dawn::BindGroupLayout> bindingInitializer) const {
        dawn::PipelineLayoutDescriptor descriptor;

        descriptor.bindGroupLayoutCount = bindingInitializer.size();
        descriptor.bindGroupLayouts = bindingInitializer.data();

        return device.CreatePipelineLayout(&descriptor);
    }

    dawn::ShaderModule MakeSimpleVSModule() const {
        return utils::CreateShaderModule(device, utils::SingleShaderStage::Vertex, R"(
        #version 450
        void main() {
            const vec2 pos[3] = vec2[3](vec2(-1.f, 1.f), vec2(1.f, 1.f), vec2(-1.f, -1.f));
            gl_Position = vec4(pos[gl_VertexIndex], 0.f, 1.f);
        })");
    }

    dawn::ShaderModule MakeFSModule(std::vector<dawn::BindingType> bindingTypes) const {
        ASSERT(bindingTypes.size() <= kMaxBindGroups);

        std::ostringstream fs;
        fs << R"(
        #version 450
        layout(location = 0) out vec4 fragColor;
        )";

        for (size_t i = 0; i < bindingTypes.size(); ++i) {
            switch (bindingTypes[i]) {
                case dawn::BindingType::UniformBuffer:
                    fs << "layout (std140, set = " << i << ", binding = 0) uniform UniformBuffer" << i << R"( {
                        vec4 color;
                    } buffer)" << i << ";\n";
                    break;
                case dawn::BindingType::StorageBuffer:
                    fs << "layout (std430, set = " << i << ", binding = 0) buffer StorageBuffer" << i << R"( {
                        vec4 color;
                    } buffer)" << i << ";\n";
                    break;
                default:
                    UNREACHABLE();
            }
        }

        fs << R"(
        void main() {
            fragColor = vec4(0.0);
        )";
        for (size_t i = 0; i < bindingTypes.size(); ++i) {
            fs << "fragColor += buffer" << i << ".color;\n";
        }
        fs << "}\n";

        return utils::CreateShaderModule(device, utils::SingleShaderStage::Fragment, fs.str().c_str());
    }

    dawn::RenderPipeline MakeTestPipeline(
        const utils::BasicRenderPass& renderPass,
        std::vector<dawn::BindingType> bindingTypes,
        std::vector<dawn::BindGroupLayout> bindGroupLayouts) {

        dawn::ShaderModule vsModule = MakeSimpleVSModule();
        dawn::ShaderModule fsModule = MakeFSModule(bindingTypes);

        dawn::PipelineLayout pipelineLayout = MakeBasicPipelineLayout(device, bindGroupLayouts);

        utils::ComboRenderPipelineDescriptor pipelineDescriptor(device);
        pipelineDescriptor.layout = pipelineLayout;
        pipelineDescriptor.vertexStage.module = vsModule;
        pipelineDescriptor.cFragmentStage.module = fsModule;
        pipelineDescriptor.cColorStates[0].format = renderPass.colorFormat;
        pipelineDescriptor.cColorStates[0].colorBlend.operation = dawn::BlendOperation::Add;
        pipelineDescriptor.cColorStates[0].colorBlend.srcFactor = dawn::BlendFactor::One;
        pipelineDescriptor.cColorStates[0].colorBlend.dstFactor = dawn::BlendFactor::One;

        return device.CreateRenderPipeline(&pipelineDescriptor);
    }
};

// Test a bindgroup reused in two command buffers in the same call to queue.Submit().
// This test passes by not asserting or crashing.
TEST_P(BindGroupTests, ReusedBindGroupSingleSubmit) {
    dawn::BindGroupLayout bgl = utils::MakeBindGroupLayout(
        device, {
                    {0, dawn::ShaderStage::Compute, dawn::BindingType::UniformBuffer},
                });
    dawn::PipelineLayout pl = utils::MakeBasicPipelineLayout(device, &bgl);

    const char* shader = R"(
        #version 450
        layout(std140, set = 0, binding = 0) uniform Contents {
            float f;
        } contents;
        void main() {
        }
    )";

    dawn::ShaderModule module =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Compute, shader);

    dawn::ComputePipelineDescriptor cpDesc;
    cpDesc.layout = pl;
    cpDesc.computeStage.module = module;
    cpDesc.computeStage.entryPoint = "main";
    dawn::ComputePipeline cp = device.CreateComputePipeline(&cpDesc);

    dawn::BufferDescriptor bufferDesc;
    bufferDesc.size = sizeof(float);
    bufferDesc.usage = dawn::BufferUsage::CopyDst | dawn::BufferUsage::Uniform;
    dawn::Buffer buffer = device.CreateBuffer(&bufferDesc);
    dawn::BindGroup bindGroup = utils::MakeBindGroup(device, bgl, {{0, buffer, 0, sizeof(float)}});

    dawn::CommandBuffer cb[2];
    cb[0] = CreateSimpleComputeCommandBuffer(cp, bindGroup);
    cb[1] = CreateSimpleComputeCommandBuffer(cp, bindGroup);
    queue.Submit(2, cb);
}

// Test a bindgroup containing a UBO which is used in both the vertex and fragment shader.
// It contains a transformation matrix for the VS and the fragment color for the FS.
// These must result in different register offsets in the native APIs.
TEST_P(BindGroupTests, ReusedUBO) {
    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    dawn::ShaderModule vsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Vertex, R"(
        #version 450
        layout (set = 0, binding = 0) uniform vertexUniformBuffer {
            mat2 transform;
        };
        void main() {
            const vec2 pos[3] = vec2[3](vec2(-1.f, 1.f), vec2(1.f, 1.f), vec2(-1.f, -1.f));
            gl_Position = vec4(transform * pos[gl_VertexIndex], 0.f, 1.f);
        })");

    dawn::ShaderModule fsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Fragment, R"(
        #version 450
        layout (set = 0, binding = 1) uniform fragmentUniformBuffer {
            vec4 color;
        };
        layout(location = 0) out vec4 fragColor;
        void main() {
            fragColor = color;
        })");

    dawn::BindGroupLayout bgl = utils::MakeBindGroupLayout(
        device, {
                    {0, dawn::ShaderStage::Vertex, dawn::BindingType::UniformBuffer},
                    {1, dawn::ShaderStage::Fragment, dawn::BindingType::UniformBuffer},
                });
    dawn::PipelineLayout pipelineLayout = utils::MakeBasicPipelineLayout(device, &bgl);

    utils::ComboRenderPipelineDescriptor textureDescriptor(device);
    textureDescriptor.layout = pipelineLayout;
    textureDescriptor.vertexStage.module = vsModule;
    textureDescriptor.cFragmentStage.module = fsModule;
    textureDescriptor.cColorStates[0].format = renderPass.colorFormat;

    dawn::RenderPipeline pipeline = device.CreateRenderPipeline(&textureDescriptor);

    struct Data {
        float transform[8];
        char padding[256 - 8 * sizeof(float)];
        float color[4];
    };
    ASSERT(offsetof(Data, color) == 256);
    constexpr float dummy = 0.0f;
    Data data {
        { 1.f, 0.f, dummy, dummy, 0.f, 1.0f, dummy, dummy },
        { 0 },
        { 0.f, 1.f, 0.f, 1.f },
    };
    dawn::Buffer buffer =
        utils::CreateBufferFromData(device, &data, sizeof(data), dawn::BufferUsage::Uniform);
    dawn::BindGroup bindGroup = utils::MakeBindGroup(device, bgl, {
        {0, buffer, 0, sizeof(Data::transform)},
        {1, buffer, 256, sizeof(Data::color)}
    });

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(3, 1, 0, 0);
    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    RGBA8 filled(0, 255, 0, 255);
    RGBA8 notFilled(0, 0, 0, 0);
    int min = 1, max = kRTSize - 3;
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color,    min, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color,    max, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color,    min, max);
    EXPECT_PIXEL_RGBA8_EQ(notFilled, renderPass.color, max, max);
}

// Test a bindgroup containing a UBO in the vertex shader and a sampler and texture in the fragment shader.
// In D3D12 for example, these different types of bindings end up in different namespaces, but the register
// offsets used must match between the shader module and descriptor range.
TEST_P(BindGroupTests, UBOSamplerAndTexture) {
    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    dawn::ShaderModule vsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Vertex, R"(
        #version 450
        layout (set = 0, binding = 0) uniform vertexUniformBuffer {
            mat2 transform;
        };
        void main() {
            const vec2 pos[3] = vec2[3](vec2(-1.f, 1.f), vec2(1.f, 1.f), vec2(-1.f, -1.f));
            gl_Position = vec4(transform * pos[gl_VertexIndex], 0.f, 1.f);
        })");

    dawn::ShaderModule fsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Fragment, R"(
        #version 450
        layout (set = 0, binding = 1) uniform sampler samp;
        layout (set = 0, binding = 2) uniform texture2D tex;
        layout (location = 0) out vec4 fragColor;
        void main() {
            fragColor = texture(sampler2D(tex, samp), gl_FragCoord.xy);
        })");

    dawn::BindGroupLayout bgl = utils::MakeBindGroupLayout(
        device, {
                    {0, dawn::ShaderStage::Vertex, dawn::BindingType::UniformBuffer},
                    {1, dawn::ShaderStage::Fragment, dawn::BindingType::Sampler},
                    {2, dawn::ShaderStage::Fragment, dawn::BindingType::SampledTexture},
                });
    dawn::PipelineLayout pipelineLayout = utils::MakeBasicPipelineLayout(device, &bgl);

    utils::ComboRenderPipelineDescriptor pipelineDescriptor(device);
    pipelineDescriptor.layout = pipelineLayout;
    pipelineDescriptor.vertexStage.module = vsModule;
    pipelineDescriptor.cFragmentStage.module = fsModule;
    pipelineDescriptor.cColorStates[0].format = renderPass.colorFormat;

    dawn::RenderPipeline pipeline = device.CreateRenderPipeline(&pipelineDescriptor);

    constexpr float dummy = 0.0f;
    constexpr float transform[] = { 1.f, 0.f, dummy, dummy, 0.f, 1.f, dummy, dummy };
    dawn::Buffer buffer = utils::CreateBufferFromData(device, &transform, sizeof(transform),
                                                      dawn::BufferUsage::Uniform);

    dawn::SamplerDescriptor samplerDescriptor;
    samplerDescriptor.minFilter = dawn::FilterMode::Nearest;
    samplerDescriptor.magFilter = dawn::FilterMode::Nearest;
    samplerDescriptor.mipmapFilter = dawn::FilterMode::Nearest;
    samplerDescriptor.addressModeU = dawn::AddressMode::ClampToEdge;
    samplerDescriptor.addressModeV = dawn::AddressMode::ClampToEdge;
    samplerDescriptor.addressModeW = dawn::AddressMode::ClampToEdge;
    samplerDescriptor.lodMinClamp = kLodMin;
    samplerDescriptor.lodMaxClamp = kLodMax;
    samplerDescriptor.compare = dawn::CompareFunction::Never;

    dawn::Sampler sampler = device.CreateSampler(&samplerDescriptor);

    dawn::TextureDescriptor descriptor;
    descriptor.dimension = dawn::TextureDimension::e2D;
    descriptor.size.width = kRTSize;
    descriptor.size.height = kRTSize;
    descriptor.size.depth = 1;
    descriptor.arrayLayerCount = 1;
    descriptor.sampleCount = 1;
    descriptor.format = dawn::TextureFormat::RGBA8Unorm;
    descriptor.mipLevelCount = 1;
    descriptor.usage = dawn::TextureUsage::CopyDst | dawn::TextureUsage::Sampled;
    dawn::Texture texture = device.CreateTexture(&descriptor);
    dawn::TextureView textureView = texture.CreateView();

    int width = kRTSize, height = kRTSize;
    int widthInBytes = width * sizeof(RGBA8);
    widthInBytes = (widthInBytes + 255) & ~255;
    int sizeInBytes = widthInBytes * height;
    int size = sizeInBytes / sizeof(RGBA8);
    std::vector<RGBA8> data = std::vector<RGBA8>(size);
    for (int i = 0; i < size; i++) {
        data[i] = RGBA8(0, 255, 0, 255);
    }
    dawn::Buffer stagingBuffer =
        utils::CreateBufferFromData(device, data.data(), sizeInBytes, dawn::BufferUsage::CopySrc);

    dawn::BindGroup bindGroup = utils::MakeBindGroup(device, bgl, {
        {0, buffer, 0, sizeof(transform)},
        {1, sampler},
        {2, textureView}
    });

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::BufferCopyView bufferCopyView =
        utils::CreateBufferCopyView(stagingBuffer, 0, widthInBytes, 0);
    dawn::TextureCopyView textureCopyView = utils::CreateTextureCopyView(texture, 0, 0, {0, 0, 0});
    dawn::Extent3D copySize = {width, height, 1};
    encoder.CopyBufferToTexture(&bufferCopyView, &textureCopyView, &copySize);
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(3, 1, 0, 0);
    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    RGBA8 filled(0, 255, 0, 255);
    RGBA8 notFilled(0, 0, 0, 0);
    int min = 1, max = kRTSize - 3;
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color,    min, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color,    max, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color,    min, max);
    EXPECT_PIXEL_RGBA8_EQ(notFilled, renderPass.color, max, max);
}

TEST_P(BindGroupTests, MultipleBindLayouts) {
    // Test fails on Metal.
    // https://bugs.chromium.org/p/dawn/issues/detail?id=33
    DAWN_SKIP_TEST_IF(IsMetal());

    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    dawn::ShaderModule vsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Vertex, R"(
        #version 450
        layout (set = 0, binding = 0) uniform vertexUniformBuffer1 {
            mat2 transform1;
        };
        layout (set = 1, binding = 0) uniform vertexUniformBuffer2 {
            mat2 transform2;
        };
        void main() {
            const vec2 pos[3] = vec2[3](vec2(-1.f, 1.f), vec2(1.f, 1.f), vec2(-1.f, -1.f));
            gl_Position = vec4((transform1 + transform2) * pos[gl_VertexIndex], 0.f, 1.f);
        })");

    dawn::ShaderModule fsModule =
        utils::CreateShaderModule(device, utils::SingleShaderStage::Fragment, R"(
        #version 450
        layout (set = 0, binding = 1) uniform fragmentUniformBuffer1 {
            vec4 color1;
        };
        layout (set = 1, binding = 1) uniform fragmentUniformBuffer2 {
            vec4 color2;
        };
        layout(location = 0) out vec4 fragColor;
        void main() {
            fragColor = color1 + color2;
        })");

    dawn::BindGroupLayout layout = utils::MakeBindGroupLayout(
        device, {
                    {0, dawn::ShaderStage::Vertex, dawn::BindingType::UniformBuffer},
                    {1, dawn::ShaderStage::Fragment, dawn::BindingType::UniformBuffer},
                });

    dawn::PipelineLayout pipelineLayout = MakeBasicPipelineLayout(device, {layout, layout});

    utils::ComboRenderPipelineDescriptor textureDescriptor(device);
    textureDescriptor.layout = pipelineLayout;
    textureDescriptor.vertexStage.module = vsModule;
    textureDescriptor.cFragmentStage.module = fsModule;
    textureDescriptor.cColorStates[0].format = renderPass.colorFormat;

    dawn::RenderPipeline pipeline = device.CreateRenderPipeline(&textureDescriptor);

    struct Data {
        float transform[8];
        char padding[256 - 8 * sizeof(float)];
        float color[4];
    };
    ASSERT(offsetof(Data, color) == 256);

    std::vector<Data> data;
    std::vector<dawn::Buffer> buffers;
    std::vector<dawn::BindGroup> bindGroups;

    data.push_back(
        {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0}, {0.0f, 1.0f, 0.0f, 1.0f}});

    data.push_back(
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, {0}, {1.0f, 0.0f, 0.0f, 1.0f}});

    for (int i = 0; i < 2; i++) {
        dawn::Buffer buffer =
            utils::CreateBufferFromData(device, &data[i], sizeof(Data), dawn::BufferUsage::Uniform);
        buffers.push_back(buffer);
        bindGroups.push_back(utils::MakeBindGroup(device, layout,
                                                  {{0, buffers[i], 0, sizeof(Data::transform)},
                                                   {1, buffers[i], 256, sizeof(Data::color)}}));
    }

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroups[0]);
    pass.SetBindGroup(1, bindGroups[1]);
    pass.Draw(3, 1, 0, 0);
    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    RGBA8 filled(255, 255, 0, 255);
    RGBA8 notFilled(0, 0, 0, 0);
    int min = 1, max = kRTSize - 3;
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, max, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, max);
    EXPECT_PIXEL_RGBA8_EQ(notFilled, renderPass.color, max, max);
}

// This test reproduces an out-of-bound bug on D3D12 backends when calling draw command twice with
// one pipeline that has 4 bind group sets in one render pass.
TEST_P(BindGroupTests, DrawTwiceInSamePipelineWithFourBindGroupSets) {
    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    dawn::BindGroupLayout layout = utils::MakeBindGroupLayout(
        device, {{0, dawn::ShaderStage::Fragment, dawn::BindingType::UniformBuffer}});

    dawn::RenderPipeline pipeline =
        MakeTestPipeline(renderPass,
                         {dawn::BindingType::UniformBuffer, dawn::BindingType::UniformBuffer,
                          dawn::BindingType::UniformBuffer, dawn::BindingType::UniformBuffer},
                         {layout, layout, layout, layout});

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);

    pass.SetPipeline(pipeline);

    std::array<float, 4> color = {0.25, 0, 0, 0.25};
    dawn::Buffer uniformBuffer =
        utils::CreateBufferFromData(device, &color, sizeof(color), dawn::BufferUsage::Uniform);
    dawn::BindGroup bindGroup =
        utils::MakeBindGroup(device, layout, {{0, uniformBuffer, 0, sizeof(color)}});

    pass.SetBindGroup(0, bindGroup);
    pass.SetBindGroup(1, bindGroup);
    pass.SetBindGroup(2, bindGroup);
    pass.SetBindGroup(3, bindGroup);
    pass.Draw(3, 1, 0, 0);

    pass.SetPipeline(pipeline);
    pass.Draw(3, 1, 0, 0);
    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    RGBA8 filled(255, 0, 0, 255);
    RGBA8 notFilled(0, 0, 0, 0);
    int min = 1, max = kRTSize - 3;
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, max, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, max);
    EXPECT_PIXEL_RGBA8_EQ(notFilled, renderPass.color, max, max);
}

// Test that bind groups can be set before the pipeline.
TEST_P(BindGroupTests, SetBindGroupBeforePipeline) {
    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    // Create a bind group layout which uses a single uniform buffer.
    dawn::BindGroupLayout layout = utils::MakeBindGroupLayout(
        device, {{0, dawn::ShaderStage::Fragment, dawn::BindingType::UniformBuffer}});

    // Create a pipeline that uses the uniform bind group layout.
    dawn::RenderPipeline pipeline =
        MakeTestPipeline(renderPass, {dawn::BindingType::UniformBuffer}, {layout});

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);

    // Create a bind group with a uniform buffer and fill it with RGBAunorm(1, 0, 0, 1).
    std::array<float, 4> color = {1, 0, 0, 1};
    dawn::Buffer uniformBuffer =
        utils::CreateBufferFromData(device, &color, sizeof(color), dawn::BufferUsage::Uniform);
    dawn::BindGroup bindGroup =
        utils::MakeBindGroup(device, layout, {{0, uniformBuffer, 0, sizeof(color)}});

    // Set the bind group, then the pipeline, and draw.
    pass.SetBindGroup(0, bindGroup);
    pass.SetPipeline(pipeline);
    pass.Draw(3, 1, 0, 0);

    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    // The result should be red.
    RGBA8 filled(255, 0, 0, 255);
    RGBA8 notFilled(0, 0, 0, 0);
    int min = 1, max = kRTSize - 3;
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, max, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, max);
    EXPECT_PIXEL_RGBA8_EQ(notFilled, renderPass.color, max, max);
}

// Test that dynamic bind groups can be set before the pipeline.
TEST_P(BindGroupTests, SetDynamicBindGroupBeforePipeline) {
    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    // Create a bind group layout which uses a single dynamic uniform buffer.
    dawn::BindGroupLayout layout = utils::MakeBindGroupLayout(
        device, {{0, dawn::ShaderStage::Fragment, dawn::BindingType::UniformBuffer, true}});

    // Create a pipeline that uses the dynamic uniform bind group layout for two bind groups.
    dawn::RenderPipeline pipeline = MakeTestPipeline(
        renderPass, {dawn::BindingType::UniformBuffer, dawn::BindingType::UniformBuffer},
        {layout, layout});

    // Prepare data RGBAunorm(1, 0, 0, 0.5) and RGBAunorm(0, 1, 0, 0.5). They will be added in the
    // shader.
    std::array<float, 4> color0 = {1, 0, 0, 0.501};
    std::array<float, 4> color1 = {0, 1, 0, 0.501};

    size_t color1Offset = Align(sizeof(color0), kMinDynamicBufferOffsetAlignment);

    std::vector<uint8_t> data(color1Offset + sizeof(color1));
    memcpy(data.data(), color0.data(), sizeof(color0));
    memcpy(data.data() + color1Offset, color1.data(), sizeof(color1));

    // Create a bind group and uniform buffer with the color data. It will be bound at the offset
    // to each color.
    dawn::Buffer uniformBuffer =
        utils::CreateBufferFromData(device, data.data(), data.size(), dawn::BufferUsage::Uniform);
    dawn::BindGroup bindGroup =
        utils::MakeBindGroup(device, layout, {{0, uniformBuffer, 0, 4 * sizeof(float)}});

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);

    // Set the first dynamic bind group.
    uint64_t dynamicOffset = 0;
    pass.SetBindGroup(0, bindGroup, 1, &dynamicOffset);

    // Set the second dynamic bind group.
    dynamicOffset = color1Offset;
    pass.SetBindGroup(1, bindGroup, 1, &dynamicOffset);

    // Set the pipeline and draw.
    pass.SetPipeline(pipeline);
    pass.Draw(3, 1, 0, 0);

    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    // The result should be RGBAunorm(1, 0, 0, 0.5) + RGBAunorm(0, 1, 0, 0.5)
    RGBA8 filled(255, 255, 0, 255);
    RGBA8 notFilled(0, 0, 0, 0);
    int min = 1, max = kRTSize - 3;
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, max, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, max);
    EXPECT_PIXEL_RGBA8_EQ(notFilled, renderPass.color, max, max);
}

// Test that bind groups set for one pipeline are still set when the pipeline changes.
TEST_P(BindGroupTests, BindGroupsPersistAfterPipelineChange) {
    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    // Create a bind group layout which uses a single dynamic uniform buffer.
    dawn::BindGroupLayout uniformLayout = utils::MakeBindGroupLayout(
        device, {{0, dawn::ShaderStage::Fragment, dawn::BindingType::UniformBuffer, true}});

    // Create a bind group layout which uses a single dynamic storage buffer.
    dawn::BindGroupLayout storageLayout = utils::MakeBindGroupLayout(
        device, {{0, dawn::ShaderStage::Fragment, dawn::BindingType::StorageBuffer, true}});

    // Create a pipeline which uses the uniform buffer and storage buffer bind groups.
    dawn::RenderPipeline pipeline0 = MakeTestPipeline(
        renderPass, {dawn::BindingType::UniformBuffer, dawn::BindingType::StorageBuffer},
        {uniformLayout, storageLayout});

    // Create a pipeline which uses the uniform buffer bind group twice.
    dawn::RenderPipeline pipeline1 = MakeTestPipeline(
        renderPass, {dawn::BindingType::UniformBuffer, dawn::BindingType::UniformBuffer},
        {uniformLayout, uniformLayout});

    // Prepare data RGBAunorm(1, 0, 0, 0.5) and RGBAunorm(0, 1, 0, 0.5). They will be added in the
    // shader.
    std::array<float, 4> color0 = {1, 0, 0, 0.5};
    std::array<float, 4> color1 = {0, 1, 0, 0.5};

    size_t color1Offset = Align(sizeof(color0), kMinDynamicBufferOffsetAlignment);

    std::vector<uint8_t> data(color1Offset + sizeof(color1));
    memcpy(data.data(), color0.data(), sizeof(color0));
    memcpy(data.data() + color1Offset, color1.data(), sizeof(color1));

    // Create a bind group and uniform buffer with the color data. It will be bound at the offset
    // to each color.
    dawn::Buffer uniformBuffer =
        utils::CreateBufferFromData(device, data.data(), data.size(), dawn::BufferUsage::Uniform);
    dawn::BindGroup bindGroup =
        utils::MakeBindGroup(device, uniformLayout, {{0, uniformBuffer, 0, 4 * sizeof(float)}});

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);

    // Set the first pipeline (uniform, storage).
    pass.SetPipeline(pipeline0);

    // Set the first bind group at a dynamic offset.
    // This bind group matches the slot in the pipeline layout.
    uint64_t dynamicOffset = 0;
    pass.SetBindGroup(0, bindGroup, 1, &dynamicOffset);

    // Set the second bind group at a dynamic offset.
    // This bind group does not match the slot in the pipeline layout.
    dynamicOffset = color1Offset;
    pass.SetBindGroup(1, bindGroup, 1, &dynamicOffset);

    // Set the second pipeline (uniform, uniform).
    // Both bind groups match the pipeline.
    // They should persist and not need to be bound again.
    pass.SetPipeline(pipeline1);
    pass.Draw(3, 1, 0, 0);

    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    // The result should be RGBAunorm(1, 0, 0, 0.5) + RGBAunorm(0, 1, 0, 0.5)
    RGBA8 filled(255, 255, 0, 255);
    RGBA8 notFilled(0, 0, 0, 0);
    int min = 1, max = kRTSize - 3;
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, max, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, max);
    EXPECT_PIXEL_RGBA8_EQ(notFilled, renderPass.color, max, max);
}

// Do a successful draw. Then, change the pipeline and one bind group.
// Draw to check that the all bind groups are set.
TEST_P(BindGroupTests, DrawThenChangePipelineAndBindGroup) {
    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    // Create a bind group layout which uses a single dynamic uniform buffer.
    dawn::BindGroupLayout uniformLayout = utils::MakeBindGroupLayout(
        device, {{0, dawn::ShaderStage::Fragment, dawn::BindingType::UniformBuffer, true}});

    // Create a bind group layout which uses a single dynamic storage buffer.
    dawn::BindGroupLayout storageLayout = utils::MakeBindGroupLayout(
        device, {{0, dawn::ShaderStage::Fragment, dawn::BindingType::StorageBuffer, true}});

    // Create a pipeline with pipeline layout (uniform, uniform, storage).
    dawn::RenderPipeline pipeline0 = MakeTestPipeline(
        renderPass, {dawn::BindingType::UniformBuffer, dawn::BindingType::UniformBuffer, dawn::BindingType::StorageBuffer},
        {uniformLayout, uniformLayout, storageLayout});

    // Create a pipeline with pipeline layout (uniform, storage, storage).
    dawn::RenderPipeline pipeline1 = MakeTestPipeline(
        renderPass, {dawn::BindingType::UniformBuffer, dawn::BindingType::StorageBuffer, dawn::BindingType::StorageBuffer },
        {uniformLayout, storageLayout, storageLayout});

    // Prepare color data.
    // The first draw will use { color0, color1, color2 }.
    // The second draw will use { color0, color3, color2 }.
    // The pipeline uses additive color blending so the result of two draws should be
    // { 2 * color0 + color1 + color2 + color3} = RGBAunorm(1, 1, 1, 1)
    std::array<float, 4> color0 = {0.501, 0, 0, 0};
    std::array<float, 4> color1 = {0, 1, 0, 0};
    std::array<float, 4> color2 = {0, 0, 0, 1};
    std::array<float, 4> color3 = {0, 0, 1, 0};

    size_t color1Offset = Align(sizeof(color0), kMinDynamicBufferOffsetAlignment);
    size_t color2Offset = Align(color1Offset + sizeof(color1), kMinDynamicBufferOffsetAlignment);
    size_t color3Offset = Align(color2Offset + sizeof(color2), kMinDynamicBufferOffsetAlignment);

    std::vector<uint8_t> data(color3Offset + sizeof(color3), 0);
    memcpy(data.data(), color0.data(), sizeof(color0));
    memcpy(data.data() + color1Offset, color1.data(), sizeof(color1));
    memcpy(data.data() + color2Offset, color2.data(), sizeof(color2));
    memcpy(data.data() + color3Offset, color3.data(), sizeof(color3));

    // Create a uniform and storage buffer bind groups to bind the color data.
    dawn::Buffer uniformBuffer =
        utils::CreateBufferFromData(device, data.data(), data.size(), dawn::BufferUsage::Uniform);

    dawn::Buffer storageBuffer =
        utils::CreateBufferFromData(device, data.data(), data.size(), dawn::BufferUsage::Storage);

    dawn::BindGroup uniformBindGroup =
        utils::MakeBindGroup(device, uniformLayout, {{0, uniformBuffer, 0, 4 * sizeof(float)}});
    dawn::BindGroup storageBindGroup =
        utils::MakeBindGroup(device, storageLayout, {{0, storageBuffer, 0, 4 * sizeof(float)}});

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);

    // Set the pipeline to (uniform, uniform, storage)
    pass.SetPipeline(pipeline0);

    // Set the first bind group to color0 in the dynamic uniform buffer.
    uint64_t dynamicOffset = 0;
    pass.SetBindGroup(0, uniformBindGroup, 1, &dynamicOffset);

    // Set the first bind group to color1 in the dynamic uniform buffer.
    dynamicOffset = color1Offset;
    pass.SetBindGroup(1, uniformBindGroup, 1, &dynamicOffset);

    // Set the first bind group to color2 in the dynamic storage buffer.
    dynamicOffset = color2Offset;
    pass.SetBindGroup(2, storageBindGroup, 1, &dynamicOffset);

    pass.Draw(3, 1, 0, 0);

    // Set the pipeline to (uniform, storage, storage)
    //  - The first bind group should persist (inherited on some backends)
    //  - The second bind group needs to be set again to pass validation.
    //    It changed from uniform to storage.
    //  - The third bind group should persist. It should be set again by the backend internally.
    pass.SetPipeline(pipeline1);

    // Set the second bind group to color3 in the dynamic storage buffer.
    dynamicOffset = color3Offset;
    pass.SetBindGroup(1, storageBindGroup, 1, &dynamicOffset);

    pass.Draw(3, 1, 0, 0);
    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    RGBA8 filled(255, 255, 255, 255);
    RGBA8 notFilled(0, 0, 0, 0);
    int min = 1, max = kRTSize - 3;
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, max, min);
    EXPECT_PIXEL_RGBA8_EQ(filled, renderPass.color, min, max);
    EXPECT_PIXEL_RGBA8_EQ(notFilled, renderPass.color, max, max);
}

// Test that visibility of bindings in BindGroupLayout can be none
// This test passes by not asserting or crashing.
TEST_P(BindGroupTests, BindGroupLayoutVisibilityCanBeNone) {
    utils::BasicRenderPass renderPass = utils::CreateBasicRenderPass(device, kRTSize, kRTSize);

    dawn::BindGroupLayoutBinding binding = {0, dawn::ShaderStage::None,
                                            dawn::BindingType::UniformBuffer};
    dawn::BindGroupLayoutDescriptor descriptor;
    descriptor.bindingCount = 1;
    descriptor.bindings = &binding;
    dawn::BindGroupLayout layout = device.CreateBindGroupLayout(&descriptor);

    dawn::RenderPipeline pipeline = MakeTestPipeline(renderPass, {}, {layout});

    std::array<float, 4> color = {1, 0, 0, 1};
    dawn::Buffer uniformBuffer =
        utils::CreateBufferFromData(device, &color, sizeof(color), dawn::BufferUsage::Uniform);
    dawn::BindGroup bindGroup =
        utils::MakeBindGroup(device, layout, {{0, uniformBuffer, 0, sizeof(color)}});

    dawn::CommandEncoder encoder = device.CreateCommandEncoder();
    dawn::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass.renderPassInfo);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(3, 1, 0, 0);
    pass.EndPass();

    dawn::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
}

DAWN_INSTANTIATE_TEST(BindGroupTests, D3D12Backend, MetalBackend, OpenGLBackend, VulkanBackend);
