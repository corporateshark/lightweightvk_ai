/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

#include <random>

constexpr uint32_t kNumBlades = 200000;
constexpr float kFieldSize = 20.0f;
constexpr uint32_t kWindTexSize = 256;
constexpr uint32_t kBladesPerStrip = 7;

// clang-format off

// Compute shader: generates a 2D wind displacement texture using FBM noise
const char* codeWindCompute = R"(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 2, rgba16f) uniform image2D kTextures2DInOut[];

layout(push_constant) uniform constants {
  float time;
  float windStrength;
  float windFreq;
  float windSpeed;
  float gustStrength;
  float gustFreq;
  uint texOut;
  uint texSize;
} pc;

// smooth gradient noise (quintic interpolation, no sin())
vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float gradientNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  // quintic Hermite for C2 continuity (no jerky derivative discontinuities)
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

  return mix(mix(dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
                 dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
             mix(dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
                 dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p) {
  float value = 0.0;
  float amplitude = 0.5;
  float frequency = 1.0;
  for (int i = 0; i < 5; i++) {
    value += amplitude * gradientNoise(p * frequency);
    frequency *= 2.0;
    amplitude *= 0.5;
  }
  return value;
}

void main() {
  ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  if (pos.x >= int(pc.texSize) || pos.y >= int(pc.texSize))
    return;

  vec2 uv = vec2(pos) / float(pc.texSize);

  // slow-scrolling layered wind: primary direction + secondary cross-flow
  float t = pc.time * pc.windSpeed * 0.15;
  vec2 windUV = uv * pc.windFreq + vec2(t, t * 0.6);

  // domain warp: distort sampling coords with high-freq noise scaled by wind strength
  // produces turbulent eddies at strong wind (Ghost of Tsushima technique)
  float warpScale = 0.4 * pc.windStrength;
  vec2 warpUV = uv * pc.windFreq * 3.0 + vec2(pc.time * 0.2, pc.time * 0.15);
  vec2 warp = vec2(gradientNoise(warpUV), gradientNoise(warpUV + vec2(7.3, 2.9)));
  windUV += warp * warpScale;

  float windX = fbm(windUV);
  float windZ = fbm(windUV + vec2(5.2, 1.3));

  // broad rolling gusts (low frequency, slow movement)
  vec2 gustUV = uv * pc.gustFreq + vec2(pc.time * 0.07, pc.time * 0.09);
  float gust = fbm(gustUV + vec2(17.0, 31.0));
  gust = smoothstep(-0.3, 0.6, gust); // soft gust envelope

  // base directional wind + gust overlay
  vec2 wind = vec2(windX, windZ) * pc.windStrength + vec2(gust, gust * 0.5) * pc.gustStrength;

  imageStore(kTextures2DInOut[pc.texOut], pos, vec4(wind, 0.0, 1.0));
}
)";

// Ground plane vertex shader: 4-vertex triangle strip quad at Y=0
const char* codeGroundVS = R"(
layout (location=0) out vec2 v_WorldXZ;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

void main() {
  float s = pc.perFrame.fieldSize;
  // triangle strip: 0=(-s,0,-s), 1=(s,0,-s), 2=(-s,0,s), 3=(s,0,s)
  vec2 pos = vec2((gl_VertexIndex & 1) * 2.0 - 1.0, (gl_VertexIndex >> 1) * 2.0 - 1.0) * s;
  v_WorldXZ = pos;
  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(pos.x, 0.0, pos.y, 1.0);
}
)";

// Ground plane fragment shader
const char* codeGroundFS = R"(
layout (location=0) in vec2 v_WorldXZ;
layout (location=0) out vec4 out_FragColor;

void main() {
  // earthy brown with subtle variation
  float n = fract(sin(dot(floor(v_WorldXZ * 4.0), vec2(12.9898, 78.233))) * 43758.5453);
  vec3 brown = mix(vec3(0.28, 0.20, 0.10), vec3(0.35, 0.25, 0.12), n);
  out_FragColor = vec4(brown, 1.0);
}
)";

// Grass vertex shader: instanced 7-vertex triangle strips
// Manually declares bindless texture arrays for vertex-stage texture sampling
const char* codeGrassVS = R"(
layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler   kSamplers[];

layout (location=0) out vec3 v_Color;
layout (location=1) out float v_AO;
layout (location=2) out vec3 v_Normal;
layout (location=3) out float v_BendAmount;

struct GrassBlade {
  float posX, posZ;
  float height, width;
  float lean, phase;
  float stiffness;
  float colorVariation;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
};

layout(std430, buffer_reference) readonly buffer BladeData {
  GrassBlade blades[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  BladeData bladeData;
} pc;

void main() {
  GrassBlade blade = pc.bladeData.blades[gl_InstanceIndex];

  // 7 vertices form a tapered blade: wide at base, pointed at tip
  // Pairs: (0,1), (2,3), (4,5) then 6 is the tip
  int segment = gl_VertexIndex / 2;
  int side = gl_VertexIndex % 2;
  float t; // 0 at base, 1 at tip
  float widthScale;

  if (gl_VertexIndex == 6) {
    // tip vertex
    t = 1.0;
    widthScale = 0.0;
  } else {
    t = float(segment) / 3.0;
    widthScale = (side == 0) ? -1.0 : 1.0;
  }

  float bladeWidth = blade.width * (1.0 - t * 0.8); // taper

  // sample wind texture
  vec2 fieldUV = vec2(blade.posX, blade.posZ) / (2.0 * pc.perFrame.fieldSize) + 0.5;
  vec4 windSample = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.perFrame.windTex], kSamplers[pc.perFrame.windSamp])),
    fieldUV);
  vec2 windDisplacement = windSample.xy;

  // wind-alignment attenuation: blades facing into/away from wind bend less
  // than blades perpendicular to it (Jahrmann-Wimmer model)
  vec2 bladeDir = normalize(vec2(cos(blade.phase), sin(blade.phase)));
  float windLen = length(windDisplacement);
  float alignment = (windLen > 0.001)
    ? 1.0 - 0.6 * abs(dot(bladeDir, windDisplacement / windLen))
    : 1.0;

  // --- main bending: smooth cubic arc from wind ---
  float bendFactor = t * t * t * alignment / blade.stiffness;

  // --- detail flutter: 4 SmoothTriangleWaves at different frequencies (Crysis model) ---
  // per-blade phase from world position + random phase; per-vertex variation from segment
  float objPhase = blade.posX * 0.7 + blade.posZ * 1.3 + blade.phase;
  float vtxPhase = float(segment) * 0.37 + objPhase;
  float time = pc.perFrame.time;

  // SmoothTriangleWave: abs(frac(x+0.5)*2-1) smoothed by x*x*(3-2*x)
  vec4 wavesIn = vec4(vtxPhase + time, objPhase + time,
                      vtxPhase + time, objPhase + time);
  vec4 freqs = vec4(1.975, 0.793, 0.375, 0.193);
  vec4 waves = fract(wavesIn * freqs) * 2.0 - 1.0;       // triangle wave
  waves = abs(waves);
  waves = waves * waves * (3.0 - 2.0 * waves) * 2.0 - 1.0; // smooth + remap to [-1,1]

  // combine: x-pair for lateral flutter, y-pair for vertical bob
  float detailLateral = (waves.x + waves.y) * 0.015;
  float detailVertical = (waves.z + waves.w) * 0.008;
  // only upper segments flutter (scale by t^2), attenuated by stiffness
  float detailAtten = t * t / blade.stiffness;

  // billboard: orient blade width to face the camera (XZ plane only)
  mat4 view = pc.perFrame.view;
  vec2 camRightXZ = normalize(vec2(view[0][0], view[2][0]));
  float halfW = widthScale * bladeWidth * 0.5;

  // --- arc-based bending ---
  // compute total horizontal bend at this vertex's height as an arc angle
  // wind bend + lean + detail flutter all contribute to the bend direction
  vec2 totalBendXZ = windDisplacement * bendFactor
                   + vec2(blade.lean * 0.3) * alignment
                   + vec2(detailLateral, detailLateral * 0.7) * detailAtten;
  float bendMag = length(totalBendXZ);

  // convert displacement to arc angle (clamp to prevent over-rotation)
  // angle = displacement / radius, where radius = blade height
  float h = blade.height;
  float arcAngle = clamp(bendMag / max(h, 0.01), 0.0, 1.2) * t;

  // bend direction in XZ plane
  vec2 bendDir = (bendMag > 0.001) ? totalBendXZ / bendMag : vec2(1.0, 0.0);

  // rotate the vertex along the bend arc: base stays fixed, upper verts arc over
  // sin(angle) = horizontal offset, cos(angle) = height factor
  float arcY = cos(arcAngle) * t * h;
  float arcH = sin(arcAngle) * t * h;

  vec3 pos;
  pos.x = blade.posX + halfW * camRightXZ.x + bendDir.x * arcH;
  pos.y = arcY;
  pos.z = blade.posZ + halfW * camRightXZ.y + bendDir.y * arcH;

  // detail vertical bob
  pos.y += detailVertical * detailAtten;

  // color: dark green at base, bright green at tip with variation
  vec3 baseColor = mix(vec3(0.05, 0.15, 0.02), vec3(0.10, 0.20, 0.03), blade.colorVariation);
  vec3 tipColor  = mix(vec3(0.30, 0.60, 0.10), vec3(0.50, 0.75, 0.20), blade.colorVariation);
  v_Color = mix(baseColor, tipColor, t);
  v_AO = mix(0.4, 1.0, t); // ambient occlusion: darker at base
  v_BendAmount = length(windDisplacement) * bendFactor;
  // normal faces camera in XZ, with vertical and wind tilt
  vec3 faceNormal = vec3(-camRightXZ.y, 0.0, camRightXZ.x); // perpendicular to camRight in XZ
  v_Normal = normalize(faceNormal * widthScale * 0.3 + vec3(windDisplacement.x * 0.2, 1.0, windDisplacement.y * 0.2));

  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(pos, 1.0);
}
)";

// Grass fragment shader
const char* codeGrassFS = R"(
layout (location=0) in vec3 v_Color;
layout (location=1) in float v_AO;
layout (location=2) in vec3 v_Normal;
layout (location=3) in float v_BendAmount;
layout (location=0) out vec4 out_FragColor;

void main() {
  // simple directional light
  vec3 lightDir = normalize(vec3(0.4, 1.0, 0.3));
  float NdotL = max(dot(normalize(v_Normal), lightDir), 0.0);
  float lighting = 0.3 + 0.7 * NdotL; // ambient + diffuse

  // wind-driven AO: bent blades darken (self-shadowing)
  float windAO = 1.0 - clamp(v_BendAmount * 0.6, 0.0, 0.35);

  vec3 color = v_Color * lighting * v_AO * windAO;
  out_FragColor = vec4(color, 1.0);
}
)";

// clang-format on

struct GrassBlade {
  float posX, posZ;
  float height, width;
  float lean, phase;
  float stiffness;
  float colorVariation;
};

struct PerFrame {
  mat4 proj;
  mat4 view;
  float time;
  uint32_t windTex;
  uint32_t windSamp;
  float fieldSize;
};

struct WindParams {
  float time;
  float windStrength;
  float windFreq;
  float windSpeed;
  float gustStrength;
  float gustFreq;
  uint32_t texOut;
  uint32_t texSize;
};

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = -90,
      .height = -90,
      .resizable = true,
      .initialCameraPos = vec3(0.0f, 3.0f, 8.0f),
      .initialCameraTarget = vec3(0.0f, 0.0f, 0.0f),
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  // Generate blade data
  std::vector<GrassBlade> blades(kNumBlades);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distPos(-kFieldSize, kFieldSize);
    std::uniform_real_distribution<float> distHeight(0.15f, 0.55f);
    std::uniform_real_distribution<float> distWidth(0.02f, 0.06f);
    std::uniform_real_distribution<float> distLean(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distPhase(0.0f, 6.2831853f);
    std::uniform_real_distribution<float> distStiff(0.5f, 2.0f);
    std::uniform_real_distribution<float> distColor(0.0f, 1.0f);

    for (uint32_t i = 0; i < kNumBlades; i++) {
      blades[i] = {
          .posX = distPos(rng),
          .posZ = distPos(rng),
          .height = distHeight(rng),
          .width = distWidth(rng),
          .lean = distLean(rng),
          .phase = distPhase(rng),
          .stiffness = distStiff(rng),
          .colorVariation = distColor(rng),
      };
    }
  }

  lvk::Holder<lvk::BufferHandle> bufBlades = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = blades.size() * sizeof(GrassBlade),
      .data = blades.data(),
      .debugName = "Buffer: blade data",
  });

  lvk::Holder<lvk::BufferHandle> bufPerFrame = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerFrame),
      .debugName = "Buffer: per frame",
  });

  // Wind displacement texture (256x256 RGBA16F storage image)
  lvk::Holder<lvk::TextureHandle> windTexture = ctx->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_RGBA_F16,
      .dimensions = {kWindTexSize, kWindTexSize},
      .usage = lvk::TextureUsageBits_Storage | lvk::TextureUsageBits_Sampled,
      .debugName = "Texture: wind",
  });

  lvk::Holder<lvk::SamplerHandle> windSampler = ctx->createSampler({
      .wrapU = lvk::SamplerWrap_Repeat,
      .wrapV = lvk::SamplerWrap_Repeat,
      .debugName = "Sampler: wind",
  });

  // Shader modules
  lvk::Holder<lvk::ShaderModuleHandle> smWindComp =
      ctx->createShaderModule({codeWindCompute, lvk::Stage_Comp, "Shader Module: wind compute"});
  lvk::Holder<lvk::ShaderModuleHandle> smGroundVert =
      ctx->createShaderModule({codeGroundVS, lvk::Stage_Vert, "Shader Module: ground (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smGroundFrag =
      ctx->createShaderModule({codeGroundFS, lvk::Stage_Frag, "Shader Module: ground (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smGrassVert =
      ctx->createShaderModule({codeGrassVS, lvk::Stage_Vert, "Shader Module: grass (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smGrassFrag =
      ctx->createShaderModule({codeGrassFS, lvk::Stage_Frag, "Shader Module: grass (frag)"});

  // Pipelines
  lvk::Holder<lvk::ComputePipelineHandle> pipelineWind = ctx->createComputePipeline({
      .smComp = smWindComp,
      .debugName = "Pipeline: wind compute",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineGround = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smGroundVert,
      .smFrag = smGroundFrag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = app.getDepthFormat(),
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: ground",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineGrass = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smGrassVert,
      .smFrag = smGrassFrag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = app.getDepthFormat(),
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: grass",
  });

  // ImGui wind parameters
  float windStrength = 0.5f;
  float windFrequency = 2.0f;
  float windSpeed = 1.5f;
  float gustStrength = 0.3f;
  float gustFrequency = 0.5f;

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    const float fov = float(45.0f * (M_PI / 180.0f));
    const float currentTime = (float)glfwGetTime();

    const PerFrame perFrame = {
        .proj = glm::perspective(fov, aspectRatio, 0.1f, 200.0f),
        .view = app.camera_.getViewMatrix(),
        .time = currentTime,
        .windTex = windTexture.index(),
        .windSamp = windSampler.index(),
        .fieldSize = kFieldSize,
    };

    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

    buffer.cmdUpdateBuffer(bufPerFrame, perFrame);

    // 1. Compute pass: generate wind texture
    {
      const WindParams windParams = {
          .time = currentTime,
          .windStrength = windStrength,
          .windFreq = windFrequency,
          .windSpeed = windSpeed,
          .gustStrength = gustStrength,
          .gustFreq = gustFrequency,
          .texOut = windTexture.index(),
          .texSize = kWindTexSize,
      };
      buffer.cmdBindComputePipeline(pipelineWind);
      buffer.cmdPushConstants(windParams);
      buffer.cmdDispatchThreadGroups(
          {(kWindTexSize + 15) / 16, (kWindTexSize + 15) / 16, 1});
    }

    // 2. Render pass
    lvk::Framebuffer framebuffer = {
        .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
        .depthStencil = {app.getDepthTexture()},
    };
    buffer.cmdBeginRendering(
        lvk::RenderPass{
            .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.53f, 0.81f, 0.92f, 1.0f}}},
            .depth = {.loadOp = lvk::LoadOp_Clear, .clearDepth = 1.0}},
        framebuffer,
        {.textures = {lvk::TextureHandle(windTexture)}});
    {
      buffer.cmdBindViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f});
      buffer.cmdBindScissorRect({0, 0, width, height});
      buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});

      // Draw ground plane
      {
        buffer.cmdBindRenderPipeline(pipelineGround);
        const struct {
          uint64_t perFrame;
        } groundPC = {
            .perFrame = ctx->gpuAddress(bufPerFrame),
        };
        buffer.cmdPushConstants(groundPC);
        buffer.cmdDraw(4);
      }

      // Draw grass blades
      {
        buffer.cmdBindRenderPipeline(pipelineGrass);
        const struct {
          uint64_t perFrame;
          uint64_t bladeData;
        } grassPC = {
            .perFrame = ctx->gpuAddress(bufPerFrame),
            .bladeData = ctx->gpuAddress(bufBlades),
        };
        buffer.cmdPushConstants(grassPC);
        buffer.cmdDraw(kBladesPerStrip, kNumBlades);
      }

      // ImGui
      app.imgui_->beginFrame(framebuffer);
      ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
      ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
      ImGui::Begin("Wind Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::SliderFloat("Wind Strength", &windStrength, 0.0f, 2.0f);
      ImGui::SliderFloat("Wind Frequency", &windFrequency, 0.5f, 5.0f);
      ImGui::SliderFloat("Wind Speed", &windSpeed, 0.5f, 5.0f);
      ImGui::SliderFloat("Gust Strength", &gustStrength, 0.0f, 1.0f);
      ImGui::SliderFloat("Gust Frequency", &gustFrequency, 0.1f, 2.0f);
      ImGui::End();
      app.drawFPS();
      app.imgui_->endFrame(buffer);
    }
    buffer.cmdEndRendering();

    ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
  });

  VULKAN_APP_EXIT();
}
