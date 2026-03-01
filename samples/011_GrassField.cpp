/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

#include <random>
#include <string>

constexpr uint32_t kNumBlades = 1000000;
constexpr float kFieldSize = 20.0f;
constexpr uint32_t kWindTexSize = 256;
constexpr uint32_t kBladesPerStrip = 7;
constexpr uint32_t kGridSize = 128; // terrain grid: kGridSize x kGridSize quads
constexpr uint32_t kMSAASamples = 4;
constexpr uint32_t kNumTrees = 150;
constexpr uint32_t kTrunkSides = 12;
constexpr uint32_t kTrunkVertsPerSegment = (kTrunkSides + 1) * 2; // 26 (triangle strip)
constexpr uint32_t kLeafVerts = 4; // billboard quad (triangle strip)

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
  float windDirX;
  float windDirY;
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

  // wind direction vector
  vec2 windDir = vec2(pc.windDirX, pc.windDirY);
  vec2 windPerp = vec2(-windDir.y, windDir.x); // perpendicular for cross-flow

  // scroll noise along wind direction
  float scrollT = pc.time * pc.windSpeed * 0.15;
  vec2 windUV = uv * pc.windFreq + windDir * scrollT + windPerp * scrollT * 0.3;

  // domain warp: distort sampling coords with high-freq noise scaled by wind strength
  // produces turbulent eddies at strong wind (Ghost of Tsushima technique)
  float warpScale = 0.4 * pc.windStrength;
  vec2 warpUV = uv * pc.windFreq * 3.0 + windDir * pc.time * 0.2;
  vec2 warp = vec2(gradientNoise(warpUV), gradientNoise(warpUV + vec2(7.3, 2.9)));
  windUV += warp * warpScale;

  float noiseX = fbm(windUV);
  float noiseZ = fbm(windUV + vec2(5.2, 1.3));

  // broad rolling gusts (low frequency, slow movement)
  vec2 gustUV = uv * pc.gustFreq + windDir * pc.time * 0.07;
  float gust = fbm(gustUV + vec2(17.0, 31.0));
  gust = smoothstep(-0.3, 0.6, gust); // soft gust envelope

  // directional base wind + noise variation + gusts along wind direction
  vec2 wind = windDir * pc.windStrength * 0.5
            + vec2(noiseX, noiseZ) * pc.windStrength
            + windDir * gust * pc.gustStrength;

  imageStore(kTextures2DInOut[pc.texOut], pos, vec4(wind, 0.0, 1.0));
}
)";

// Terrain height function (GLSL snippet, included in both ground and grass shaders)
// Layered sine waves creating gentle rolling hills and mounds
const char* codeTerrainFunc = R"(
float terrainHeight(vec2 p) {
  return 0.45 * sin(p.x * 0.25) * sin(p.y * 0.20)
       + 0.25 * sin(p.x * 0.55 + 1.3) * sin(p.y * 0.45 + 2.1)
       + 0.12 * sin(p.x * 1.1 + 3.7) * cos(p.y * 0.9 + 0.8)
       + 0.06 * cos(p.x * 2.3 + 0.5) * sin(p.y * 1.8 + 1.5);
}

vec3 terrainNormal(vec2 p) {
  float eps = 0.1;
  float hC = terrainHeight(p);
  float hR = terrainHeight(p + vec2(eps, 0.0));
  float hU = terrainHeight(p + vec2(0.0, eps));
  return normalize(vec3(hC - hR, eps, hC - hU));
}
)";

// Ground vertex shader: tessellated grid with terrain height
// Each instance = one row of triangle strip connecting row i to row i+1
const char* codeGroundVS = R"(
layout (location=0) out vec2 v_WorldXZ;
layout (location=1) out vec3 v_Normal;
layout (location=2) out vec4 v_ShadowCoords;
layout (location=3) out vec3 v_WorldPos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

%TERRAIN_FUNC%

void main() {
  uint N = pc.perFrame.gridSize;
  float s = pc.perFrame.fieldSize;

  // gl_InstanceIndex = row index, gl_VertexIndex encodes column + top/bottom
  uint col = gl_VertexIndex / 2;
  uint isTop = gl_VertexIndex % 2;
  uint row = gl_InstanceIndex + isTop;

  vec2 xz = vec2(float(col) / float(N), float(row) / float(N)) * 2.0 - 1.0;
  xz *= s;

  float y = terrainHeight(xz);
  v_WorldXZ = xz;
  v_Normal = terrainNormal(xz);
  v_WorldPos = vec3(xz.x, y, xz.y);
  v_ShadowCoords = pc.perFrame.light * vec4(v_WorldPos, 1.0);
  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(v_WorldPos, 1.0);
}
)";

// Ground fragment shader with terrain-aware lighting and shadows
const char* codeGroundFS = R"(
layout (location=0) in vec2 v_WorldXZ;
layout (location=1) in vec3 v_Normal;
layout (location=2) in vec4 v_ShadowCoords;
layout (location=3) in vec3 v_WorldPos;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

float PCF3(vec3 uvw) {
  float size = 1.0 / textureBindlessSize2D(pc.perFrame.texShadow).x;
  float shadow = 0.0;
  for (int v=-1; v<=+1; v++)
    for (int u=-1; u<=+1; u++)
      shadow += textureBindless2DShadow(pc.perFrame.texShadow, pc.perFrame.sampShadow, uvw + size * vec3(u, v, 0));
  return shadow / 9;
}

float shadow(vec4 s) {
  s = s / s.w;
  if (s.z > -1.0 && s.z < 1.0) {
    float shadowSample = PCF3(vec3(s.x, 1.0 - s.y, s.z + pc.perFrame.depthBias));
    return mix(0.4, 1.0, shadowSample);
  }
  return 1.0;
}

void main() {
  // earthy brown with subtle variation
  float n = fract(sin(dot(floor(v_WorldXZ * 4.0), vec2(12.9898, 78.233))) * 43758.5453);
  vec3 albedo = mix(vec3(0.28, 0.20, 0.10), vec3(0.35, 0.25, 0.12), n);

  vec3 N = normalize(v_Normal);
  vec3 L = normalize(vec3(pc.perFrame.lightDirX, pc.perFrame.lightDirY, pc.perFrame.lightDirZ));
  vec3 camPos = -(transpose(mat3(pc.perFrame.view)) * vec3(pc.perFrame.view[3]));
  vec3 V = normalize(camPos - v_WorldPos);

  // hemisphere ambient: sky blue from above, warm ground bounce from below
  float skyBlend = N.y * 0.5 + 0.5;
  vec3 ambient = mix(vec3(0.10, 0.08, 0.05), vec3(0.20, 0.25, 0.35), skyBlend);

  // wrapped diffuse (softer shadow transition than hard Lambertian)
  vec3 sunColor = vec3(1.4, 1.3, 1.1);
  float wrapDiffuse = max(0.0, (dot(N, L) + 0.3) / 1.3);
  vec3 diffuse = sunColor * wrapDiffuse;

  // subtle specular (moist earth sheen)
  vec3 H = normalize(L + V);
  float spec = pow(max(dot(N, H), 0.0), 24.0) * 0.08;

  float shd = shadow(v_ShadowCoords);

  // SSAO
  vec2 ssaoUV = gl_FragCoord.xy / vec2(textureBindlessSize2D(pc.perFrame.texSSAO));
  float ao = textureBindless2D(pc.perFrame.texSSAO, pc.perFrame.sampSSAO, ssaoUV).r;

  vec3 color = albedo * (ambient * ao + diffuse * shd) + sunColor * spec * shd;
  out_FragColor = vec4(color, 1.0);
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
layout (location=4) out vec4 v_ShadowCoords;
layout (location=5) out vec3 v_WorldPos;

struct GrassBlade {
  float posX, posZ;
  float height, width;
  float lean, phase;
  float stiffness;
  float colorVariation;
  float curvature; // natural rest bend amount
  float padding;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(std430, buffer_reference) readonly buffer BladeData {
  GrassBlade blades[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  BladeData bladeData;
} pc;

%TERRAIN_FUNC%

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

  // taper varies with blade height: short blades stay wide, tall blades narrow quickly
  float taperRate = mix(0.5, 0.9, clamp(blade.height / 0.8, 0.0, 1.0));
  float bladeWidth = blade.width * (1.0 - t * taperRate);

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
  // natural rest curvature: blade droops in its lean direction even without wind
  vec2 restBend = bladeDir * blade.curvature;

  // compute total horizontal bend at this vertex's height as an arc angle
  // rest curvature + wind bend + lean + detail flutter all contribute
  vec2 totalBendXZ = restBend
                   + windDisplacement * bendFactor
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

  // terrain height offset
  pos.y += terrainHeight(vec2(blade.posX, blade.posZ));

  // rich meadow color palette: interpolate across multiple hues based on colorVariation
  float cv = blade.colorVariation;
  vec3 baseColor, tipColor;
  if (cv < 0.3) {
    // dark green / emerald
    float f = cv / 0.3;
    baseColor = mix(vec3(0.03, 0.10, 0.02), vec3(0.05, 0.14, 0.02), f);
    tipColor  = mix(vec3(0.15, 0.40, 0.05), vec3(0.25, 0.55, 0.10), f);
  } else if (cv < 0.6) {
    // bright green / spring
    float f = (cv - 0.3) / 0.3;
    baseColor = mix(vec3(0.05, 0.14, 0.02), vec3(0.08, 0.13, 0.03), f);
    tipColor  = mix(vec3(0.25, 0.55, 0.10), vec3(0.40, 0.65, 0.12), f);
  } else if (cv < 0.85) {
    // olive / warm green
    float f = (cv - 0.6) / 0.25;
    baseColor = mix(vec3(0.08, 0.13, 0.03), vec3(0.12, 0.11, 0.04), f);
    tipColor  = mix(vec3(0.40, 0.65, 0.12), vec3(0.50, 0.50, 0.15), f);
  } else {
    // dried golden / straw
    float f = (cv - 0.85) / 0.15;
    baseColor = mix(vec3(0.12, 0.11, 0.04), vec3(0.18, 0.14, 0.05), f);
    tipColor  = mix(vec3(0.55, 0.50, 0.18), vec3(0.70, 0.60, 0.25), f);
  }
  v_Color = mix(baseColor, tipColor, t);
  v_AO = mix(0.6, 1.0, t); // ambient occlusion: darker at base
  v_BendAmount = length(windDisplacement) * bendFactor;
  // normal faces camera in XZ, with vertical and wind tilt
  vec3 faceNormal = vec3(-camRightXZ.y, 0.0, camRightXZ.x); // perpendicular to camRight in XZ
  v_Normal = normalize(faceNormal * widthScale * 0.3 + vec3(windDisplacement.x * 0.2, 1.0, windDisplacement.y * 0.2));

  v_WorldPos = pos;
  v_ShadowCoords = pc.perFrame.light * vec4(pos, 1.0);
  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(pos, 1.0);
}
)";

// Grass fragment shader
const char* codeGrassFS = R"(
layout (location=0) in vec3 v_Color;
layout (location=1) in float v_AO;
layout (location=2) in vec3 v_Normal;
layout (location=3) in float v_BendAmount;
layout (location=4) in vec4 v_ShadowCoords;
layout (location=5) in vec3 v_WorldPos;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

float PCF3(vec3 uvw) {
  float size = 1.0 / textureBindlessSize2D(pc.perFrame.texShadow).x;
  float shadow = 0.0;
  for (int v=-1; v<=+1; v++)
    for (int u=-1; u<=+1; u++)
      shadow += textureBindless2DShadow(pc.perFrame.texShadow, pc.perFrame.sampShadow, uvw + size * vec3(u, v, 0));
  return shadow / 9;
}

float shadow(vec4 s) {
  s = s / s.w;
  if (s.z > -1.0 && s.z < 1.0) {
    float shadowSample = PCF3(vec3(s.x, 1.0 - s.y, s.z + pc.perFrame.depthBias));
    return mix(0.4, 1.0, shadowSample);
  }
  return 1.0;
}

void main() {
  vec3 N = normalize(v_Normal);
  vec3 L = normalize(vec3(pc.perFrame.lightDirX, pc.perFrame.lightDirY, pc.perFrame.lightDirZ));
  vec3 camPos = -(transpose(mat3(pc.perFrame.view)) * vec3(pc.perFrame.view[3]));
  vec3 V = normalize(camPos - v_WorldPos);

  // hemisphere ambient: sky blue from above, warm ground bounce from below
  float skyBlend = N.y * 0.5 + 0.5;
  vec3 ambient = mix(vec3(0.10, 0.08, 0.05), vec3(0.18, 0.22, 0.30), skyBlend);

  // wrapped diffuse (softer than hard Lambertian cutoff)
  vec3 sunColor = vec3(1.4, 1.3, 1.1);
  float wrapDiffuse = max(0.0, (dot(N, L) + 0.4) / 1.4);
  vec3 diffuse = sunColor * wrapDiffuse;

  // subsurface translucency: light transmits through thin blades when backlit
  vec3 transLightDir = L + N * 0.3;
  float transDot = max(0.0, dot(-normalize(transLightDir), V));
  float translucency = pow(transDot, 6.0) * 0.5;
  translucency *= v_AO; // tips are thinner, transmit more light
  vec3 transColor = v_Color * sunColor * translucency;

  // subtle Blinn-Phong specular (blade sheen)
  vec3 H = normalize(L + V);
  float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.12;

  // wind-driven AO: bent blades darken (self-shadowing)
  float windAO = 1.0 - clamp(v_BendAmount * 0.6, 0.0, 0.35);

  float shd = shadow(v_ShadowCoords);

  // SSAO
  vec2 ssaoUV = gl_FragCoord.xy / vec2(textureBindlessSize2D(pc.perFrame.texSSAO));
  float ao = textureBindless2D(pc.perFrame.texSSAO, pc.perFrame.sampSSAO, ssaoUV).r;

  vec3 color = v_Color * (ambient * ao + diffuse * shd) * v_AO * windAO
             + transColor * shd
             + sunColor * spec * shd;
  out_FragColor = vec4(color, 1.0);
}
)";

// Shadow pass: depth-only ground vertex shader (light perspective)
const char* codeShadowGroundVS = R"(
layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

%TERRAIN_FUNC%

void main() {
  uint N = pc.perFrame.gridSize;
  float s = pc.perFrame.fieldSize;

  uint col = gl_VertexIndex / 2;
  uint isTop = gl_VertexIndex % 2;
  uint row = gl_InstanceIndex + isTop;

  vec2 xz = vec2(float(col) / float(N), float(row) / float(N)) * 2.0 - 1.0;
  xz *= s;

  float y = terrainHeight(xz);
  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(xz.x, y, xz.y, 1.0);
}
)";

// Shadow pass: depth-only grass vertex shader (light perspective)
// Simplified: wind + arc bending + terrain offset + billboarding, no detail flutter or color
const char* codeShadowGrassVS = R"(
layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler   kSamplers[];

struct GrassBlade {
  float posX, posZ;
  float height, width;
  float lean, phase;
  float stiffness;
  float colorVariation;
  float curvature;
  float padding;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(std430, buffer_reference) readonly buffer BladeData {
  GrassBlade blades[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  BladeData bladeData;
} pc;

%TERRAIN_FUNC%

void main() {
  GrassBlade blade = pc.bladeData.blades[gl_InstanceIndex];

  int segment = gl_VertexIndex / 2;
  int side = gl_VertexIndex % 2;
  float t;
  float widthScale;

  if (gl_VertexIndex == 6) {
    t = 1.0;
    widthScale = 0.0;
  } else {
    t = float(segment) / 3.0;
    widthScale = (side == 0) ? -1.0 : 1.0;
  }

  float taperRate = mix(0.5, 0.9, clamp(blade.height / 0.8, 0.0, 1.0));
  float bladeWidth = blade.width * (1.0 - t * taperRate);

  // sample wind texture
  vec2 fieldUV = vec2(blade.posX, blade.posZ) / (2.0 * pc.perFrame.fieldSize) + 0.5;
  vec4 windSample = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.perFrame.windTex], kSamplers[pc.perFrame.windSamp])),
    fieldUV);
  vec2 windDisplacement = windSample.xy;

  // wind-alignment attenuation
  vec2 bladeDir = normalize(vec2(cos(blade.phase), sin(blade.phase)));
  float windLen = length(windDisplacement);
  float alignment = (windLen > 0.001)
    ? 1.0 - 0.6 * abs(dot(bladeDir, windDisplacement / windLen))
    : 1.0;

  float bendFactor = t * t * t * alignment / blade.stiffness;

  // billboard: orient blade width to face the light (via view matrix)
  mat4 view = pc.perFrame.view;
  vec2 camRightXZ = normalize(vec2(view[0][0], view[2][0]));
  float halfW = widthScale * bladeWidth * 0.5;

  // arc-based bending
  vec2 restBend = bladeDir * blade.curvature;
  vec2 totalBendXZ = restBend
                   + windDisplacement * bendFactor
                   + vec2(blade.lean * 0.3) * alignment;
  float bendMag = length(totalBendXZ);

  float h = blade.height;
  float arcAngle = clamp(bendMag / max(h, 0.01), 0.0, 1.2) * t;
  vec2 bendDir = (bendMag > 0.001) ? totalBendXZ / bendMag : vec2(1.0, 0.0);

  float arcY = cos(arcAngle) * t * h;
  float arcH = sin(arcAngle) * t * h;

  vec3 pos;
  pos.x = blade.posX + halfW * camRightXZ.x + bendDir.x * arcH;
  pos.y = arcY;
  pos.z = blade.posZ + halfW * camRightXZ.y + bendDir.y * arcH;

  // terrain height offset
  pos.y += terrainHeight(vec2(blade.posX, blade.posZ));

  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(pos, 1.0);
}
)";

// Shadow pass: empty fragment shader (depth-only)
const char* codeShadowFS = R"(
void main() {}
)";

// SSAO compute shader: hemisphere sampling from depth buffer
const char* codeSSAOCompute = R"(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler   kSamplers[];
layout (set = 0, binding = 2, rgba16f) uniform image2D kTextures2DInOut[];

layout(push_constant) uniform constants {
  uint texDepth;
  uint sampDepth;
  uint texOut;
  uint width;
  uint height;
  float proj00;
  float proj11;
  float proj22;
  float proj32;
  float radius;
  float bias;
  float intensity;
} pc;

vec3 viewPosFromDepth(vec2 uv, float d) {
  float z_ndc = 2.0 * d - 1.0;
  float z_eye = -pc.proj32 / (z_ndc + pc.proj22);
  float negZ = -z_eye;
  return vec3(
    (2.0 * uv.x - 1.0) * negZ / pc.proj00,
    -(2.0 * uv.y - 1.0) * negZ / pc.proj11,
    z_eye
  );
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// 16 hemisphere samples biased toward the surface (cosine-weighted)
const int KERNEL_SIZE = 16;
const vec3 kernel[KERNEL_SIZE] = vec3[](
  vec3( 0.04, 0.04, 0.06),
  vec3(-0.08, 0.10, 0.12),
  vec3( 0.12,-0.04, 0.10),
  vec3(-0.05,-0.12, 0.18),
  vec3( 0.16, 0.08, 0.08),
  vec3(-0.12, 0.18, 0.14),
  vec3( 0.08,-0.16, 0.12),
  vec3( 0.22, 0.00, 0.10),
  vec3(-0.18,-0.08, 0.22),
  vec3( 0.00, 0.22, 0.18),
  vec3( 0.26,-0.12, 0.08),
  vec3(-0.08, 0.26, 0.14),
  vec3( 0.14, 0.14, 0.28),
  vec3(-0.26, 0.04, 0.18),
  vec3( 0.04,-0.28, 0.22),
  vec3( 0.18, 0.18, 0.20)
);

void main() {
  ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  if (pos.x >= int(pc.width) || pos.y >= int(pc.height)) return;

  vec2 texelSize = 1.0 / vec2(pc.width, pc.height);
  vec2 uv = (vec2(pos) + 0.5) * texelSize;

  float depth = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.texDepth], kSamplers[pc.sampDepth])), uv).r;
  if (depth >= 1.0) {
    imageStore(kTextures2DInOut[pc.texOut], pos, vec4(1.0));
    return;
  }

  vec3 P = viewPosFromDepth(uv, depth);

  // reconstruct normal from depth neighbors
  float dR = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.texDepth], kSamplers[pc.sampDepth])),
    uv + vec2(texelSize.x, 0)).r;
  float dU = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.texDepth], kSamplers[pc.sampDepth])),
    uv + vec2(0, texelSize.y)).r;
  vec3 PR = viewPosFromDepth(uv + vec2(texelSize.x, 0), dR);
  vec3 PU = viewPosFromDepth(uv + vec2(0, texelSize.y), dU);
  vec3 N = normalize(cross(PR - P, PU - P));
  if (dot(N, P) > 0.0) N = -N;

  // per-pixel random rotation
  float angle = hash(vec2(pos)) * 6.2831853;
  float cosA = cos(angle), sinA = sin(angle);

  // build TBN from normal
  vec3 T = abs(N.z) < 0.99 ? normalize(cross(N, vec3(0, 0, 1))) : normalize(cross(N, vec3(1, 0, 0)));
  vec3 B = cross(N, T);
  mat3 TBN = mat3(T, B, N);

  float occlusion = 0.0;
  for (int i = 0; i < KERNEL_SIZE; i++) {
    // rotate sample in tangent plane
    vec3 k = kernel[i];
    vec2 rotated = vec2(k.x * cosA - k.y * sinA, k.x * sinA + k.y * cosA);
    vec3 sampleDir = TBN * vec3(rotated, k.z);

    vec3 samplePos = P + sampleDir * pc.radius;

    // project to screen
    float clipX = samplePos.x * pc.proj00;
    float clipY = samplePos.y * pc.proj11;
    float clipZ = samplePos.z * pc.proj22 + pc.proj32;
    float clipW = -samplePos.z;

    vec2 sampleUV = vec2(clipX / clipW * 0.5 + 0.5, 0.5 - clipY / clipW * 0.5);
    sampleUV = clamp(sampleUV, vec2(0.001), vec2(0.999));

    float sampleDepth = texture(
      nonuniformEXT(sampler2D(kTextures2D[pc.texDepth], kSamplers[pc.sampDepth])),
      sampleUV).r;
    float sampleZ = viewPosFromDepth(sampleUV, sampleDepth).z;

    float rangeCheck = smoothstep(0.0, 1.0, pc.radius / abs(P.z - sampleZ));
    occlusion += (sampleZ >= samplePos.z + pc.bias ? 1.0 : 0.0) * rangeCheck;
  }

  float ao = 1.0 - (occlusion / float(KERNEL_SIZE)) * pc.intensity;
  ao = clamp(ao, 0.0, 1.0);
  imageStore(kTextures2DInOut[pc.texOut], pos, vec4(ao, ao, ao, 1.0));
}
)";

// Tree trunk vertex shader: procedural octagonal cylinder from segment data
const char* codeTrunkVS = R"(
layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler   kSamplers[];

layout (location=0) out vec3 v_Normal;
layout (location=1) out vec4 v_ShadowCoords;
layout (location=2) out vec3 v_WorldPos;
layout (location=3) out float v_Height;
layout (location=4) flat out float v_TreeType;

struct TreeTrunkSegment {
  float startX, startY, startZ, startRadius;
  float endX, endY, endZ, endRadius;
  float baseX, baseY, baseZ, treeType;
  float refRX, refRY, refRZ, refRPad;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(std430, buffer_reference) readonly buffer SegmentData {
  TreeTrunkSegment segments[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  SegmentData segmentData;
} pc;

void main() {
  TreeTrunkSegment seg = pc.segmentData.segments[gl_InstanceIndex];

  const int NUM_SIDES = 12;
  int pairIndex = gl_VertexIndex / 2;
  int isTop = gl_VertexIndex & 1;

  float angle = float(pairIndex) / float(NUM_SIDES) * 6.2831853;
  float ca = cos(angle);
  float sa = sin(angle);

  vec3 sPos = vec3(seg.startX, seg.startY, seg.startZ);
  vec3 ePos = vec3(seg.endX, seg.endY, seg.endZ);

  vec3 up = normalize(ePos - sPos);
  vec3 refR = vec3(seg.refRX, seg.refRY, seg.refRZ);
  vec3 right = normalize(refR - up * dot(refR, up));
  vec3 fwd = cross(right, up);

  float radius = (isTop == 1) ? seg.endRadius : seg.startRadius;
  vec3 offset = (right * ca + fwd * sa) * radius;
  vec3 center = (isTop == 1) ? ePos : sPos;
  vec3 pos = center + offset;

  vec2 fieldUV = vec2(seg.baseX, seg.baseZ) / (2.0 * pc.perFrame.fieldSize) + 0.5;
  vec2 wind = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.perFrame.windTex], kSamplers[pc.perFrame.windSamp])),
    fieldUV).xy;

  float h = pos.y - seg.baseY;
  pos.x += wind.x * h * h * 0.015;
  pos.z += wind.y * h * h * 0.015;

  float flutter = sin(pc.perFrame.time * 4.0 + pos.x * 8.0 + pos.z * 6.0)
                * 0.002 * h * min(1.0 / max(radius, 0.005) * 0.02, 1.0);
  pos.x += flutter;
  pos.z += flutter * 0.7;

  v_Normal = normalize(right * ca + fwd * sa);
  v_WorldPos = pos;
  v_ShadowCoords = pc.perFrame.light * vec4(pos, 1.0);
  v_Height = pos.y;
  v_TreeType = seg.treeType;

  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(pos, 1.0);
}
)";

// Tree trunk fragment shader: bark lighting with per-type color
const char* codeTrunkFS = R"(
layout (location=0) in vec3 v_Normal;
layout (location=1) in vec4 v_ShadowCoords;
layout (location=2) in vec3 v_WorldPos;
layout (location=3) in float v_Height;
layout (location=4) flat in float v_TreeType;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

float PCF3(vec3 uvw) {
  float size = 0.5 / textureBindlessSize2D(pc.perFrame.texShadow).x;
  float shadow = 0.0;
  for (int v=-1; v<=+1; v++)
    for (int u=-1; u<=+1; u++)
      shadow += textureBindless2DShadow(pc.perFrame.texShadow, pc.perFrame.sampShadow, uvw + size * vec3(u, v, 0));
  return shadow / 9;
}

float shadow(vec4 s) {
  s = s / s.w;
  if (s.z > -1.0 && s.z < 1.0) {
    float shadowSample = PCF3(vec3(s.x, 1.0 - s.y, s.z + pc.perFrame.depthBias));
    return mix(0.4, 1.0, shadowSample);
  }
  return 1.0;
}

void main() {
  vec3 N = normalize(v_Normal);
  vec3 L = normalize(vec3(pc.perFrame.lightDirX, pc.perFrame.lightDirY, pc.perFrame.lightDirZ));
  vec3 camPos = -(transpose(mat3(pc.perFrame.view)) * vec3(pc.perFrame.view[3]));
  vec3 V = normalize(camPos - v_WorldPos);

  // bark albedo: per-type color with noise and height variation
  float n = fract(sin(dot(v_WorldPos.xz * 8.0, vec2(12.9898, 78.233))) * 43758.5453);
  int tt = int(v_TreeType + 0.5);
  vec3 barkA, barkB;
  if (tt == 1) {        // Maple: grey-brown
    barkA = vec3(0.30, 0.25, 0.20);
    barkB = vec3(0.42, 0.35, 0.28);
  } else if (tt == 2) { // Birch: pale white-grey
    barkA = vec3(0.65, 0.62, 0.58);
    barkB = vec3(0.80, 0.78, 0.72);
  } else {              // Oak: classic brown
    barkA = vec3(0.25, 0.15, 0.08);
    barkB = vec3(0.35, 0.22, 0.10);
  }
  vec3 albedo = mix(barkA, barkB, n);
  albedo *= mix(0.85, 1.0, fract(v_Height * 3.0));

  float skyBlend = N.y * 0.5 + 0.5;
  vec3 ambient = mix(vec3(0.10, 0.08, 0.05), vec3(0.20, 0.25, 0.35), skyBlend);

  vec3 sunColor = vec3(1.4, 1.3, 1.1);
  float wrapDiffuse = max(0.0, (dot(N, L) + 0.3) / 1.3);
  vec3 diffuse = sunColor * wrapDiffuse;

  float shd = shadow(v_ShadowCoords);

  vec2 ssaoUV = gl_FragCoord.xy / vec2(textureBindlessSize2D(pc.perFrame.texSSAO));
  float ao = textureBindless2D(pc.perFrame.texSSAO, pc.perFrame.sampSSAO, ssaoUV).r;

  vec3 color = albedo * (ambient * ao + diffuse * shd);
  out_FragColor = vec4(color, 1.0);
}
)";

// Tree leaf vertex shader: billboard quad from leaf data
const char* codeLeafVS = R"(
layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler   kSamplers[];

layout (location=0) out vec2 v_UV;
layout (location=1) out vec3 v_Color;
layout (location=2) out vec4 v_ShadowCoords;
layout (location=3) out vec3 v_WorldPos;
layout (location=4) flat out float v_LeafType;

struct TreeLeaf {
  float posX, posY, posZ, size;
  float phase, colorVar, dirTheta, dirPhi;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(std430, buffer_reference) readonly buffer LeafData {
  TreeLeaf leaves[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  LeafData leafData;
} pc;

void main() {
  TreeLeaf leaf = pc.leafData.leaves[gl_InstanceIndex];

  float u = float(gl_VertexIndex & 1);
  float v = float(gl_VertexIndex >> 1);
  v_UV = vec2(u, v);

  mat4 view = pc.perFrame.view;

  // Fixed leaf orientation from stored spherical coords
  float theta = leaf.dirTheta;
  float phi = leaf.dirPhi;
  vec3 leafUp = vec3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
  vec3 leafRight = normalize(cross(vec3(0.0, 1.0, 0.0), leafUp));
  // handle degenerate case when leafUp is parallel to world up
  if (length(cross(vec3(0.0, 1.0, 0.0), leafUp)) < 0.001)
    leafRight = vec3(1.0, 0.0, 0.0);

  // Camera-facing axes
  vec3 camRight = vec3(view[0][0], view[1][0], view[2][0]);
  vec3 camUp = vec3(view[0][1], view[1][1], view[2][1]);

  // Blend: 35% camera bias for visibility, 65% fixed for depth/parallax
  vec3 right = normalize(mix(leafRight, camRight, 0.35));
  vec3 up = normalize(mix(leafUp, camUp, 0.35));

  vec3 center = vec3(leaf.posX, leaf.posY, leaf.posZ);

  // Wind-texture sway (synchronized with grass/trunks)
  vec2 fieldUV = vec2(leaf.posX, leaf.posZ) / (2.0 * pc.perFrame.fieldSize) + 0.5;
  vec2 wind = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.perFrame.windTex], kSamplers[pc.perFrame.windSamp])),
    fieldUV).xy;

  float h = center.y;
  center.x += wind.x * h * h * 0.015;
  center.z += wind.y * h * h * 0.015;

  // Local leaf flutter (wind-modulated amplitude)
  float flutter = sin(pc.perFrame.time * 2.5 + leaf.phase) * 0.02 * (1.0 + length(wind) * 0.5);
  center.x += flutter;
  center.z += flutter * 0.7;

  vec3 pos = center + (u - 0.5) * leaf.size * right + (v - 0.5) * leaf.size * up;

  // Per-type leaf color: cv encodes type in [0,0.33), [0.33,0.66), [0.66,1.0)
  float cv = leaf.colorVar;
  vec3 leafColor;
  float leafType;
  if (cv < 0.333) {
    // Oak: warm greens
    float t = cv / 0.333;
    leafColor = mix(vec3(0.08, 0.28, 0.03), vec3(0.30, 0.50, 0.10), t);
    leafType = 0.0;
  } else if (cv < 0.666) {
    // Maple: autumn greens to orange-red
    float t = (cv - 0.333) / 0.333;
    leafColor = mix(vec3(0.20, 0.35, 0.05), vec3(0.55, 0.25, 0.05), t);
    leafType = 1.0;
  } else {
    // Birch: bright yellow-greens
    float t = (cv - 0.666) / 0.334;
    leafColor = mix(vec3(0.25, 0.50, 0.08), vec3(0.50, 0.58, 0.15), t);
    leafType = 2.0;
  }
  v_Color = leafColor;
  v_LeafType = leafType;

  v_WorldPos = pos;
  v_ShadowCoords = pc.perFrame.light * vec4(pos, 1.0);
  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(pos, 1.0);
}
)";

// Tree leaf fragment shader: per-type leaf shapes with foliage lighting
const char* codeLeafFS = R"(
layout (location=0) in vec2 v_UV;
layout (location=1) in vec3 v_Color;
layout (location=2) in vec4 v_ShadowCoords;
layout (location=3) in vec3 v_WorldPos;
layout (location=4) flat in float v_LeafType;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

float PCF3(vec3 uvw) {
  float size = 1.0 / textureBindlessSize2D(pc.perFrame.texShadow).x;
  float shadow = 0.0;
  for (int v=-1; v<=+1; v++)
    for (int u=-1; u<=+1; u++)
      shadow += textureBindless2DShadow(pc.perFrame.texShadow, pc.perFrame.sampShadow, uvw + size * vec3(u, v, 0));
  return shadow / 9;
}

float shadow(vec4 s) {
  s = s / s.w;
  if (s.z > -1.0 && s.z < 1.0) {
    float shadowSample = PCF3(vec3(s.x, 1.0 - s.y, s.z + pc.perFrame.depthBias));
    return mix(0.4, 1.0, shadowSample);
  }
  return 1.0;
}

void main() {
  float vn = v_UV.y;  // 0=stem, 1=tip
  float dx = (v_UV.x - 0.5) * 2.0;  // [-1,1] from center
  int lt = int(v_LeafType + 0.5);

  // Per-type leaf width profiles
  float w;
  if (lt == 1) {
    // Maple: broad, wide leaf with subtle lobes
    float base = pow(vn + 0.001, 0.3) * pow(max(1.0 - vn, 0.0), 1.2) / 0.30;
    float lobe = 1.0 + 0.15 * sin(vn * 9.0); // subtle lobe undulation
    w = base * lobe;
  } else if (lt == 2) {
    // Birch: small round/oval leaf
    w = pow(vn + 0.001, 0.5) * pow(max(1.0 - vn, 0.0), 1.0) / 0.35;
  } else {
    // Oak: elongated pointed leaf
    w = pow(vn + 0.001, 0.4) * pow(max(1.0 - vn, 0.0), 1.8) / 0.33;
  }

  if (abs(dx) > w * 0.95 || vn < 0.01) discard;

  // Central vein darkening
  float veinDist = abs(dx);
  float veinWidth = 0.06 * (1.0 - vn * 0.7);
  float vein = smoothstep(veinWidth, veinWidth * 0.2, veinDist);

  vec3 L = normalize(vec3(pc.perFrame.lightDirX, pc.perFrame.lightDirY, pc.perFrame.lightDirZ));
  vec3 camPos = -(transpose(mat3(pc.perFrame.view)) * vec3(pc.perFrame.view[3]));
  vec3 V = normalize(camPos - v_WorldPos);
  vec3 N = V; // billboard faces camera

  float skyBlend = 0.75;
  vec3 ambient = mix(vec3(0.10, 0.08, 0.05), vec3(0.18, 0.22, 0.30), skyBlend);

  vec3 sunColor = vec3(1.4, 1.3, 1.1);
  float NdotL = dot(N, L);
  float wrapDiffuse = max(0.0, (abs(NdotL) + 0.4) / 1.4);
  vec3 diffuse = sunColor * wrapDiffuse;

  // subsurface translucency
  float trans = pow(max(0.0, dot(-L, V)), 4.0) * 0.3;
  vec3 transColor = v_Color * sunColor * trans;

  float shd = shadow(v_ShadowCoords);

  vec2 ssaoUV = gl_FragCoord.xy / vec2(textureBindlessSize2D(pc.perFrame.texSSAO));
  float ao = textureBindless2D(pc.perFrame.texSSAO, pc.perFrame.sampSSAO, ssaoUV).r;

  // Apply vein darkening and height-based brightness
  vec3 leafColor = v_Color * (1.0 - vein * 0.2);
  float heightBright = 1.0 + v_WorldPos.y * 0.05;
  leafColor *= heightBright;

  vec3 color = leafColor * (ambient * ao + diffuse * shd) + transColor * shd;
  out_FragColor = vec4(color, 1.0);
}
)";

// Shadow pass: depth-only trunk vertex shader
const char* codeShadowTrunkVS = R"(
layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler   kSamplers[];

struct TreeTrunkSegment {
  float startX, startY, startZ, startRadius;
  float endX, endY, endZ, endRadius;
  float baseX, baseY, baseZ, treeType;
  float refRX, refRY, refRZ, refRPad;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(std430, buffer_reference) readonly buffer SegmentData {
  TreeTrunkSegment segments[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  SegmentData segmentData;
} pc;

void main() {
  TreeTrunkSegment seg = pc.segmentData.segments[gl_InstanceIndex];

  const int NUM_SIDES = 12;
  int pairIndex = gl_VertexIndex / 2;
  int isTop = gl_VertexIndex & 1;

  float angle = float(pairIndex) / float(NUM_SIDES) * 6.2831853;

  vec3 sPos = vec3(seg.startX, seg.startY, seg.startZ);
  vec3 ePos = vec3(seg.endX, seg.endY, seg.endZ);

  vec3 up = normalize(ePos - sPos);
  vec3 refR = vec3(seg.refRX, seg.refRY, seg.refRZ);
  vec3 right = normalize(refR - up * dot(refR, up));
  vec3 fwd = cross(right, up);

  float radius = (isTop == 1) ? seg.endRadius : seg.startRadius;
  vec3 offset = (right * cos(angle) + fwd * sin(angle)) * radius;
  vec3 center = (isTop == 1) ? ePos : sPos;
  vec3 pos = center + offset;

  vec2 fieldUV = vec2(seg.baseX, seg.baseZ) / (2.0 * pc.perFrame.fieldSize) + 0.5;
  vec2 wind = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.perFrame.windTex], kSamplers[pc.perFrame.windSamp])),
    fieldUV).xy;

  float h = pos.y - seg.baseY;
  pos.x += wind.x * h * h * 0.015;
  pos.z += wind.y * h * h * 0.015;

  float flutter = sin(pc.perFrame.time * 4.0 + pos.x * 8.0 + pos.z * 6.0)
                * 0.002 * h * min(1.0 / max(radius, 0.005) * 0.02, 1.0);
  pos.x += flutter;
  pos.z += flutter * 0.7;

  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(pos, 1.0);
}
)";

// Shadow pass: depth-only leaf vertex shader (billboard + UV for alpha cutout)
const char* codeShadowLeafVS = R"(
layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler   kSamplers[];

layout (location=0) out vec2 v_UV;
layout (location=1) flat out float v_LeafType;

struct TreeLeaf {
  float posX, posY, posZ, size;
  float phase, colorVar, dirTheta, dirPhi;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint windTex;
  uint windSamp;
  float fieldSize;
  uint gridSize;
  uint texShadow;
  uint sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint texSSAO;
  uint sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

layout(std430, buffer_reference) readonly buffer LeafData {
  TreeLeaf leaves[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  LeafData leafData;
} pc;

void main() {
  TreeLeaf leaf = pc.leafData.leaves[gl_InstanceIndex];

  float u = float(gl_VertexIndex & 1);
  float v = float(gl_VertexIndex >> 1);
  v_UV = vec2(u, v);

  mat4 view = pc.perFrame.view;

  // Fixed leaf orientation from stored spherical coords
  float theta = leaf.dirTheta;
  float phi = leaf.dirPhi;
  vec3 leafUp = vec3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
  vec3 leafRight = normalize(cross(vec3(0.0, 1.0, 0.0), leafUp));
  if (length(cross(vec3(0.0, 1.0, 0.0), leafUp)) < 0.001)
    leafRight = vec3(1.0, 0.0, 0.0);

  // Camera-facing axes
  vec3 camRight = vec3(view[0][0], view[1][0], view[2][0]);
  vec3 camUp = vec3(view[0][1], view[1][1], view[2][1]);

  // Blend: 35% camera bias for visibility, 65% fixed for depth/parallax
  vec3 right = normalize(mix(leafRight, camRight, 0.35));
  vec3 up = normalize(mix(leafUp, camUp, 0.35));

  vec3 center = vec3(leaf.posX, leaf.posY, leaf.posZ);

  // Wind-texture sway (synchronized with grass/trunks)
  vec2 fieldUV = vec2(leaf.posX, leaf.posZ) / (2.0 * pc.perFrame.fieldSize) + 0.5;
  vec2 wind = texture(
    nonuniformEXT(sampler2D(kTextures2D[pc.perFrame.windTex], kSamplers[pc.perFrame.windSamp])),
    fieldUV).xy;

  float h = center.y;
  center.x += wind.x * h * h * 0.015;
  center.z += wind.y * h * h * 0.015;

  // Local leaf flutter (wind-modulated amplitude)
  float flutter = sin(pc.perFrame.time * 2.5 + leaf.phase) * 0.02 * (1.0 + length(wind) * 0.5);
  center.x += flutter;
  center.z += flutter * 0.7;

  vec3 pos = center + (u - 0.5) * leaf.size * right + (v - 0.5) * leaf.size * up;

  // Compute leaf type from colorVar bands
  float cv = leaf.colorVar;
  if (cv < 0.333) v_LeafType = 0.0;
  else if (cv < 0.666) v_LeafType = 1.0;
  else v_LeafType = 2.0;

  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(pos, 1.0);
}
)";

// Shadow pass: leaf fragment shader (per-type leaf cutout)
const char* codeShadowLeafFS = R"(
layout (location=0) in vec2 v_UV;
layout (location=1) flat in float v_LeafType;

void main() {
  float vn = v_UV.y;
  float dx = (v_UV.x - 0.5) * 2.0;
  int lt = int(v_LeafType + 0.5);

  float w;
  if (lt == 1) {
    // Maple: broad with subtle lobes
    float base = pow(vn + 0.001, 0.3) * pow(max(1.0 - vn, 0.0), 1.2) / 0.30;
    float lobe = 1.0 + 0.15 * sin(vn * 9.0);
    w = base * lobe;
  } else if (lt == 2) {
    // Birch: small round/oval
    w = pow(vn + 0.001, 0.5) * pow(max(1.0 - vn, 0.0), 1.0) / 0.35;
  } else {
    // Oak: elongated pointed
    w = pow(vn + 0.001, 0.4) * pow(max(1.0 - vn, 0.0), 1.8) / 0.33;
  }

  if (abs(dx) > w * 0.95 || vn < 0.01) discard;
}
)";

// SSAO blur compute shader: 5x5 box blur
const char* codeSSAOBlurCompute = R"(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler   kSamplers[];
layout (set = 0, binding = 2, rgba16f) uniform image2D kTextures2DInOut[];

layout(push_constant) uniform constants {
  uint texIn;
  uint sampIn;
  uint texOut;
  uint width;
  uint height;
} pc;

void main() {
  ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  if (pos.x >= int(pc.width) || pos.y >= int(pc.height)) return;

  vec2 texelSize = 1.0 / vec2(pc.width, pc.height);
  float sum = 0.0;
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      vec2 uv = (vec2(pos) + vec2(x, y) + 0.5) * texelSize;
      uv = clamp(uv, vec2(0.0), vec2(1.0));
      sum += texture(
        nonuniformEXT(sampler2D(kTextures2D[pc.texIn], kSamplers[pc.sampIn])), uv).r;
    }
  }
  imageStore(kTextures2DInOut[pc.texOut], pos, vec4(sum / 25.0, 0, 0, 1));
}
)";

// clang-format on

struct GrassBlade {
  float posX, posZ;
  float height, width;
  float lean, phase;
  float stiffness;
  float colorVariation;
  float curvature;
  float padding;
};

struct PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  float time;
  uint32_t windTex;
  uint32_t windSamp;
  float fieldSize;
  uint32_t gridSize;
  uint32_t texShadow;
  uint32_t sampShadow;
  float depthBias;
  float lightDirX;
  float lightDirY;
  float lightDirZ;
  uint32_t texSSAO;
  uint32_t sampSSAO;
  float padding2;
  float padding3;
  float padding4;
};

struct WindParams {
  float time;
  float windStrength;
  float windFreq;
  float windSpeed;
  float gustStrength;
  float gustFreq;
  float windDirX;
  float windDirY;
  uint32_t texOut;
  uint32_t texSize;
};

struct TreeTrunkSegment {
  float startX, startY, startZ, startRadius;
  float endX, endY, endZ, endRadius;
  float baseX, baseY, baseZ, treeType;
  float refRX, refRY, refRZ, refRPad;
};

struct TreeLeaf {
  float posX, posY, posZ, size;
  float phase, colorVar, dirTheta, dirPhi;
};

float terrainHeightCPP(float px, float pz) {
  return 0.45f * sinf(px * 0.25f) * sinf(pz * 0.20f)
       + 0.25f * sinf(px * 0.55f + 1.3f) * sinf(pz * 0.45f + 2.1f)
       + 0.12f * sinf(px * 1.1f + 3.7f) * cosf(pz * 0.9f + 0.8f)
       + 0.06f * cosf(px * 2.3f + 0.5f) * sinf(pz * 1.8f + 1.5f);
}

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

  // Generate blade data with 3 grass types for meadow-like variety
  std::vector<GrassBlade> blades(kNumBlades);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distPos(-kFieldSize, kFieldSize);
    std::uniform_real_distribution<float> distLean(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distPhase(0.0f, 6.2831853f);
    std::uniform_real_distribution<float> distColor(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    // Type A: short grass (45%) — thin blades, stiff, slight droop
    std::uniform_real_distribution<float> distHeightA(0.12f, 0.30f);
    std::uniform_real_distribution<float> distWidthA(0.010f, 0.022f);
    std::uniform_real_distribution<float> distStiffA(1.5f, 3.0f);
    std::uniform_real_distribution<float> distCurvA(0.03f, 0.12f);

    // Type B: medium grass (35%) — moderate height, slender
    std::uniform_real_distribution<float> distHeightB(0.30f, 0.60f);
    std::uniform_real_distribution<float> distWidthB(0.008f, 0.018f);
    std::uniform_real_distribution<float> distStiffB(0.8f, 1.8f);
    std::uniform_real_distribution<float> distCurvB(0.08f, 0.22f);

    // Type C: tall grass (20%) — thin, flexible, graceful droop
    std::uniform_real_distribution<float> distHeightC(0.55f, 1.00f);
    std::uniform_real_distribution<float> distWidthC(0.006f, 0.014f);
    std::uniform_real_distribution<float> distStiffC(0.4f, 1.0f);
    std::uniform_real_distribution<float> distCurvC(0.15f, 0.40f);

    for (uint32_t i = 0; i < kNumBlades; i++) {
      const float typeRoll = dist01(rng);
      float h, w, stiff, curv;
      if (typeRoll < 0.45f) {
        h = distHeightA(rng); w = distWidthA(rng); stiff = distStiffA(rng); curv = distCurvA(rng);
      } else if (typeRoll < 0.80f) {
        h = distHeightB(rng); w = distWidthB(rng); stiff = distStiffB(rng); curv = distCurvB(rng);
      } else {
        h = distHeightC(rng); w = distWidthC(rng); stiff = distStiffC(rng); curv = distCurvC(rng);
      }
      blades[i] = {
          .posX = distPos(rng),
          .posZ = distPos(rng),
          .height = h,
          .width = w,
          .lean = distLean(rng),
          .phase = distPhase(rng),
          .stiffness = stiff,
          .colorVariation = distColor(rng),
          .curvature = curv,
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

  // Generate tree data with recursive branching
  std::vector<TreeTrunkSegment> trunkSegments;
  std::vector<TreeLeaf> treeLeaves;
  {
    std::mt19937 treeRng(123);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> distPhase(0.0f, 6.2831853f);

    // Procedurally place trees with minimum spacing
    std::uniform_real_distribution<float> distTree(-kFieldSize + 1.0f, kFieldSize - 1.0f);
    std::vector<vec3> treeBasePositions;
    treeBasePositions.reserve(kNumTrees);
    const float minDist = 1.8f;
    while (treeBasePositions.size() < kNumTrees) {
      vec3 candidate(distTree(treeRng), 0.0f, distTree(treeRng));
      bool tooClose = false;
      for (const auto& p : treeBasePositions) {
        float dx = candidate.x - p.x, dz = candidate.z - p.z;
        if (dx * dx + dz * dz < minDist * minDist) { tooClose = true; break; }
      }
      if (!tooClose) treeBasePositions.push_back(candidate);
    }

    // Helper: compute initial reference right for a direction
    auto computeRefRight = [](vec3 dir) -> vec3 {
      vec3 arb = fabsf(dir.y) < 0.99f ? vec3(0, 1, 0) : vec3(1, 0, 0);
      return glm::normalize(glm::cross(dir, arb));
    };

    // Helper: spawn a cluster of leaves at a point
    auto spawnLeaves = [&](vec3 pos, int count, float sizeMin, float sizeRange,
                           float clusterRadius, float cvBase, float cvRange, float yBias) {
      for (int i = 0; i < count; i++) {
        float theta = distPhase(treeRng);
        float phi = dist01(treeRng) * 1.5708f;
        float r = clusterRadius * (0.3f + dist01(treeRng) * 0.7f);
        vec3 offset(
            r * sinf(phi) * cosf(theta),
            r * cosf(phi) * 0.7f * yBias + 0.03f,
            r * sinf(phi) * sinf(theta));
        float leafDirTheta = distPhase(treeRng);
        float leafDirPhi = 0.3f + dist01(treeRng) * 1.0f;
        float cv = cvBase + dist01(treeRng) * cvRange;
        treeLeaves.push_back({
            pos.x + offset.x, pos.y + offset.y, pos.z + offset.z,
            sizeMin + dist01(treeRng) * sizeRange,
            distPhase(treeRng), cv, leafDirTheta, leafDirPhi,
        });
      }
    };

    // --- OAK generator: wide rounded canopy, leaves from depth 2+ ---
    auto generateOak = [&](vec3 base, float trunkHeight, float trunkRadius, vec3 trunkDir, vec3 refRight) {
      float tt = 0.0f;
      std::function<void(vec3, vec3, float, float, int, vec3)> growOak;
      growOak = [&](vec3 start, vec3 dir, float length, float radius, int depth, vec3 curRight) {
        int numSubSegs = (depth <= 1) ? 4 : 3;
        float endRadius = radius * 0.80f;
        float subLen = length / (float)numSubSegs;
        curRight = glm::normalize(curRight - dir * glm::dot(curRight, dir));
        vec3 segStart = start;
        for (int s = 0; s < numSubSegs; s++) {
          float t0 = (float)s / (float)numSubSegs;
          float t1 = (float)(s + 1) / (float)numSubSegs;
          trunkSegments.push_back({
              segStart.x, segStart.y, segStart.z, glm::mix(radius, endRadius, t0),
              (segStart + dir * subLen).x, (segStart + dir * subLen).y, (segStart + dir * subLen).z, glm::mix(radius, endRadius, t1),
              base.x, base.y, base.z, tt,
              curRight.x, curRight.y, curRight.z, 0.0f,
          });
          segStart = segStart + dir * subLen;
        }
        vec3 end = segStart;

        // Spawn leaves at depth >= 2 (fills the full rounded canopy)
        if (depth >= 2) {
          float depthFrac = (float)(depth - 2) / 2.0f; // 0 at depth 2, 1 at depth 4
          int leafCount = (int)(60.0f + 100.0f * depthFrac + dist01(treeRng) * 40.0f);
          spawnLeaves(end, leafCount, 0.03f, 0.03f, 0.25f + 0.15f * depthFrac, 0.0f, 0.333f, 1.0f);
        }

        if (depth >= 4) return;

        int numChildren = 2 + (dist01(treeRng) > 0.5f ? 1 : 0);
        for (int i = 0; i < numChildren; i++) {
          float spreadAngle = 0.35f + dist01(treeRng) * 0.40f;
          float rotAngle = distPhase(treeRng);
          vec3 fwd = glm::cross(curRight, dir);
          vec3 childDir = glm::normalize(
              dir + (curRight * cosf(rotAngle) + fwd * sinf(rotAngle)) * tanf(spreadAngle) + vec3(0, 0.05f, 0));
          float childLength = length * (0.55f + dist01(treeRng) * 0.25f);
          vec3 childRight = glm::normalize(curRight - childDir * glm::dot(curRight, childDir));
          growOak(end, childDir, childLength, endRadius, depth + 1, childRight);
        }
      };
      growOak(base, trunkDir, trunkHeight, trunkRadius, 0, refRight);
    };

    // --- PINE generator: conical shape with central leader + whorled horizontal branches ---
    // --- MAPLE generator: medium tree, broad rounded canopy, layered branches ---
    auto generateMaple = [&](vec3 base, float trunkHeight, float trunkRadius, vec3 trunkDir, vec3 refRight) {
      float tt = 1.0f;
      std::function<void(vec3, vec3, float, float, int, vec3)> growMaple;
      growMaple = [&](vec3 start, vec3 dir, float length, float radius, int depth, vec3 curRight) {
        int numSubSegs = (depth <= 1) ? 5 : 3;
        float subLen = length / (float)numSubSegs;
        float endRadius = radius * 0.65f;
        vec3 segStart = start;
        for (int s = 0; s < numSubSegs; s++) {
          float t0 = (float)s / (float)numSubSegs;
          float t1 = (float)(s + 1) / (float)numSubSegs;
          vec3 segEnd = segStart + dir * subLen;
          trunkSegments.push_back({
              segStart.x, segStart.y, segStart.z, glm::mix(radius, endRadius, t0),
              segEnd.x, segEnd.y, segEnd.z, glm::mix(radius, endRadius, t1),
              base.x, base.y, base.z, tt,
              curRight.x, curRight.y, curRight.z, 0.0f,
          });
          segStart = segEnd;
        }

        int maxDepth = 4;
        if (depth >= maxDepth) return;

        // Spawn leaves from depth 2+ (broad canopy fill)
        if (depth >= 2) {
          vec3 leafCenter = start + dir * (length * 0.5f);
          int leafCount = (depth >= 3) ? (60 + (int)(dist01(treeRng) * 40.0f)) : (30 + (int)(dist01(treeRng) * 20.0f));
          float leafSpread = length * 0.6f;
          spawnLeaves(leafCenter, leafCount, 0.03f, 0.02f, leafSpread, 0.333f, 0.333f, 1.0f);
        }

        // Branch into 2-3 children with wide spread (maple has broad canopy)
        int numChildren = 2 + (dist01(treeRng) < 0.5f ? 1 : 0);
        for (int c = 0; c < numChildren; c++) {
          float childLen = length * (0.6f + dist01(treeRng) * 0.15f);
          float childRadius = endRadius;

          // Wide horizontal spread for broad rounded canopy
          float spreadAngle = 0.35f + dist01(treeRng) * 0.40f;
          float rotAngle = 6.2831853f * (float)c / (float)numChildren + (dist01(treeRng) - 0.5f) * 0.5f;

          // Build child direction: spread from parent + slight upward bias
          vec3 fwd = glm::normalize(dir);
          vec3 rgt = glm::normalize(curRight - fwd * glm::dot(curRight, fwd));
          vec3 upV = glm::normalize(glm::cross(fwd, rgt));

          vec3 childDir = fwd * cosf(spreadAngle) +
                          (rgt * cosf(rotAngle) + upV * sinf(rotAngle)) * sinf(spreadAngle);
          // Slight upward bias to keep canopy rounded
          childDir.y += 0.15f;
          childDir = glm::normalize(childDir);

          vec3 childRight = glm::normalize(rgt - childDir * glm::dot(rgt, childDir));
          if (glm::length(childRight) < 0.001f) childRight = rgt;

          growMaple(segStart, childDir, childLen, childRadius, depth + 1, childRight);
        }
      };

      growMaple(base, trunkDir, trunkHeight, trunkRadius, 0, refRight);
    };

    // --- BIRCH generator: tall thin trunk, drooping leaf clusters from depth 2+ ---
    auto generateBirch = [&](vec3 base, float trunkHeight, float trunkRadius, vec3 trunkDir, vec3 refRight) {
      float tt = 2.0f;
      std::function<void(vec3, vec3, float, float, int, vec3)> growBirch;
      growBirch = [&](vec3 start, vec3 dir, float length, float radius, int depth, vec3 curRight) {
        int numSubSegs = (depth <= 1) ? 4 : 3;
        float endRadius = radius * 0.78f;
        float subLen = length / (float)numSubSegs;
        curRight = glm::normalize(curRight - dir * glm::dot(curRight, dir));
        vec3 segStart = start;
        for (int s = 0; s < numSubSegs; s++) {
          float t0 = (float)s / (float)numSubSegs;
          float t1 = (float)(s + 1) / (float)numSubSegs;
          vec3 segEnd = segStart + dir * subLen;
          trunkSegments.push_back({
              segStart.x, segStart.y, segStart.z, glm::mix(radius, endRadius, t0),
              segEnd.x, segEnd.y, segEnd.z, glm::mix(radius, endRadius, t1),
              base.x, base.y, base.z, tt,
              curRight.x, curRight.y, curRight.z, 0.0f,
          });
          segStart = segEnd;
        }
        vec3 end = segStart;

        // Spawn drooping leaf clusters from depth 2+ (yBias < 1 = hang downward)
        if (depth >= 2) {
          float depthFrac = (float)(depth - 2) / 2.0f;
          int leafCount = (int)(50.0f + 80.0f * depthFrac + dist01(treeRng) * 30.0f);
          // Drooping: yBias 0.3 = clusters extend more sideways/downward
          spawnLeaves(end, leafCount, 0.04f, 0.03f, 0.20f + 0.15f * depthFrac, 0.666f, 0.334f, 0.3f);
        }

        if (depth >= 4) return;

        // Birch: more horizontal branching, slight droop
        int numChildren = 2 + (dist01(treeRng) > 0.4f ? 1 : 0);
        for (int i = 0; i < numChildren; i++) {
          float spreadAngle = 0.30f + dist01(treeRng) * 0.45f;
          float rotAngle = distPhase(treeRng);
          vec3 fwd = glm::cross(curRight, dir);
          vec3 childDir = glm::normalize(
              dir + (curRight * cosf(rotAngle) + fwd * sinf(rotAngle)) * tanf(spreadAngle) + vec3(0, -0.05f, 0)); // slight droop
          float childLength = length * (0.55f + dist01(treeRng) * 0.25f);
          vec3 childRight = glm::normalize(curRight - childDir * glm::dot(curRight, childDir));
          growBirch(end, childDir, childLength, endRadius, depth + 1, childRight);
        }
      };
      growBirch(base, trunkDir, trunkHeight, trunkRadius, 0, refRight);
    };

    for (uint32_t t = 0; t < kNumTrees; t++) {
      vec3 base = treeBasePositions[t];
      base.y = terrainHeightCPP(base.x, base.z);

      // Assign tree type: ~40% oak, ~30% maple, ~30% birch
      float typeRoll = dist01(treeRng);
      int treeType = (typeRoll < 0.4f) ? 0 : (typeRoll < 0.7f) ? 1 : 2;

      vec3 trunkDir(0.0f, 1.0f, 0.0f);
      trunkDir.x += (dist01(treeRng) - 0.5f) * 0.1f;
      trunkDir.z += (dist01(treeRng) - 0.5f) * 0.1f;
      trunkDir = glm::normalize(trunkDir);
      vec3 refR = computeRefRight(trunkDir);

      if (treeType == 0) {
        // Oak: medium, wide rounded canopy
        float h = 0.7f + dist01(treeRng) * 0.6f;
        float r = 0.06f + dist01(treeRng) * 0.04f;
        generateOak(base, h, r, trunkDir, refR);
      } else if (treeType == 1) {
        // Maple: medium, broad rounded canopy
        float h = 0.9f + dist01(treeRng) * 0.6f;
        float r = 0.05f + dist01(treeRng) * 0.04f;
        generateMaple(base, h, r, trunkDir, refR);
      } else {
        // Birch: tall thin trunk, drooping canopy
        float h = 1.0f + dist01(treeRng) * 0.7f;
        float r = 0.03f + dist01(treeRng) * 0.02f;
        generateBirch(base, h, r, trunkDir, refR);
      }
    }
  }
  const uint32_t totalTrunkSegments = (uint32_t)trunkSegments.size();
  const uint32_t totalLeaves = (uint32_t)treeLeaves.size();

  lvk::Holder<lvk::BufferHandle> bufTrunkSegments = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = trunkSegments.size() * sizeof(TreeTrunkSegment),
      .data = trunkSegments.data(),
      .debugName = "Buffer: trunk segments",
  });

  lvk::Holder<lvk::BufferHandle> bufLeaves = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = treeLeaves.size() * sizeof(TreeLeaf),
      .data = treeLeaves.data(),
      .debugName = "Buffer: tree leaves",
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

  // Shadow map resources
  constexpr uint32_t kShadowMapSize = 4096;

  lvk::Holder<lvk::TextureHandle> shadowMap = ctx->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_Z_UN16,
      .dimensions = {kShadowMapSize, kShadowMapSize},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .numMipLevels = 1,
      .debugName = "Texture: shadow map",
  });

  lvk::Holder<lvk::SamplerHandle> shadowSampler = ctx->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .depthCompareOp = lvk::CompareOp_LessEqual,
      .depthCompareEnabled = true,
      .debugName = "Sampler: shadow",
  });

  lvk::Holder<lvk::BufferHandle> bufPerFrameShadow = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerFrame),
      .debugName = "Buffer: per frame (shadow)",
  });

  // SSAO resources: clamp sampler for depth texture reading
  lvk::Holder<lvk::SamplerHandle> ssaoSampler = ctx->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .debugName = "Sampler: SSAO clamp",
  });

  // 1x1 white texture used as SSAO fallback when disabled
  const uint16_t whitePixel[] = {0x3C00, 0x3C00, 0x3C00, 0x3C00}; // fp16 1.0
  lvk::Holder<lvk::TextureHandle> whiteTex = ctx->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_RGBA_F16,
      .dimensions = {1, 1},
      .usage = lvk::TextureUsageBits_Sampled,
      .numMipLevels = 1,
      .data = whitePixel,
      .debugName = "Texture: white 1x1",
  });

  // Inject terrain function into shaders that use %TERRAIN_FUNC% placeholder
  auto injectTerrain = [](const char* src) -> std::string {
    std::string s(src);
    const std::string placeholder = "%TERRAIN_FUNC%";
    size_t pos = s.find(placeholder);
    if (pos != std::string::npos)
      s.replace(pos, placeholder.size(), codeTerrainFunc);
    return s;
  };
  const std::string groundVS = injectTerrain(codeGroundVS);
  const std::string grassVS = injectTerrain(codeGrassVS);
  const std::string shadowGroundVS = injectTerrain(codeShadowGroundVS);
  const std::string shadowGrassVS = injectTerrain(codeShadowGrassVS);

  // Shader modules
  lvk::Holder<lvk::ShaderModuleHandle> smWindComp =
      ctx->createShaderModule({codeWindCompute, lvk::Stage_Comp, "Shader Module: wind compute"});
  lvk::Holder<lvk::ShaderModuleHandle> smGroundVert =
      ctx->createShaderModule({groundVS.c_str(), lvk::Stage_Vert, "Shader Module: ground (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smGroundFrag =
      ctx->createShaderModule({codeGroundFS, lvk::Stage_Frag, "Shader Module: ground (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smGrassVert =
      ctx->createShaderModule({grassVS.c_str(), lvk::Stage_Vert, "Shader Module: grass (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smGrassFrag =
      ctx->createShaderModule({codeGrassFS, lvk::Stage_Frag, "Shader Module: grass (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smShadowGroundVert =
      ctx->createShaderModule({shadowGroundVS.c_str(), lvk::Stage_Vert, "Shader Module: shadow ground (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smShadowGrassVert =
      ctx->createShaderModule({shadowGrassVS.c_str(), lvk::Stage_Vert, "Shader Module: shadow grass (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smShadowFrag =
      ctx->createShaderModule({codeShadowFS, lvk::Stage_Frag, "Shader Module: shadow (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smSSAOComp =
      ctx->createShaderModule({codeSSAOCompute, lvk::Stage_Comp, "Shader Module: SSAO compute"});
  lvk::Holder<lvk::ShaderModuleHandle> smSSAOBlurComp =
      ctx->createShaderModule({codeSSAOBlurCompute, lvk::Stage_Comp, "Shader Module: SSAO blur compute"});

  // Tree shader modules
  lvk::Holder<lvk::ShaderModuleHandle> smTrunkVert =
      ctx->createShaderModule({codeTrunkVS, lvk::Stage_Vert, "Shader Module: trunk (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smTrunkFrag =
      ctx->createShaderModule({codeTrunkFS, lvk::Stage_Frag, "Shader Module: trunk (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smLeafVert =
      ctx->createShaderModule({codeLeafVS, lvk::Stage_Vert, "Shader Module: leaf (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smLeafFrag =
      ctx->createShaderModule({codeLeafFS, lvk::Stage_Frag, "Shader Module: leaf (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smShadowTrunkVert =
      ctx->createShaderModule({codeShadowTrunkVS, lvk::Stage_Vert, "Shader Module: shadow trunk (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smShadowLeafVert =
      ctx->createShaderModule({codeShadowLeafVS, lvk::Stage_Vert, "Shader Module: shadow leaf (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smShadowLeafFrag =
      ctx->createShaderModule({codeShadowLeafFS, lvk::Stage_Frag, "Shader Module: shadow leaf (frag)"});

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
      .depthFormat = lvk::Format_Z_F32,
      .cullMode = lvk::CullMode_None,
      .samplesCount = kMSAASamples,
      .debugName = "Pipeline: ground",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineGrass = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smGrassVert,
      .smFrag = smGrassFrag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = lvk::Format_Z_F32,
      .cullMode = lvk::CullMode_None,
      .samplesCount = kMSAASamples,
      .debugName = "Pipeline: grass",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineShadowGround = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smShadowGroundVert,
      .smFrag = smShadowFrag,
      .depthFormat = lvk::Format_Z_UN16,
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: shadow ground",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineShadowGrass = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smShadowGrassVert,
      .smFrag = smShadowFrag,
      .depthFormat = lvk::Format_Z_UN16,
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: shadow grass",
  });

  // Depth prepass pipelines (1x, for SSAO depth input — reuse shadow shader modules)
  lvk::Holder<lvk::RenderPipelineHandle> pipelineDepthGround = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smShadowGroundVert,
      .smFrag = smShadowFrag,
      .depthFormat = lvk::Format_Z_F32,
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: depth prepass ground",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineDepthGrass = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smShadowGrassVert,
      .smFrag = smShadowFrag,
      .depthFormat = lvk::Format_Z_F32,
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: depth prepass grass",
  });

  // Tree pipelines (main pass)
  lvk::Holder<lvk::RenderPipelineHandle> pipelineTrunk = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smTrunkVert,
      .smFrag = smTrunkFrag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = lvk::Format_Z_F32,
      .cullMode = lvk::CullMode_Back,
      .samplesCount = kMSAASamples,
      .debugName = "Pipeline: trunk",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineLeaf = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smLeafVert,
      .smFrag = smLeafFrag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = lvk::Format_Z_F32,
      .cullMode = lvk::CullMode_None,
      .samplesCount = kMSAASamples,
      .debugName = "Pipeline: leaf",
  });

  // Tree pipelines (shadow pass)
  lvk::Holder<lvk::RenderPipelineHandle> pipelineShadowTrunk = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smShadowTrunkVert,
      .smFrag = smShadowFrag,
      .depthFormat = lvk::Format_Z_UN16,
      .cullMode = lvk::CullMode_Back,
      .debugName = "Pipeline: shadow trunk",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineShadowLeaf = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smShadowLeafVert,
      .smFrag = smShadowLeafFrag,
      .depthFormat = lvk::Format_Z_UN16,
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: shadow leaf",
  });

  // Tree pipelines (depth prepass for SSAO)
  lvk::Holder<lvk::RenderPipelineHandle> pipelineDepthTrunk = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smShadowTrunkVert,
      .smFrag = smShadowFrag,
      .depthFormat = lvk::Format_Z_F32,
      .cullMode = lvk::CullMode_Back,
      .debugName = "Pipeline: depth prepass trunk",
  });

  lvk::Holder<lvk::RenderPipelineHandle> pipelineDepthLeaf = ctx->createRenderPipeline({
      .topology = lvk::Topology_TriangleStrip,
      .smVert = smShadowLeafVert,
      .smFrag = smShadowLeafFrag,
      .depthFormat = lvk::Format_Z_F32,
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: depth prepass leaf",
  });

  // SSAO compute pipelines
  lvk::Holder<lvk::ComputePipelineHandle> pipelineSSAO = ctx->createComputePipeline({
      .smComp = smSSAOComp,
      .debugName = "Pipeline: SSAO",
  });

  lvk::Holder<lvk::ComputePipelineHandle> pipelineSSAOBlur = ctx->createComputePipeline({
      .smComp = smSSAOBlurComp,
      .debugName = "Pipeline: SSAO blur",
  });

  // Light/shadow constants
  const mat4 scaleBias = mat4(0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1, 0, 0.5, 0.5, 0, 1);

  // ImGui parameters
  float windStrength = 0.5f;
  float windFrequency = 2.0f;
  float windSpeed = 1.5f;
  float gustStrength = 0.3f;
  float gustFrequency = 0.5f;
  float windAngle = 45.0f; // degrees
  float depthBias = -0.0001f;
  float sunElevation = 63.0f; // degrees from horizontal
  float sunAzimuth = 53.0f;   // degrees around Y axis
  bool showShadowMap = false;
  float ssaoRadius = 0.5f;
  float ssaoBias = 0.025f;
  float ssaoIntensity = 1.2f;
  bool ssaoEnabled = true;

  // MSAA textures (recreated on resize)
  lvk::Holder<lvk::TextureHandle> msaaColor;
  lvk::Holder<lvk::TextureHandle> msaaDepth;
  // SSAO textures (recreated on resize)
  lvk::Holder<lvk::TextureHandle> ssaoDepthPrepass;
  lvk::Holder<lvk::TextureHandle> ssaoRaw;
  lvk::Holder<lvk::TextureHandle> ssaoBlurred;
  uint32_t msaaWidth = 0, msaaHeight = 0;

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    // Recreate MSAA textures on resize
    if (msaaWidth != width || msaaHeight != height) {
      msaaWidth = width;
      msaaHeight = height;
      msaaColor = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = ctx->getSwapchainFormat(),
          .dimensions = {width, height},
          .numSamples = kMSAASamples,
          .usage = lvk::TextureUsageBits_Attachment,
          .numMipLevels = 1,
          .debugName = "Texture: MSAA color",
      });
      msaaDepth = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_Z_F32,
          .dimensions = {width, height},
          .numSamples = kMSAASamples,
          .usage = lvk::TextureUsageBits_Attachment,
          .numMipLevels = 1,
          .debugName = "Texture: MSAA depth",
      });
      // SSAO textures (1x resolution)
      ssaoDepthPrepass = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_Z_F32,
          .dimensions = {width, height},
          .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
          .numMipLevels = 1,
          .debugName = "Texture: SSAO depth prepass",
      });
      ssaoRaw = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_F16,
          .dimensions = {width, height},
          .usage = lvk::TextureUsageBits_Storage | lvk::TextureUsageBits_Sampled,
          .debugName = "Texture: SSAO raw",
      });
      ssaoBlurred = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_F16,
          .dimensions = {width, height},
          .usage = lvk::TextureUsageBits_Storage | lvk::TextureUsageBits_Sampled,
          .debugName = "Texture: SSAO blurred",
      });
    }

    const float fov = float(45.0f * (M_PI / 180.0f));
    const float currentTime = (float)glfwGetTime();

    // Compute light direction from sun angles
    const float elevRad = glm::radians(sunElevation);
    const float aziRad = glm::radians(sunAzimuth);
    const vec3 lightDir = normalize(vec3(
        cosf(elevRad) * sinf(aziRad),
        sinf(elevRad),
        cosf(elevRad) * cosf(aziRad)));
    const mat4 lightView = glm::lookAt(lightDir * 40.0f, vec3(0), vec3(0, 1, 0));

    // Compute tight ortho frustum from scene AABB in light-view space
    const vec3 sceneMin(-kFieldSize, -1.0f, -kFieldSize);
    const vec3 sceneMax( kFieldSize,  4.0f,  kFieldSize);
    vec3 lMin(FLT_MAX), lMax(-FLT_MAX);
    for (int i = 0; i < 8; i++) {
      const vec3 corner(
          (i & 1) ? sceneMax.x : sceneMin.x,
          (i & 2) ? sceneMax.y : sceneMin.y,
          (i & 4) ? sceneMax.z : sceneMin.z);
      const vec3 lc = vec3(lightView * vec4(corner, 1.0f));
      lMin = glm::min(lMin, lc);
      lMax = glm::max(lMax, lc);
    }
    const float nearPlane = -lMax.z - 1.0f;
    const float farPlane = -lMin.z + 1.0f;
    mat4 lightProj = glm::ortho(lMin.x, lMax.x, lMin.y, lMax.y, nearPlane, farPlane);
    // Fix for Vulkan [0,1] depth range (GLM defaults to OpenGL [-1,1] which clips the near half of ortho frustums)
    lightProj[2][2] = -1.0f / (farPlane - nearPlane);
    lightProj[3][2] = -nearPlane / (farPlane - nearPlane);

    const mat4 projMatrix = glm::perspective(fov, aspectRatio, 0.1f, 200.0f);

    const PerFrame perFrame = {
        .proj = projMatrix,
        .view = app.camera_.getViewMatrix(),
        .light = scaleBias * lightProj * lightView,
        .time = currentTime,
        .windTex = windTexture.index(),
        .windSamp = windSampler.index(),
        .fieldSize = kFieldSize,
        .gridSize = kGridSize,
        .texShadow = shadowMap.index(),
        .sampShadow = shadowSampler.index(),
        .depthBias = depthBias,
        .lightDirX = lightDir.x,
        .lightDirY = lightDir.y,
        .lightDirZ = lightDir.z,
        .texSSAO = ssaoEnabled ? ssaoBlurred.index() : whiteTex.index(),
        .sampSSAO = ssaoSampler.index(),
    };

    const PerFrame perFrameShadow = {
        .proj = lightProj,
        .view = lightView,
        .light = mat4(1.0f),
        .time = currentTime,
        .windTex = windTexture.index(),
        .windSamp = windSampler.index(),
        .fieldSize = kFieldSize,
        .gridSize = kGridSize,
    };

    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

    buffer.cmdUpdateBuffer(bufPerFrame, perFrame);
    buffer.cmdUpdateBuffer(bufPerFrameShadow, perFrameShadow);

    // 1. Compute pass: generate wind texture
    {
      const float windRad = glm::radians(windAngle);
      const WindParams windParams = {
          .time = currentTime,
          .windStrength = windStrength,
          .windFreq = windFrequency,
          .windSpeed = windSpeed,
          .gustStrength = gustStrength,
          .gustFreq = gustFrequency,
          .windDirX = cosf(windRad),
          .windDirY = sinf(windRad),
          .texOut = windTexture.index(),
          .texSize = kWindTexSize,
      };
      buffer.cmdBindComputePipeline(pipelineWind);
      buffer.cmdPushConstants(windParams);
      buffer.cmdDispatchThreadGroups(
          {(kWindTexSize + 15) / 16, (kWindTexSize + 15) / 16, 1});
    }

    // 2. Shadow pass: render depth from light perspective
    {
      lvk::Framebuffer shadowFramebuffer = {
          .depthStencil = {shadowMap},
      };
      buffer.cmdBeginRendering(
          lvk::RenderPass{
              .color = {},
              .depth = {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f}},
          shadowFramebuffer,
          {.textures = {lvk::TextureHandle(windTexture)}});
      {
        buffer.cmdBindViewport({0.0f, 0.0f, (float)kShadowMapSize, (float)kShadowMapSize, 0.0f, +1.0f});
        buffer.cmdBindScissorRect({0, 0, kShadowMapSize, kShadowMapSize});
        buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});

        // Shadow ground
        {
          buffer.cmdBindRenderPipeline(pipelineShadowGround);
          const struct {
            uint64_t perFrame;
          } shadowGroundPC = {
              .perFrame = ctx->gpuAddress(bufPerFrameShadow),
          };
          buffer.cmdPushConstants(shadowGroundPC);
          buffer.cmdDraw(2 * (kGridSize + 1), kGridSize);
        }

        // Shadow grass
        {
          buffer.cmdBindRenderPipeline(pipelineShadowGrass);
          const struct {
            uint64_t perFrame;
            uint64_t bladeData;
          } shadowGrassPC = {
              .perFrame = ctx->gpuAddress(bufPerFrameShadow),
              .bladeData = ctx->gpuAddress(bufBlades),
          };
          buffer.cmdPushConstants(shadowGrassPC);
          buffer.cmdDraw(kBladesPerStrip, kNumBlades);
        }

        // Shadow trunk
        {
          buffer.cmdBindRenderPipeline(pipelineShadowTrunk);
          const struct {
            uint64_t perFrame;
            uint64_t segmentData;
          } pc = {
              .perFrame = ctx->gpuAddress(bufPerFrameShadow),
              .segmentData = ctx->gpuAddress(bufTrunkSegments),
          };
          buffer.cmdPushConstants(pc);
          buffer.cmdDraw(kTrunkVertsPerSegment, totalTrunkSegments);
        }

        // Shadow leaves
        {
          buffer.cmdBindRenderPipeline(pipelineShadowLeaf);
          const struct {
            uint64_t perFrame;
            uint64_t leafData;
          } pc = {
              .perFrame = ctx->gpuAddress(bufPerFrameShadow),
              .leafData = ctx->gpuAddress(bufLeaves),
          };
          buffer.cmdPushConstants(pc);
          buffer.cmdDraw(kLeafVerts, totalLeaves);
        }
      }
      buffer.cmdEndRendering();
      buffer.transitionToShaderReadOnly(shadowMap);
    }

    // 3. Depth prepass (1x, camera perspective — for SSAO)
    if (ssaoEnabled) {
      lvk::Framebuffer depthFB = {
          .depthStencil = {ssaoDepthPrepass},
      };
      buffer.cmdBeginRendering(
          lvk::RenderPass{
              .color = {},
              .depth = {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f}},
          depthFB,
          {.textures = {lvk::TextureHandle(windTexture)}});
      {
        buffer.cmdBindViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f});
        buffer.cmdBindScissorRect({0, 0, width, height});
        buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});

        // Depth ground (reuses shadow shaders with camera matrices)
        {
          buffer.cmdBindRenderPipeline(pipelineDepthGround);
          const struct {
            uint64_t perFrame;
          } depthGroundPC = {
              .perFrame = ctx->gpuAddress(bufPerFrame),
          };
          buffer.cmdPushConstants(depthGroundPC);
          buffer.cmdDraw(2 * (kGridSize + 1), kGridSize);
        }

        // Depth grass
        {
          buffer.cmdBindRenderPipeline(pipelineDepthGrass);
          const struct {
            uint64_t perFrame;
            uint64_t bladeData;
          } depthGrassPC = {
              .perFrame = ctx->gpuAddress(bufPerFrame),
              .bladeData = ctx->gpuAddress(bufBlades),
          };
          buffer.cmdPushConstants(depthGrassPC);
          buffer.cmdDraw(kBladesPerStrip, kNumBlades);
        }

        // Depth trunk
        {
          buffer.cmdBindRenderPipeline(pipelineDepthTrunk);
          const struct {
            uint64_t perFrame;
            uint64_t segmentData;
          } pc = {
              .perFrame = ctx->gpuAddress(bufPerFrame),
              .segmentData = ctx->gpuAddress(bufTrunkSegments),
          };
          buffer.cmdPushConstants(pc);
          buffer.cmdDraw(kTrunkVertsPerSegment, totalTrunkSegments);
        }

        // Depth leaves
        {
          buffer.cmdBindRenderPipeline(pipelineDepthLeaf);
          const struct {
            uint64_t perFrame;
            uint64_t leafData;
          } pc = {
              .perFrame = ctx->gpuAddress(bufPerFrame),
              .leafData = ctx->gpuAddress(bufLeaves),
          };
          buffer.cmdPushConstants(pc);
          buffer.cmdDraw(kLeafVerts, totalLeaves);
        }
      }
      buffer.cmdEndRendering();
      buffer.transitionToShaderReadOnly(ssaoDepthPrepass);

      // 4. SSAO compute
      {
        const struct {
          uint32_t texDepth;
          uint32_t sampDepth;
          uint32_t texOut;
          uint32_t width;
          uint32_t height;
          float proj00;
          float proj11;
          float proj22;
          float proj32;
          float radius;
          float bias;
          float intensity;
        } ssaoPC = {
            .texDepth = ssaoDepthPrepass.index(),
            .sampDepth = ssaoSampler.index(),
            .texOut = ssaoRaw.index(),
            .width = width,
            .height = height,
            .proj00 = projMatrix[0][0],
            .proj11 = projMatrix[1][1],
            .proj22 = projMatrix[2][2],
            .proj32 = projMatrix[3][2],
            .radius = ssaoRadius,
            .bias = ssaoBias,
            .intensity = ssaoIntensity,
        };
        buffer.cmdBindComputePipeline(pipelineSSAO);
        buffer.cmdPushConstants(ssaoPC);
        buffer.cmdDispatchThreadGroups({(width + 15) / 16, (height + 15) / 16, 1});
      }

      // 5. SSAO blur (reads ssaoRaw as sampled texture, writes ssaoBlurred as storage)
      buffer.transitionToShaderReadOnly(ssaoRaw);
      {
        const struct {
          uint32_t texIn;
          uint32_t sampIn;
          uint32_t texOut;
          uint32_t width;
          uint32_t height;
        } blurPC = {
            .texIn = ssaoRaw.index(),
            .sampIn = ssaoSampler.index(),
            .texOut = ssaoBlurred.index(),
            .width = width,
            .height = height,
        };
        buffer.cmdBindComputePipeline(pipelineSSAOBlur);
        buffer.cmdPushConstants(blurPC);
        buffer.cmdDispatchThreadGroups({(width + 15) / 16, (height + 15) / 16, 1});
      }
    }

    // Main render pass (4x MSAA → resolve to swapchain)
    {
      lvk::Framebuffer framebuffer = {
          .color = {{.texture = msaaColor, .resolveTexture = ctx->getCurrentSwapchainTexture()}},
          .depthStencil = {msaaDepth},
      };
      buffer.cmdBeginRendering(
          lvk::RenderPass{
              .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_MsaaResolve, .clearColor = {0.53f, 0.81f, 0.92f, 1.0f}}},
              .depth = {.loadOp = lvk::LoadOp_Clear, .clearDepth = 1.0}},
          framebuffer,
          {.textures = {lvk::TextureHandle(windTexture), ssaoEnabled ? lvk::TextureHandle(ssaoBlurred) : lvk::TextureHandle(whiteTex)}});
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
          buffer.cmdDraw(2 * (kGridSize + 1), kGridSize);
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

        // Draw tree trunks
        {
          buffer.cmdBindRenderPipeline(pipelineTrunk);
          const struct {
            uint64_t perFrame;
            uint64_t segmentData;
          } pc = {
              .perFrame = ctx->gpuAddress(bufPerFrame),
              .segmentData = ctx->gpuAddress(bufTrunkSegments),
          };
          buffer.cmdPushConstants(pc);
          buffer.cmdDraw(kTrunkVertsPerSegment, totalTrunkSegments);
        }

        // Draw tree leaves
        {
          buffer.cmdBindRenderPipeline(pipelineLeaf);
          const struct {
            uint64_t perFrame;
            uint64_t leafData;
          } pc = {
              .perFrame = ctx->gpuAddress(bufPerFrame),
              .leafData = ctx->gpuAddress(bufLeaves),
          };
          buffer.cmdPushConstants(pc);
          buffer.cmdDraw(kLeafVerts, totalLeaves);
        }
      }
      buffer.cmdEndRendering();
    }

    // ImGui pass (1x, draws over resolved swapchain)
    {
      lvk::Framebuffer framebuffer = {
          .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
      };
      buffer.cmdBeginRendering(
          lvk::RenderPass{
              .color = {{.loadOp = lvk::LoadOp_Load, .storeOp = lvk::StoreOp_Store}}},
          framebuffer);
      {
        app.imgui_->beginFrame(framebuffer);
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
        ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
        ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Wind");
        ImGui::SliderFloat("Wind Strength", &windStrength, 0.0f, 2.0f);
        ImGui::SliderFloat("Wind Direction", &windAngle, 0.0f, 360.0f, "%.0f deg");
        ImGui::SliderFloat("Wind Frequency", &windFrequency, 0.5f, 5.0f);
        ImGui::SliderFloat("Wind Speed", &windSpeed, 0.5f, 5.0f);
        ImGui::SliderFloat("Gust Strength", &gustStrength, 0.0f, 1.0f);
        ImGui::SliderFloat("Gust Frequency", &gustFrequency, 0.1f, 2.0f);
        ImGui::Separator();
        ImGui::Text("Sun / Shadow");
        ImGui::SliderFloat("Sun Elevation", &sunElevation, 5.0f, 89.0f, "%.0f deg");
        ImGui::SliderFloat("Sun Azimuth", &sunAzimuth, 0.0f, 360.0f, "%.0f deg");
        ImGui::SliderFloat("Depth Bias", &depthBias, -0.001f, 0.001f, "%.5f");
        ImGui::Checkbox("Show Shadow Map", &showShadowMap);
        if (showShadowMap) {
          ImGui::Image(shadowMap.index(), ImVec2(256, 256));
        }
        ImGui::Separator();
        ImGui::Text("SSAO");
        ImGui::Checkbox("Enable SSAO", &ssaoEnabled);
        if (ssaoEnabled) {
          ImGui::SliderFloat("SSAO Radius", &ssaoRadius, 0.1f, 2.0f);
          ImGui::SliderFloat("SSAO Bias", &ssaoBias, 0.001f, 0.1f, "%.3f");
          ImGui::SliderFloat("SSAO Intensity", &ssaoIntensity, 0.5f, 3.0f);
        }
        ImGui::End();
        app.drawFPS();
        app.imgui_->endFrame(buffer);
      }
      buffer.cmdEndRendering();
    }

    ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
  });

  VULKAN_APP_EXIT();
}
