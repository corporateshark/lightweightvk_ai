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
  float padding2;
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
  float padding2;
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

  vec3 color = albedo * (ambient + diffuse * shd) + sunColor * spec * shd;
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
  float padding2;
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
  v_AO = mix(0.4, 1.0, t); // ambient occlusion: darker at base
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
  float padding2;
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

  vec3 color = v_Color * (ambient + diffuse * shd) * v_AO * windAO
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
  float padding2;
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
  float padding2;
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
  float padding2;
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
  constexpr uint32_t kShadowMapSize = 2048;

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

  // MSAA textures (recreated on resize)
  lvk::Holder<lvk::TextureHandle> msaaColor;
  lvk::Holder<lvk::TextureHandle> msaaDepth;
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
    const vec3 sceneMax( kFieldSize,  2.0f,  kFieldSize);
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

    const PerFrame perFrame = {
        .proj = glm::perspective(fov, aspectRatio, 0.1f, 200.0f),
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
      }
      buffer.cmdEndRendering();
      buffer.transitionToShaderReadOnly(shadowMap);
    }

    // 3. Main render pass (4x MSAA → resolve to swapchain)
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
      }
      buffer.cmdEndRendering();
    }

    // 4. ImGui pass (1x, draws over resolved swapchain)
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
