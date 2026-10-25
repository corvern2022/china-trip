(function(){function e(e,t){return[...new TextEncoder().encode(e)].map(e=>{let n=t.get(`<0x${e.toString(16).toUpperCase().padStart(2,`0`)}>`);if(n===void 0)throw Error(`모델에 필요한 문자 토큰이 없어요`);return n})}function t(e){let t=``,n=[],r=e=>{n.length&&(t+=new TextDecoder().decode(new Uint8Array(n),{stream:e})),n=[]};for(let i of e){let e=/^<0x([0-9A-Fa-f]{2})>$/.exec(i);e?n.push(parseInt(e[1],16)):(r(!1),t+=i.replace(/▁/g,` `))}return r(!0),t}function n(e){let t=e&32768?-1:1,n=e>>>10&31,r=e&1023;return n===0?t*r*2**-24:n===31?r?NaN:t*(1/0):t*(1+r/1024)*2**(n-15)}function r(e){if(e.length%34)throw Error(`잘못된 Q8 블록 크기`);let t=new Uint8Array(e.length/34*36),r=new DataView(e.buffer,e.byteOffset,e.byteLength),i=new DataView(t.buffer);for(let a=0,o=0;a<e.length;a+=34,o+=36)i.setFloat32(o,n(r.getUint16(a,!0)),!0),t.set(e.subarray(a+2,a+34),o+4);return t}async function i(e,t,n,i){if(!Number.isSafeInteger(n)||n<=0||n%32||!t.body)throw Error(`잘못된 Q8 텐서`);let a=n/32*36,o=e.createBuffer({size:a,usage:i}),s=t.body.getReader(),c=/* @__PURE__ */ new Uint8Array,l=0;try{for(;;){let{value:t,done:n}=await s.read();if(n)break;let i=new Uint8Array(c.length+t.length);i.set(c),i.set(t,c.length);let u=i.length-i.length%34;if(c=i.slice(u),!u)continue;let d=r(i.subarray(0,u));if(l+d.length>a)throw Error(`Q8 텐서가 예상보다 커요`);e.queue.writeBuffer(o,l,d),l+=d.length,await e.queue.onSubmittedWorkDone()}if(c.length||l!==a)throw Error(`Q8 텐서가 중간에 끊겼어요`);return o}catch(e){throw o.destroy(),await s.cancel().catch(()=>{}),e}finally{s.releaseLock()}}var a={embeddingLookup:`
struct Params { hidden_size: u32, token_id: u32 }
@group(0) @binding(0) var<storage, read> embedding_q8: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.hidden_size) { return; }
  let blocks_per_row = params.hidden_size / 32u;
  let row_offset = params.token_id * blocks_per_row * 9u;
  let block_idx = i / 32u;
  let elem_in_block = i % 32u;
  let block_off = row_offset + block_idx * 9u;
  let scale = bitcast<f32>(embedding_q8[block_off]);
  let packed_idx = elem_in_block / 4u;
  let byte_idx = elem_in_block % 4u;
  let packed = embedding_q8[block_off + 1u + packed_idx];
  let q = f32(extractBits(bitcast<i32>(packed), byte_idx * 8u, 8u));
  output[i] = q * scale * sqrt(f32(params.hidden_size));
}`,rmsNorm:`
struct Params { hidden_size: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let hidden_size = params.hidden_size;
  var partial_sum: f32 = 0.0;
  var i = tid;
  while (i < hidden_size) {
    let val = input[i];
    partial_sum += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial_sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(hidden_size) + params.eps);
  i = tid;
  while (i < hidden_size) {
    output[i] = input[i] * weight[i] / rms;
    i += 256u;
  }
}`,perHeadRmsNorm:`
struct Params { num_heads: u32, head_dim: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let base = head * params.head_dim;
  var partial: f32 = 0.0;
  var i = tid;
  while (i < params.head_dim) {
    let val = data[base + i];
    partial += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(params.head_dim) + params.eps);
  i = tid;
  while (i < params.head_dim) {
    data[base + i] = data[base + i] * weight[i] / rms;
    i += 256u;
  }
}`,linearQ8:`
struct Params { M: u32, N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight_q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_tile: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = gid.x;
  let tid = lid.x;
  let N = params.N;
  let blocks_per_row = N / 32u;
  let row_offset = m * blocks_per_row * 9u;
  var total_sum: f32 = 0.0;
  let TILE: u32 = 2048u;

  var tile_start: u32 = 0u;
  while (tile_start < N) {
    let tile_size = min(TILE, N - tile_start);
    var i = tid;
    while (i < tile_size) {
      shared_tile[i] = input[tile_start + i];
      i += 256u;
    }
    workgroupBarrier();
    if (m < params.M) {
      let block_start = tile_start / 32u;
      let block_end = (tile_start + tile_size) / 32u;
      for (var b = block_start; b < block_end; b++) {
        let block_off = row_offset + b * 9u;
        let scale = bitcast<f32>(weight_q8[block_off]);
        let local_base = b * 32u - tile_start;
        var block_sum: f32 = 0.0;
        for (var chunk: u32 = 0u; chunk < 4u; chunk++) {
          let c = local_base + chunk * 8u;
          let qs_base = block_off + 1u + chunk * 2u;
          let packed0 = weight_q8[qs_base];
          let packed1 = weight_q8[qs_base + 1u];
          block_sum += f32(extractBits(bitcast<i32>(packed0), 0u, 8u))  * shared_tile[c + 0u];
          block_sum += f32(extractBits(bitcast<i32>(packed0), 8u, 8u))  * shared_tile[c + 1u];
          block_sum += f32(extractBits(bitcast<i32>(packed0), 16u, 8u)) * shared_tile[c + 2u];
          block_sum += f32(extractBits(bitcast<i32>(packed0), 24u, 8u)) * shared_tile[c + 3u];
          block_sum += f32(extractBits(bitcast<i32>(packed1), 0u, 8u))  * shared_tile[c + 4u];
          block_sum += f32(extractBits(bitcast<i32>(packed1), 8u, 8u))  * shared_tile[c + 5u];
          block_sum += f32(extractBits(bitcast<i32>(packed1), 16u, 8u)) * shared_tile[c + 6u];
          block_sum += f32(extractBits(bitcast<i32>(packed1), 24u, 8u)) * shared_tile[c + 7u];
        }
        total_sum += block_sum * scale;
      }
    }
    workgroupBarrier();
    tile_start += TILE;
  }
  if (m < params.M) {
    output[m] = total_sum;
  }
}`,geluMul:`
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEF_A: f32 = 0.044715;

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= size) { return; }
  let x = gate[i];
  let tanh_arg = clamp(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x), -15.0, 15.0);
  let gelu = 0.5 * x * (1.0 + tanh(tanh_arg));
  output[i] = gelu * up[i];
}`,add:`
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= size) { return; }
  output[i] = a[i] + b[i];
}`,rope:`
struct Params { num_heads: u32, head_dim: u32, position: u32, theta: f32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.num_heads * half_dim;
  if (idx >= total_pairs) { return; }
  let head = idx / half_dim;
  let i = idx % half_dim;
  let base = head * params.head_dim;
  let freq = 1.0 / pow(params.theta, f32(i * 2u) / f32(params.head_dim));
  let angle = f32(params.position) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);
  let x0 = data[base + i];
  let x1 = data[base + i + half_dim];
  data[base + i] = x0 * cos_a - x1 * sin_a;
  data[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
}`,kvCacheStore:`
struct Params { num_kv_heads: u32, head_dim: u32, position: u32, max_seq_len: u32 }
@group(0) @binding(0) var<storage, read> k_in: array<f32>;
@group(0) @binding(1) var<storage, read> v_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.num_kv_heads * params.head_dim;
  if (i >= total) { return; }
  let head = i / params.head_dim;
  let d = i % params.head_dim;
  let cache_idx = params.position * total + head * params.head_dim + d;
  k_cache[cache_idx] = k_in[i];
  v_cache[cache_idx] = v_in[i];
}`,attnScore:`
struct Params { num_q_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32, scale: f32, pad0: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.num_q_heads * params.seq_len;
  if (idx >= total) { return; }
  let head = idx / params.seq_len;
  let pos = idx % params.seq_len;
  let kv_head = head * params.num_kv_heads / params.num_q_heads;
  let q_offset = head * params.head_dim;
  let kv_stride = params.num_kv_heads * params.head_dim;
  let k_offset = pos * kv_stride + kv_head * params.head_dim;
  var dot: f32 = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[q_offset + d] * k_cache[k_offset + d];
  }
  scores[head * params.seq_len + pos] = dot * params.scale;
}`,softmax:`
struct Params { num_heads: u32, seq_len: u32 }
@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let base = head * params.seq_len;
  var local_max: f32 = -1e30;
  var i = tid;
  while (i < params.seq_len) {
    local_max = max(local_max, scores[base + i]);
    i += 256u;
  }
  shared_max[tid] = local_max;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]); }
    workgroupBarrier();
  }
  let max_val = shared_max[0];
  var local_sum: f32 = 0.0;
  i = tid;
  while (i < params.seq_len) {
    let e = exp(scores[base + i] - max_val);
    scores[base + i] = e;
    local_sum += e;
    i += 256u;
  }
  shared_sum[tid] = local_sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let sum_val = shared_sum[0];
  i = tid;
  while (i < params.seq_len) {
    scores[base + i] = scores[base + i] / sum_val;
    i += 256u;
  }
}`,attnOutput:`
struct Params { num_q_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.num_q_heads * params.head_dim;
  if (idx >= total) { return; }
  let head = idx / params.head_dim;
  let d = idx % params.head_dim;
  let kv_head = head * params.num_kv_heads / params.num_q_heads;
  let kv_stride = params.num_kv_heads * params.head_dim;
  var sum: f32 = 0.0;
  for (var pos: u32 = 0u; pos < params.seq_len; pos++) {
    let prob = probs[head * params.seq_len + pos];
    let v_idx = pos * kv_stride + kv_head * params.head_dim + d;
    sum += prob * v_cache[v_idx];
  }
  output[head * params.head_dim + d] = sum;
}`,argmax:`
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> size: u32;
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var local_max: f32 = -1e30;
  var local_idx: u32 = 0u;
  var i = tid;
  while (i < size) {
    let val = logits[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
    i += 256u;
  }
  shared_max[tid] = local_max;
  shared_idx[tid] = local_idx;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride && shared_max[tid + stride] > shared_max[tid]) {
      shared_max[tid] = shared_max[tid + stride];
      shared_idx[tid] = shared_idx[tid + stride];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    result[0] = shared_idx[0];
  }
}`,topk256:`
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var local_max: f32 = -1e30;
  var local_idx: u32 = 0u;
  var i = tid;
  while (i < size) {
    let val = logits[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
    i += 256u;
  }
  result[tid * 2u] = local_max;
  result[tid * 2u + 1u] = bitcast<f32>(local_idx);
}`,fusedNormAdd:`
struct Params { size: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let N = params.size;
  var partial: f32 = 0.0;
  var i = tid;
  while (i < N) {
    let val = input[i];
    partial += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(N) + params.eps);
  i = tid;
  while (i < N) {
    output[i] = residual[i] + input[i] * weight[i] / rms;
    i += 256u;
  }
}`,fusedPerHeadNormRope:`
struct Params { num_heads: u32, head_dim: u32, eps: f32, position: u32, theta: f32, pad: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let tid = lid.x;
  let base = head * params.head_dim;
  var partial: f32 = 0.0;
  var i = tid;
  while (i < params.head_dim) {
    let val = data[base + i];
    partial += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(params.head_dim) + params.eps);
  i = tid;
  while (i < params.head_dim) {
    data[base + i] = data[base + i] * weight[i] / rms;
    i += 256u;
  }
  workgroupBarrier();
  let half_dim = params.head_dim / 2u;
  i = tid;
  while (i < half_dim) {
    let freq = 1.0 / pow(params.theta, f32(i * 2u) / f32(params.head_dim));
    let angle = f32(params.position) * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let x0 = data[base + i];
    let x1 = data[base + i + half_dim];
    data[base + i] = x0 * cos_a - x1 * sin_a;
    data[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
    i += 256u;
  }
}`},o=class{buffer;view;offset;textDecoder;constructor(e){e instanceof Uint8Array?(this.buffer=e.buffer,this.view=new DataView(this.buffer,e.byteOffset,e.byteLength),this.offset=0):(this.buffer=e,this.view=new DataView(this.buffer),this.offset=0),this.textDecoder=new TextDecoder(`utf-8`)}readUint32(){let e=this.view.getUint32(this.offset,!0);return this.offset+=4,e}readUint64(){let e=this.view.getBigUint64(this.offset,!0);return this.offset+=8,e}readString(){let e=Number(this.readUint64()),t=new Uint8Array(this.buffer,this.offset,e);return this.offset+=e,this.textDecoder.decode(t)}readValue(e){switch(e){case 0:{let e=this.view.getUint8(this.offset);return this.offset+=1,{type:`uint8`,value:e}}case 1:{let e=this.view.getInt8(this.offset);return this.offset+=1,{type:`int8`,value:e}}case 2:{let e=this.view.getUint16(this.offset,!0);return this.offset+=2,{type:`uint16`,value:e}}case 3:{let e=this.view.getInt16(this.offset,!0);return this.offset+=2,{type:`int16`,value:e}}case 4:return{type:`uint32`,value:this.readUint32()};case 5:{let e=this.view.getInt32(this.offset,!0);return this.offset+=4,{type:`int32`,value:e}}case 6:{let e=this.view.getFloat32(this.offset,!0);return this.offset+=4,{type:`float32`,value:e}}case 7:{let e=this.view.getUint8(this.offset);return this.offset+=1,{type:`bool`,value:e!==0}}case 8:return{type:`string`,value:this.readString()};case 9:{let e=this.readUint32(),t=Number(this.readUint64()),n=[];for(let r=0;r<t;r++)n.push(this.readValue(e));return{type:`array`,value:n}}case 10:return{type:`uint64`,value:this.readUint64()};case 11:{let e=this.view.getBigInt64(this.offset,!0);return this.offset+=8,{type:`int64`,value:e}}case 12:{let e=this.view.getFloat64(this.offset,!0);return this.offset+=8,{type:`float64`,value:e}}default:return{type:`unknown`,value:null}}}parse(){let e=this.readUint32();if(e!==1179993927)throw Error(`Invalid GGUF magic: 0x${e.toString(16)}`);let t=this.readUint32(),n=this.readUint64(),r=this.readUint64(),i=/* @__PURE__ */ new Map;for(let e=0n;e<r;e++){let e=this.readString(),t=this.readUint32(),n=this.readValue(t);i.set(e,n)}let a=[];for(let e=0n;e<n;e++){let e=this.readString(),t=this.readUint32(),n=[];for(let e=0;e<t;e++)n.push(this.readUint64());let r=this.readUint32(),i=this.readUint64();a.push({name:e,dims:n,type:r,offset:i})}return{version:t,tensor_count:n,kv_count:r,kv:i,tensors:a,dataOffset:Math.ceil(this.offset/32)*32}}f16ToF32(e){let t=e>>15&1,n=e>>10&31,r=e&1023;return n===0?r===0?t?-0:0:(t?-1:1)*2**-14*(r/1024):n===31?r===0?t?-1/0:1/0:NaN:(t?-1:1)*2**(n-15)*(1+r/1024)}dequantizeQ8_0(e,t){let n=new Float32Array(t),r=0,i=e;for(;r<t;){let e=this.view.getUint16(i,!0),a=this.f16ToF32(e);i+=2;let o=Math.min(32,t-r);for(let e=0;e<o;e++){let t=this.view.getInt8(i+e);n[r++]=t*a}i+=32}return n}packQ8_0ForGPU(e,t){let n=t/32,r=/* @__PURE__ */ new ArrayBuffer(n*36),i=new Uint32Array(r),a=new Float32Array(r),o=e;for(let e=0;e<n;e++){let t=this.view.getUint16(o,!0),n=this.f16ToF32(t);a[e*9]=n,o+=2;for(let t=0;t<8;t++){let n=0;for(let e=0;e<4;e++){let r=this.view.getUint8(o+t*4+e);n|=r<<e*8}i[e*9+1+t]=n}o+=32}return new Uint32Array(r)}dequantizeF16(e,t){let n=new Float32Array(t);for(let r=0;r<t;r++){let t=this.view.getUint16(e+r*2,!0);n[r]=this.f16ToF32(t)}return n}getTensorData(e,t){let n=t+Number(e.offset),r=Number(e.dims.reduce((e,t)=>e*t,1n));if(e.type===0)return new Float32Array(this.buffer,n,r);if(e.type===1)return this.dequantizeF16(n,r);if(e.type===8)return this.dequantizeQ8_0(n,r);throw Error(`Unsupported type: ${e.type}`)}getTensorDataQ8Packed(e,t){let n=t+Number(e.offset),r=Number(e.dims.reduce((e,t)=>e*t,1n));return this.packQ8_0ForGPU(n,r)}};function s(e){let t=Number(e.dims.reduce((e,t)=>e*t,1n));if(e.type===8)return t/32*34;if(e.type===0)return t*4;if(e.type===1)return t*2;throw Error(`Unknown tensor type: ${e.type}`)}function c(e,t,n){let r=t=>{let n=e.kv.get(t);return n?(n.value,Number(n.value)):null},i=r(`gemma3.embedding_length`),a=r(`gemma3.block_count`),o=r(`gemma3.feed_forward_length`),s=r(`gemma3.attention.head_count`),c=r(`gemma3.attention.head_count_kv`),l=r(`gemma3.attention.key_length`),u=r(`gemma3.context_length`);i!==null&&(t.hidden_size=i),a!==null&&(t.num_layers=a),o!==null&&(t.intermediate_size=o),s!==null&&(t.num_q_heads=s),c!==null&&(t.num_kv_heads=c),l!==null&&(t.head_dim=l);let d=n??2048;return u!==null&&(t.context_length=Math.min(Number(u),d)),t.q_dim=t.num_q_heads*t.head_dim,t.kv_dim=t.num_kv_heads*t.head_dim,t.num_layers<=18?`270M`:`1B`}var l={"<start_of_turn>":105,"<end_of_turn>":106,"<eos>":1,"<bos>":2},u=[`<start_function_declaration>`,`<end_function_declaration>`,`<start_function_call>`,`<end_function_call>`,`<start_function_response>`,`<end_function_response>`,`<escape>`],d=class{vocab=[];vocabByLength=[];tokenByText=/* @__PURE__ */ new Map;maxTokenLen=0;specialTokens={...l};funcTokens={};specialPatternRegex=/\\<start_of_turn\\>|\\<end_of_turn\\>|\\<eos\\>|\\<bos\\>/g;extractFromGGUF(e){let t=e.kv.get(`tokenizer.ggml.tokens`);if(t&&t.type===`array`)this.vocab=t.value.map(e=>e.value);else throw Error(`No tokenizer found in GGUF metadata`);this.vocabByLength=[];for(let e=0;e<this.vocab.length;e++)this.vocab[e]&&this.vocab[e].length>0&&this.vocabByLength.push([e,this.vocab[e]]);this.buildTokenIndex(),this.initFunctionTokens()}buildTokenIndex(){this.tokenByText=/* @__PURE__ */ new Map,this.maxTokenLen=0;for(let e=0;e<this.vocab.length;e++){let t=this.vocab[e];t&&t.length>0&&(this.tokenByText.has(t)||this.tokenByText.set(t,e),t.length>this.maxTokenLen&&(this.maxTokenLen=t.length))}}initFunctionTokens(){for(let e of u){let t=this.tokenByText.get(e);t!==void 0&&(this.funcTokens[e]=t,this.specialTokens[e]=t)}this.rebuildSpecialPattern()}rebuildSpecialPattern(){let e=Object.keys(this.specialTokens);e.sort((e,t)=>t.length-e.length);let t=e.map(e=>e.replace(/[<>]/g,e=>`\\`+e)).join(`|`);this.specialPatternRegex=new RegExp(t,`g`)}encodeSegment(t,n=!0){let r=[],i=t.replace(/ /g,`▁`);for(n&&(i=`▁`+i);i.length>0;){let t=0,n=-1,a=Math.min(i.length,this.maxTokenLen);for(let e=a;e>=1;e--){let r=i.substring(0,e),a=this.tokenByText.get(r);if(a!==void 0){t=e,n=a;break}}t===0?(r.push(...e(String.fromCodePoint(i.codePointAt(0)),this.tokenByText)),i=i.slice(String.fromCodePoint(i.codePointAt(0)).length)):(r.push(n),i=i.slice(t))}return r}encode(e){let t=[2],n=new RegExp(this.specialPatternRegex.source,`g`),r=0,i,a=!1;for(;(i=n.exec(e))!==null;){let n=e.slice(r,i.index);n.length>0&&t.push(...this.encodeSegment(n,!a)),t.push(this.specialTokens[i[0]]),r=i.index+i[0].length,a=!0}let o=e.slice(r);return o.length>0&&t.push(...this.encodeSegment(o,!a)),t}decodeToken(e){return e<this.vocab.length&&this.vocab[e]?this.vocab[e].replace(/\u2581/g,` `):`<unk:${e}>`}decodeTokens(e){return t(e.map(e=>this.vocab[e]??``))}};function f(e,t){let n=``,r=null;try{if(t){let e=JSON.parse(t);Array.isArray(e)&&e.length>0&&(r=e)}}catch{r=null}if(r){let e=``;for(let t of r){if(e+=`<start_function_declaration>declaration:${t.name}{`,e+=`description:<escape>${t.description}<escape>`,t.parameters){e+=`,parameters:{properties:{`;let n=Object.entries(t.parameters.properties||{});e+=n.map(([e,t])=>{let n=`${e}:{description:<escape>${t.description}<escape>,type:<escape>${t.type}<escape>`;return t.enum&&(n+=`,enum:[${t.enum.map(e=>`<escape>${e}<escape>`).join(`,`)}]`),n+=`}`,n}).join(`,`),e+=`}`,t.parameters.required&&(e+=`,required:[${t.parameters.required.map(e=>`<escape>${e}<escape>`).join(`,`)}]`),e+=`,type:<escape>${t.parameters.type}<escape>`,e+=`}`}e+=`}<end_function_declaration>`}n=`<start_of_turn>developer
You are a model that can do function calling with the following functions
${e}
<end_of_turn>
`}let i=n;for(let t of e)i+=`<start_of_turn>${t.role}
${t.text}<end_of_turn>
`;return i+=`<start_of_turn>model
`,i}function p(){return{hidden_size:640,q_dim:1024,kv_dim:256,num_q_heads:4,num_kv_heads:1,head_dim:256,intermediate_size:2048,vocab_size:262144,num_layers:18,context_length:2048,rms_norm_eps:1e-6,rope_theta_global:1e6,rope_theta_swa:1e6,swa_period:6}}function m(e,t){return e%t.swa_period<t.swa_period-1}function h(e,t){return m(e,t)?t.rope_theta_swa:t.rope_theta_global}var g={"1b":`https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q8_0.gguf`,"270m":`https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf`,func:`https://huggingface.co/unsloth/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-Q8_0.gguf`},_=class{config;device;pipelines;modelBuffers;workBuffers;uniformBuffers;kvCaches;bindGroupCache;tokenizer;conversationHistory=[];kvPosition=0;onProgress;deviceLost=!1;constructor(e){this.config=p(),e.contextLength&&(this.config.context_length=e.contextLength),this.onProgress=e.onProgress}async init(e){this.config.rope_theta_swa=e.localRopeTheta??1e4,await this.initWebGPU();let t=e.model||`1b`,n=g[t]||t;this.reportProgress(0,1,`Downloading header...`);let r=await fetch(n,{headers:{Range:`bytes=0-20971519`}});if(r.status===206){let t=new o(new Uint8Array(await r.arrayBuffer())).parse();c(t,this.config,e.contextLength),this.tokenizer=new d,this.tokenizer.extractFromGGUF(t);let i=t.tensors,a=t.dataOffset;this.createPipelines(),this.createUniformBuffers(),await this.uploadWeightsStreaming(n,i,a)}else{let t=r.headers.get(`content-length`),n=t?parseInt(t):11e8,i=r.body.getReader(),a=new Uint8Array(n),s=0;for(;;){let{done:e,value:t}=await i.read();if(e)break;a.set(t,s),s+=t.length,this.reportProgress(s,n,`Downloading model...`)}let l=new o(a),u=l.parse();c(u,this.config,e.contextLength),this.tokenizer=new d,this.tokenizer.extractFromGGUF(u),this.createPipelines(),this.createUniformBuffers(),this.uploadWeightsFromBuffer(l,u),await this.device.queue.onSubmittedWorkDone()}this.createWorkBuffers(),this.createBindGroups()}wb(e,t,n){this.device.queue.writeBuffer(e,t,n)}reportProgress(e,t,n){this.onProgress&&this.onProgress({loaded:e,total:t,status:n})}async initWebGPU(){if(!navigator.gpu)throw Error(`WebGPU not supported`);let e=await navigator.gpu.requestAdapter({powerPreference:`high-performance`});if(!e)throw Error(`No WebGPU adapter found`);this.device=await e.requestDevice({requiredLimits:{maxStorageBufferBindingSize:e.limits.maxStorageBufferBindingSize,maxBufferSize:e.limits.maxBufferSize}}),this.device.lost.then(e=>{this.deviceLost=!0,console.error(`WebGPU device lost: ${e.message} (reason: ${e.reason})`)})}createPipelines(){this.pipelines={};for(let[e,t]of Object.entries(a)){let n=this.device.createShaderModule({code:t});this.pipelines[e]=this.device.createComputePipeline({layout:`auto`,compute:{module:n,entryPoint:`main`}})}}makeUniformMixed(e){let t=Math.max(e.length*4,16),n=new ArrayBuffer(t),r=new Uint32Array(n),i=new Float32Array(n);for(let t=0;t<e.length;t++){let n=e[t];`u`in n?r[t]=n.u:i[t]=n.f}let a=this.device.createBuffer({size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(a.getMappedRange()).set(new Uint8Array(n)),a.unmap(),a}createUniformBuffers(){let e=this.config,t=e.hidden_size,n=e.q_dim,r=e.kv_dim,i=e.intermediate_size,a=e.context_length,o=e.vocab_size,s=e.head_dim,c=e.num_q_heads,l=e.num_kv_heads;this.uniformBuffers={rmsNorm:this.makeUniformMixed([{u:t},{f:e.rms_norm_eps}]),perHeadRmsNormQ:this.makeUniformMixed([{u:c},{u:s},{f:e.rms_norm_eps},{u:0}]),perHeadRmsNormK:this.makeUniformMixed([{u:l},{u:s},{f:e.rms_norm_eps},{u:0}]),linearQ8_Q_H:this.makeUniformMixed([{u:n},{u:t}]),linearQ8_KV_H:this.makeUniformMixed([{u:r},{u:t}]),linearQ8_H_Q:this.makeUniformMixed([{u:t},{u:n}]),linearQ8_I_H:this.makeUniformMixed([{u:i},{u:t}]),linearQ8_H_I:this.makeUniformMixed([{u:t},{u:i}]),sizeH:this.makeUniformMixed([{u:t}]),sizeI:this.makeUniformMixed([{u:i}]),embeddingLookup:this.makeUniformMixed([{u:t},{u:0}]),ropeQ:[],ropeK:[],kvCacheStore:this.makeUniformMixed([{u:l},{u:s},{u:0},{u:a}]),attnScore:this.makeUniformMixed([{u:c},{u:l},{u:s},{u:0},{f:1/Math.sqrt(s)},{u:0}]),softmax:this.makeUniformMixed([{u:c},{u:0}]),attnOutput:this.makeUniformMixed([{u:c},{u:l},{u:s},{u:0}]),linearQ8_V_H:this.makeUniformMixed([{u:o},{u:t}]),argmaxSize:this.makeUniformMixed([{u:o}]),fusedNormRopeQ:[],fusedNormRopeK:[]};for(let t=0;t<e.num_layers;t++){let n=h(t,e);this.uniformBuffers.ropeQ.push(this.makeUniformMixed([{u:c},{u:s},{u:0},{f:n}])),this.uniformBuffers.ropeK.push(this.makeUniformMixed([{u:l},{u:s},{u:0},{f:n}])),this.uniformBuffers.fusedNormRopeQ.push(this.makeUniformMixed([{u:c},{u:s},{f:e.rms_norm_eps},{u:0},{f:n},{u:0}])),this.uniformBuffers.fusedNormRopeK.push(this.makeUniformMixed([{u:l},{u:s},{f:e.rms_norm_eps},{u:0},{f:n},{u:0}]))}}uploadWeightsFromBuffer(e,t){let n=t.tensors,r=t.dataOffset;this.modelBuffers={embeddingQ8:null,layers:[],finalNorm:null};let i=n.find(e=>e.name===`token_embd.weight`);if(i){if(i.type===8){let t=e.getTensorDataQ8Packed(i,r);this.modelBuffers.embeddingQ8=this.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(this.modelBuffers.embeddingQ8,0,t)}else{let t=e.getTensorData(i,r);this.modelBuffers.embeddingQ8=this.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(this.modelBuffers.embeddingQ8,0,t)}}let a=n.find(e=>e.name===`output_norm.weight`);if(a){let t=e.getTensorData(a,r);this.modelBuffers.finalNorm=this.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(this.modelBuffers.finalNorm,0,t)}let o=[`attn_q`,`attn_k`,`attn_v`,`attn_output`,`ffn_gate`,`ffn_up`,`ffn_down`],s=[`attn_norm`,`ffn_norm`,`attn_q_norm`,`attn_k_norm`,`post_attention_norm`,`post_ffw_norm`];for(let t=0;t<this.config.num_layers;t++){let i=`blk.${t}.`,a={};for(let t of s){let o=n.find(e=>e.name===i+t+`.weight`);if(o){let n=e.getTensorData(o,r);a[t]=this.device.createBuffer({size:n.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(a[t],0,n)}}for(let t of o){let o=n.find(e=>e.name===i+t+`.weight`);if(o){let n=e.getTensorDataQ8Packed(o,r);a[t]=this.device.createBuffer({size:n.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(a[t],0,n)}}this.modelBuffers.layers.push(a)}}async uploadWeightsStreaming(e,t,n){this.modelBuffers={embeddingQ8:null,layers:[],finalNorm:null};let r=async(t,n)=>{let r=await fetch(e,{headers:{Range:`bytes=${t}-${t+n-1}`}});return new Uint8Array(await r.arrayBuffer())},a=(e,t,n)=>{let r=new o(e),i=Number(n.dims.reduce((e,t)=>e*t,1n));if(n.type===8){let e=r.packQ8_0ForGPU(t,i),n=this.device.createBuffer({size:e.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});return this.wb(n,0,e),{buf:n,packed:!0,byteLength:e.byteLength,numElements:i}}{let a=i,o;if(n.type===0){o=new Float32Array(a);let n=new Uint8Array(e.buffer,e.byteOffset+t,a*4);new Uint8Array(o.buffer).set(n)}else o=r.dequantizeF16(t,a);let s=this.device.createBuffer({size:o.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});return this.wb(s,0,o),{buf:s,packed:!1,byteLength:o.byteLength,numElements:i}}},c=0,l=t.reduce((e,t)=>e+s(t),0),u=[`attn_q`,`attn_k`,`attn_v`,`attn_output`,`ffn_gate`,`ffn_up`,`ffn_down`],d=[`attn_norm`,`ffn_norm`,`attn_q_norm`,`attn_k_norm`,`post_attention_norm`,`post_ffw_norm`],f=t.find(e=>e.name===`token_embd.weight`);if(f){let t=n+Number(f.offset),r=s(f);if(f.type!==8)throw Error(`Expected pinned Q8 embedding`);let a=await fetch(e,{headers:{Range:`bytes=${t}-${t+r-1}`}});this.modelBuffers.embeddingQ8=await i(this.device,a,Number(f.dims.reduce((e,t)=>e*t,1n)),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),c+=r,this.reportProgress(c,l,`Streaming weights to GPU...`)}let p=t.find(e=>e.name===`output_norm.weight`);if(p){let e=n+Number(p.offset),t=s(p),i=a(await r(e,t),0,p);this.modelBuffers.finalNorm=i.buf,c+=t}for(let e=0;e<this.config.num_layers;e++){let i=`blk.${e}.`,o=t.filter(e=>e.name.startsWith(i));if(o.length===0)continue;let f=1/0,p=0;for(let e of o){let t=Number(e.offset),n=t+s(e);t<f&&(f=t),n>p&&(p=n)}let m=n+f,h=p-f,g=await r(m,h),_={};for(let e of d){let t=o.find(t=>t.name===i+e+`.weight`);t&&(_[e]=a(g,Number(t.offset)-f,t).buf)}for(let e of u){let t=o.find(t=>t.name===i+e+`.weight`);t&&(_[e]=a(g,Number(t.offset)-f,t).buf)}this.modelBuffers.layers.push(_),await this.device.queue.onSubmittedWorkDone(),await new Promise(e=>setTimeout(e,0)),c+=h,this.reportProgress(c,l,`Layer ${e+1}/${this.config.num_layers}`)}}createWorkBuffers(){let e=this.config,t=e.hidden_size,n=e.q_dim,r=e.kv_dim,i=e.intermediate_size,a=e.context_length,o=e.vocab_size,s=e.head_dim,c=e.num_kv_heads,l=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,u=l|GPUBufferUsage.COPY_DST;this.workBuffers={hidden:this.device.createBuffer({size:t*4,usage:u}),hiddenReadback:this.device.createBuffer({size:t*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),residual:this.device.createBuffer({size:t*4,usage:l}),normed:this.device.createBuffer({size:t*4,usage:l}),q:this.device.createBuffer({size:n*4,usage:l}),k:this.device.createBuffer({size:r*4,usage:l}),v:this.device.createBuffer({size:r*4,usage:l}),attnOut:this.device.createBuffer({size:n*4,usage:l}),attnProj:this.device.createBuffer({size:t*4,usage:l}),postAttnNormed:this.device.createBuffer({size:t*4,usage:l}),attnScores:this.device.createBuffer({size:e.num_q_heads*a*4,usage:l}),ffnGate:this.device.createBuffer({size:i*4,usage:l}),ffnUp:this.device.createBuffer({size:i*4,usage:l}),ffnMul:this.device.createBuffer({size:i*4,usage:l}),ffnDown:this.device.createBuffer({size:t*4,usage:l}),postFfnNormed:this.device.createBuffer({size:t*4,usage:l}),logits:this.device.createBuffer({size:o*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),logitsReadback:this.device.createBuffer({size:o*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),argmaxResult:this.device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),argmaxReadback:this.device.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),topk256Result:this.device.createBuffer({size:2048,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),topk256Readback:this.device.createBuffer({size:2048,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})};let d=a*c*s*4;this.kvCaches=[];for(let t=0;t<e.num_layers;t++)this.kvCaches.push({k:this.device.createBuffer({size:d,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),v:this.device.createBuffer({size:d,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC})})}createBindGroups(){let e={embeddingLookup:this.device.createBindGroup({layout:this.pipelines.embeddingLookup.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.modelBuffers.embeddingQ8}},{binding:1,resource:{buffer:this.workBuffers.hidden}},{binding:2,resource:{buffer:this.uniformBuffers.embeddingLookup}}]}),finalNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.hidden}},{binding:1,resource:{buffer:this.modelBuffers.finalNorm}},{binding:2,resource:{buffer:this.workBuffers.normed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),lmHead:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:this.modelBuffers.embeddingQ8}},{binding:2,resource:{buffer:this.workBuffers.logits}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_V_H}}]}),argmax:this.device.createBindGroup({layout:this.pipelines.argmax.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.logits}},{binding:1,resource:{buffer:this.workBuffers.argmaxResult}},{binding:2,resource:{buffer:this.uniformBuffers.argmaxSize}}]}),topk256:this.device.createBindGroup({layout:this.pipelines.topk256.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.logits}},{binding:1,resource:{buffer:this.workBuffers.topk256Result}},{binding:2,resource:{buffer:this.uniformBuffers.argmaxSize}}]}),layers:[]};for(let t=0;t<this.config.num_layers;t++){let n=this.modelBuffers.layers[t],r=this.kvCaches[t],i={attnNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.hidden}},{binding:1,resource:{buffer:n.attn_norm}},{binding:2,resource:{buffer:this.workBuffers.normed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),linearQ:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:n.attn_q}},{binding:2,resource:{buffer:this.workBuffers.q}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_Q_H}}]}),linearK:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:n.attn_k}},{binding:2,resource:{buffer:this.workBuffers.k}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_KV_H}}]}),linearV:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:n.attn_v}},{binding:2,resource:{buffer:this.workBuffers.v}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_KV_H}}]}),ropeQ:this.device.createBindGroup({layout:this.pipelines.rope.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.q}},{binding:1,resource:{buffer:this.uniformBuffers.ropeQ[t]}}]}),ropeK:this.device.createBindGroup({layout:this.pipelines.rope.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.k}},{binding:1,resource:{buffer:this.uniformBuffers.ropeK[t]}}]}),qNorm:this.device.createBindGroup({layout:this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.q}},{binding:1,resource:{buffer:n.attn_q_norm}},{binding:2,resource:{buffer:this.uniformBuffers.perHeadRmsNormQ}}]}),kNorm:this.device.createBindGroup({layout:this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.k}},{binding:1,resource:{buffer:n.attn_k_norm}},{binding:2,resource:{buffer:this.uniformBuffers.perHeadRmsNormK}}]}),fusedNormRopeQ:this.device.createBindGroup({layout:this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.q}},{binding:1,resource:{buffer:n.attn_q_norm}},{binding:2,resource:{buffer:this.uniformBuffers.fusedNormRopeQ[t]}}]}),fusedNormRopeK:this.device.createBindGroup({layout:this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.k}},{binding:1,resource:{buffer:n.attn_k_norm}},{binding:2,resource:{buffer:this.uniformBuffers.fusedNormRopeK[t]}}]}),kvStore:this.device.createBindGroup({layout:this.pipelines.kvCacheStore.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.k}},{binding:1,resource:{buffer:this.workBuffers.v}},{binding:2,resource:{buffer:r.k}},{binding:3,resource:{buffer:r.v}},{binding:4,resource:{buffer:this.uniformBuffers.kvCacheStore}}]}),attnScore:this.device.createBindGroup({layout:this.pipelines.attnScore.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.q}},{binding:1,resource:{buffer:r.k}},{binding:2,resource:{buffer:this.workBuffers.attnScores}},{binding:3,resource:{buffer:this.uniformBuffers.attnScore}}]}),softmax:this.device.createBindGroup({layout:this.pipelines.softmax.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnScores}},{binding:1,resource:{buffer:this.uniformBuffers.softmax}}]}),attnOutput:this.device.createBindGroup({layout:this.pipelines.attnOutput.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnScores}},{binding:1,resource:{buffer:r.v}},{binding:2,resource:{buffer:this.workBuffers.attnOut}},{binding:3,resource:{buffer:this.uniformBuffers.attnOutput}}]}),linearAttnOut:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnOut}},{binding:1,resource:{buffer:n.attn_output}},{binding:2,resource:{buffer:this.workBuffers.attnProj}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_H_Q}}]}),postAttnNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnProj}},{binding:1,resource:{buffer:n.post_attention_norm}},{binding:2,resource:{buffer:this.workBuffers.postAttnNormed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),residualAdd1:this.device.createBindGroup({layout:this.pipelines.add.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.hidden}},{binding:1,resource:{buffer:this.workBuffers.postAttnNormed}},{binding:2,resource:{buffer:this.workBuffers.residual}},{binding:3,resource:{buffer:this.uniformBuffers.sizeH}}]}),ffnNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.residual}},{binding:1,resource:{buffer:n.ffn_norm}},{binding:2,resource:{buffer:this.workBuffers.normed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),ffnGate:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:n.ffn_gate}},{binding:2,resource:{buffer:this.workBuffers.ffnGate}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_I_H}}]}),ffnUp:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:n.ffn_up}},{binding:2,resource:{buffer:this.workBuffers.ffnUp}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_I_H}}]}),geluMul:this.device.createBindGroup({layout:this.pipelines.geluMul.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.ffnGate}},{binding:1,resource:{buffer:this.workBuffers.ffnUp}},{binding:2,resource:{buffer:this.workBuffers.ffnMul}},{binding:3,resource:{buffer:this.uniformBuffers.sizeI}}]}),ffnDown:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.ffnMul}},{binding:1,resource:{buffer:n.ffn_down}},{binding:2,resource:{buffer:this.workBuffers.ffnDown}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_H_I}}]}),postFfnNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.ffnDown}},{binding:1,resource:{buffer:n.post_ffw_norm}},{binding:2,resource:{buffer:this.workBuffers.postFfnNormed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),residualAdd2:this.device.createBindGroup({layout:this.pipelines.add.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.residual}},{binding:1,resource:{buffer:this.workBuffers.postFfnNormed}},{binding:2,resource:{buffer:this.workBuffers.hidden}},{binding:3,resource:{buffer:this.uniformBuffers.sizeH}}]}),fusedPostAttnNormAdd:this.device.createBindGroup({layout:this.pipelines.fusedNormAdd.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnProj}},{binding:1,resource:{buffer:n.post_attention_norm}},{binding:2,resource:{buffer:this.workBuffers.hidden}},{binding:3,resource:{buffer:this.workBuffers.residual}},{binding:4,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),fusedPostFfnNormAdd:this.device.createBindGroup({layout:this.pipelines.fusedNormAdd.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.ffnDown}},{binding:1,resource:{buffer:n.post_ffw_norm}},{binding:2,resource:{buffer:this.workBuffers.residual}},{binding:3,resource:{buffer:this.workBuffers.hidden}},{binding:4,resource:{buffer:this.uniformBuffers.rmsNorm}}]})};e.layers.push(i)}this.bindGroupCache=e}encodeTransformerPass(e,t,n){let r=this.config,i=r.hidden_size,a=r.q_dim,o=r.kv_dim,s=r.intermediate_size,c=r.head_dim,l=r.num_q_heads,u=r.num_kv_heads,d=n+1;this.wb(this.uniformBuffers.embeddingLookup,4,new Uint32Array([t]));let f=new Uint32Array([n]);for(let e=0;e<r.num_layers;e++)this.wb(this.uniformBuffers.fusedNormRopeQ[e],12,f),this.wb(this.uniformBuffers.fusedNormRopeK[e],12,f);this.wb(this.uniformBuffers.kvCacheStore,8,f);let p=new Uint32Array([d]);this.wb(this.uniformBuffers.attnScore,12,p),this.wb(this.uniformBuffers.softmax,4,p),this.wb(this.uniformBuffers.attnOutput,12,p);let m;m=e.beginComputePass(),m.setPipeline(this.pipelines.embeddingLookup),m.setBindGroup(0,this.bindGroupCache.embeddingLookup),m.dispatchWorkgroups(Math.ceil(i/256)),m.end();for(let t=0;t<r.num_layers;t++){let n=this.bindGroupCache.layers[t];m=e.beginComputePass(),m.setPipeline(this.pipelines.rmsNorm),m.setBindGroup(0,n.attnNorm),m.dispatchWorkgroups(1),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.linearQ8),m.setBindGroup(0,n.linearQ),m.dispatchWorkgroups(Math.ceil(a/256)),m.setBindGroup(0,n.linearK),m.dispatchWorkgroups(Math.ceil(o/256)),m.setBindGroup(0,n.linearV),m.dispatchWorkgroups(Math.ceil(o/256)),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.fusedPerHeadNormRope),m.setBindGroup(0,n.fusedNormRopeQ),m.dispatchWorkgroups(l),m.setBindGroup(0,n.fusedNormRopeK),m.dispatchWorkgroups(u),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.kvCacheStore),m.setBindGroup(0,n.kvStore),m.dispatchWorkgroups(Math.ceil(o/256)),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.attnScore),m.setBindGroup(0,n.attnScore),m.dispatchWorkgroups(Math.ceil(l*d/256)),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.softmax),m.setBindGroup(0,n.softmax),m.dispatchWorkgroups(l),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.attnOutput),m.setBindGroup(0,n.attnOutput),m.dispatchWorkgroups(Math.ceil(l*c/256)),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.linearQ8),m.setBindGroup(0,n.linearAttnOut),m.dispatchWorkgroups(Math.ceil(i/256)),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.fusedNormAdd),m.setBindGroup(0,n.fusedPostAttnNormAdd),m.dispatchWorkgroups(1),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.rmsNorm),m.setBindGroup(0,n.ffnNorm),m.dispatchWorkgroups(1),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.linearQ8),m.setBindGroup(0,n.ffnGate),m.dispatchWorkgroups(Math.ceil(s/256)),m.setBindGroup(0,n.ffnUp),m.dispatchWorkgroups(Math.ceil(s/256)),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.geluMul),m.setBindGroup(0,n.geluMul),m.dispatchWorkgroups(Math.ceil(s/256)),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.linearQ8),m.setBindGroup(0,n.ffnDown),m.dispatchWorkgroups(Math.ceil(i/256)),m.end(),m=e.beginComputePass(),m.setPipeline(this.pipelines.fusedNormAdd),m.setBindGroup(0,n.fusedPostFfnNormAdd),m.dispatchWorkgroups(1),m.end()}m=e.beginComputePass(),m.setPipeline(this.pipelines.rmsNorm),m.setBindGroup(0,this.bindGroupCache.finalNorm),m.dispatchWorkgroups(1),m.end()}async sampleNextToken(e,t,n,r,i){if(this.deviceLost)throw Error(`WebGPU device lost`);let a=this.config.vocab_size,o;o=e.beginComputePass(),o.setPipeline(this.pipelines.linearQ8),o.setBindGroup(0,this.bindGroupCache.lmHead),o.dispatchWorkgroups(Math.ceil(a/256)),o.end();let s=t===0&&r<=1;if(s?(o=e.beginComputePass(),o.setPipeline(this.pipelines.argmax),o.setBindGroup(0,this.bindGroupCache.argmax),o.dispatchWorkgroups(1),o.end(),e.copyBufferToBuffer(this.workBuffers.argmaxResult,0,this.workBuffers.argmaxReadback,0,4)):(o=e.beginComputePass(),o.setPipeline(this.pipelines.topk256),o.setBindGroup(0,this.bindGroupCache.topk256),o.dispatchWorkgroups(1),o.end(),e.copyBufferToBuffer(this.workBuffers.topk256Result,0,this.workBuffers.topk256Readback,0,2048)),this.device.queue.submit([e.finish()]),s){try{await this.workBuffers.argmaxReadback.mapAsync(GPUMapMode.READ)}catch(e){throw Error(`GPU readback failed (device lost?): ${e}`)}let e=new Uint32Array(this.workBuffers.argmaxReadback.getMappedRange())[0];return this.workBuffers.argmaxReadback.unmap(),e}try{await this.workBuffers.topk256Readback.mapAsync(GPUMapMode.READ)}catch(e){throw Error(`GPU readback failed (device lost?): ${e}`)}let c=new Float32Array(this.workBuffers.topk256Readback.getMappedRange().slice(0));this.workBuffers.topk256Readback.unmap();let l=Array(256),u=new Uint32Array(c.buffer.slice(0));for(let e=0;e<256;e++)l[e]={val:c[e*2],id:u[e*2+1]};if(r>1&&i.length>0){let e=new Set(i);for(let t=0;t<256;t++)e.has(l[t].id)&&(l[t].val>0?l[t].val/=r:l[t].val*=r)}if(l.sort((e,t)=>t.val-e.val),t===0)return l[0].id;let d=l[0].val,f=0,p=/* @__PURE__ */ new Float32Array(256);for(let e=0;e<256;e++)p[e]=Math.exp((l[e].val-d)/t),f+=p[e];let m=0,h=256;for(let e=0;e<256;e++)if(m+=p[e]/f,m>=n){h=e+1;break}let g=0;for(let e=0;e<h;e++)g+=p[e];let _=Math.random()*g;for(let e=0;e<h;e++)if(_-=p[e],_<=0)return l[e].id;return l[h-1].id}async forwardPassAndGetToken(e,t,n=0,r=.9,i=1,a=[]){let o=this.device.createCommandEncoder();return this.encodeTransformerPass(o,e,t),this.sampleNextToken(o,n,r,i,a)}forwardPassOnly(e,t){let n=this.device.createCommandEncoder();this.encodeTransformerPass(n,e,t),this.device.queue.submit([n.finish()])}async prefillBatched(e,t=0){for(let n=0;n<e.length;n++)this.forwardPassOnly(e[n],t+n);await this.device.queue.onSubmittedWorkDone()}resetKVCaches(){let e=this.config.head_dim,t=this.config.num_kv_heads,n=this.config.context_length,r=new Float32Array(n*t*e);for(let e=0;e<this.config.num_layers;e++)this.wb(this.kvCaches[e].k,0,r),this.wb(this.kvCaches[e].v,0,r)}async getFirstTokenAfterPrefill(e,t,n,r){let i=this.device.createCommandEncoder();return this.sampleNextToken(i,e,t,n,r)}addUserMessage(e){this.conversationHistory.push({role:`user`,text:e})}async*generate(e={}){if(this.deviceLost)throw Error(`WebGPU device lost — call dispose() and recreate the engine`);let t=e.temperature??.7,n=e.topP??.9,r=e.repPenalty??1.2,i=e.maxTokens??32768,a=e.toolsJson??`[]`,o=e.signal,s;if(this.kvPosition===0){let e=f(this.conversationHistory,a);s=this.tokenizer.encode(e)}else{let e=`<end_of_turn>
<start_of_turn>user
${this.conversationHistory[this.conversationHistory.length-1].text}<end_of_turn>
<start_of_turn>model
`;s=this.tokenizer.encode(e).slice(1)}if(this.kvPosition+s.length>=this.config.context_length-10){let e=this.conversationHistory[this.conversationHistory.length-1];this.conversationHistory=[{role:`user`,text:e.text}];let t=f(this.conversationHistory,a);s=this.tokenizer.encode(t),this.resetKVCaches(),this.kvPosition=0}await this.prefillBatched(s,this.kvPosition);let c=[...s];this.kvPosition+=s.length;let l=await this.getFirstTokenAfterPrefill(t,n,r,c);c.push(l);let u=this.tokenizer.funcTokens[`<end_function_call>`],d=[l],p=0,m=``;l!==1&&l!==0&&l!==106&&(m=this.tokenizer.decodeTokens(d),yield m);for(let e=1;e<i&&!(l===1||l===0||l===106||u&&l===u||o?.aborted);e++){let e=this.kvPosition+p;if(e>=this.config.context_length-1||(l=await this.forwardPassAndGetToken(l,e,t,n,r,c),p++,l===1||l===0||l===106))break;if(u&&l===u){c.push(l),d.push(l);break}c.push(l),d.push(l),yield this.tokenizer.decodeTokens(d).slice(m.length),m=this.tokenizer.decodeTokens(d)}let h=this.tokenizer.decodeTokens(d);this.conversationHistory.push({role:`model`,text:h}),this.kvPosition+=p}resetConversation(){this.conversationHistory=[],this.kvPosition=0,this.resetKVCaches()}dispose(){let e=e=>{e&&e.destroy()};if(e(this.modelBuffers?.embeddingQ8),e(this.modelBuffers?.finalNorm),this.modelBuffers?.layers)for(let t of this.modelBuffers.layers)for(let n of Object.values(t))e(n);if(this.workBuffers)for(let t of Object.values(this.workBuffers))e(t);if(this.kvCaches)for(let t of this.kvCaches)e(t.k),e(t.v);if(this.uniformBuffers)for(let t of Object.values(this.uniformBuffers))if(Array.isArray(t))for(let n of t)e(n);else e(t);this.device?.destroy()}};async function v(e={}){let t=new _(e);return await t.init(e),t}function y(e){let t=new TextEncoder;if(t.encode(e).length<=768)return e;let n=Array.from(e),r=(e,n)=>{let r=[],i=0;for(let a of e){if(i+=t.encode(a).length,i>n)break;r.push(a)}return r};return`${r(n,380).join(``)}\n${r(n.reverse(),380).reverse().join(``)}`}function b(e,t){let n=e.match(/"([^"\n]{1,12})"/)?.[1]??`몽실`;return y(`목장 양의 대사를 완성해. 너는 ${n}, 상대는 양치기야. 성격: ${e.match(/성격: (.+?)(?: 양치기와의|$)/)?.[1]?.slice(0,35)??`다정하고 장난기 있어`}. 한국어 반말 한 문장만 써. 이름표, 설명, 괄호는 빼.\n양치기: 안녕\n${n}: 왔구나! 같이 놀자.\n양치기: 내일 또 올게\n${n}: 좋아, 내일도 기다릴게.\n양치기: ${t.replace(/^양치기:\s*/,``).slice(0,60)}\n${n}:`)}let x=`https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/f0b45be0aac41bd6a100a4b5734cad5f67255bfb/gemma-3-1b-it-Q8_0.gguf`,S=1069306400;function C(e,t=S){let n=e?.match(/^bytes=(\d+)-(\d+)$/);if(!n)return null;let r=Number(n[1]),i=Number(n[2]);return Number.isSafeInteger(r)&&Number.isSafeInteger(i)&&r<=i&&i<t?{start:r,end:i}:null}function w(e,t=()=>{},n={url:x,bytes:S,cache:`gemma3-q8-f0b45be0`}){return async(r,i)=>{if((r instanceof Request?r.url:String(r))!==n.url)return e(r,i);let a=new Headers(i?.headers??(r instanceof Request?r.headers:void 0)),o=C(a.get(`Range`),n.bytes);if(!o)throw Error(`모델을 작은 조각으로 요청하지 못했어요`);let s;try{s=await(await navigator.storage.getDirectory()).getDirectoryHandle(n.cache,{create:!0})}catch{}let c=async(o,c)=>{let l=c-o+1,u=`${o}-${c}`;if(s)try{await s.getFileHandle(`${u}.ok`);let e=await s.getFileHandle(u),n=e;if(n.createSyncAccessHandle){let e=await n.createSyncAccessHandle();try{if(e.getSize()===l){let n=new Uint8Array(l),r=0;for(;r<l;){let t=e.read(n.subarray(r),{at:r});if(!t)throw Error(`모델 조각 읽기가 중단됐어요`);r+=t}return t(l),n}}finally{e.close()}}else{let n=await e.getFile();if(n.size===l){let e=new Uint8Array(await n.arrayBuffer());return t(l),e}}}catch{}a.set(`Range`,`bytes=${o}-${c}`);let d=await e(r,{...i,headers:a,signal:i?.signal??AbortSignal.timeout(18e4)});if(d.status!==206||d.headers.get(`content-range`)!==`bytes ${o}-${c}/${n.bytes}`||!d.body)throw await d.body?.cancel(),Error(`모델 서버가 조각 받기에 응답하지 않았어요. 다시 시도해 주세요`);let f=new Uint8Array(l),p=d.body.getReader(),m=0;try{for(;;){let{done:e,value:t}=await p.read();if(e)break;if(m+t.length>l)throw Error(`모델 조각 크기가 맞지 않아요`);f.set(t,m),m+=t.length}if(m!==l)throw Error(`모델 받기가 중간에 끊겼어요. 다시 시도해 주세요`)}catch(e){throw await p.cancel().catch(()=>{}),e}finally{p.releaseLock()}if(s){let e=await(await s.getFileHandle(u,{create:!0})).createWritable();try{await e.write(f),await e.close()}catch(t){throw await e.abort().catch(()=>{}),t}await(await(await s.getFileHandle(`${u}.ok`,{create:!0})).createWritable()).close()}return t(l),f},l=o.start,u=new Response(new ReadableStream({async pull(e){if(l>o.end){e.close();return}let t=Math.min(o.end,l+4194304-1);e.enqueue(await c(l,t)),l=t+1}},{highWaterMark:0}),{status:206,headers:{"content-length":String(o.end-o.start+1),"content-range":`bytes ${o.start}-${o.end}/${n.bytes}`}});return u.arrayBuffer=async()=>{let e=new Uint8Array(o.end-o.start+1),t=u.body.getReader(),n=0;try{for(;;){let{done:r,value:i}=await t.read();if(r)break;e.set(i,n),n+=i.length}return e.buffer}catch(e){throw await t.cancel().catch(()=>{}),e}finally{t.releaseLock()}},u}}let T=self,E=0;globalThis.fetch=w(fetch.bind(globalThis),e=>{E+=e,T.postMessage({type:`progress`,bytes:E,total:S})});let D=[`목초 먹고 싶어?`,`오늘 뭐 했어?`,`무슨 일 있어?`,`내일 또 놀러 올게`,`친구와 다퉜어. 위로해 줘.`,`나 이제 가야 해`,`당근 줄까?`,`너 이름이 뭐야?`];T.onmessage=({data:e})=>{(async()=>{let t=performance.now(),n=await v({model:x,contextLength:1024,localRopeTheta:e.theta});T.postMessage({type:`ready`,ms:Math.round(performance.now()-t)});try{for(let t of D){n.resetConversation(),n.addUserMessage(e.character?b(`당신은 목장 양 "몽실"입니다. 성격: 다정하고 장난기 있어 양치기와의 하트 3/10`,`양치기: ${t}`):`너는 목장에서 사는 양 몽실이야. 양치기와 다정하게 대화해. 질문에 맞춰 한국어 반말 한 문장으로 짧게 답해. 대사만 써.\n양치기: ${t}`);let r=performance.now(),i=``;for await(let e of n.generate({maxTokens:64,temperature:0}))i+=e;T.postMessage({type:`answer`,question:t,text:i,ms:Math.round(performance.now()-r)})}T.postMessage({type:`complete`})}finally{n.dispose()}})().catch(e=>T.postMessage({type:`error`,error:e instanceof Error?e.message:String(e)}))}})();