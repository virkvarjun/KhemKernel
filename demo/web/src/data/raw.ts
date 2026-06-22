// Real source files imported verbatim as strings (Vite ?raw). These are the
// single source of truth for every code snippet shown in the guide.
// Paths are relative to demo/web/src/data/ up to the repo root.

import ops from "../../../../picochem/ops.py?raw";
import embeddings from "../../../../picochem/embeddings.py?raw";
import ffn from "../../../../picochem/ffn.py?raw";
import attention from "../../../../picochem/attention.py?raw";
import encoder from "../../../../picochem/encoder.py?raw";
import decoder from "../../../../picochem/decoder.py?raw";
import model from "../../../../picochem/model.py?raw";
import optimizer from "../../../../picochem/optimizer.py?raw";
import scheduler from "../../../../picochem/scheduler.py?raw";
import bpe from "../../../../picochem/bpe.py?raw";
import data from "../../../../picochem/data.py?raw";
import dataLoader from "../../../../picochem/data_loader.py?raw";
import traces from "../../../../picochem/traces.py?raw";
import checkpointing from "../../../../picochem/checkpointing.py?raw";
import deviceLayers from "../../../../picochem/device_layers.py?raw";
import evaluate from "../../../../picochem/evaluate.py?raw";

import matmulTiled from "../../../../picochem/kernels/cuda/matmul_tiled.cu?raw";
import matmulBackward from "../../../../picochem/kernels/cuda/matmul_backward.cu?raw";
import batchedMatmul from "../../../../picochem/kernels/cuda/batched_matmul.cu?raw";
import softmax from "../../../../picochem/kernels/cuda/softmax.cu?raw";
import softmaxBackward from "../../../../picochem/kernels/cuda/softmax_backward.cu?raw";
import layerNorm from "../../../../picochem/kernels/cuda/layer_norm.cu?raw";
import layerNormBackward from "../../../../picochem/kernels/cuda/layer_norm_backward.cu?raw";
import gelu from "../../../../picochem/kernels/cuda/gelu.cu?raw";
import crossEntropy from "../../../../picochem/kernels/cuda/cross_entropy.cu?raw";
import embeddingCu from "../../../../picochem/kernels/cuda/embedding.cu?raw";
import adam from "../../../../picochem/kernels/cuda/adam.cu?raw";
import bias from "../../../../picochem/kernels/cuda/bias.cu?raw";
import transpose from "../../../../picochem/kernels/cuda/transpose.cu?raw";
import vectorAdd from "../../../../picochem/kernels/cuda/vector_add.cu?raw";
import bindings from "../../../../picochem/kernels/cuda/bindings.cpp?raw";

import trainDevice from "../../../../scripts/train_device.py?raw";
import trainCpu from "../../../../scripts/train.py?raw";
import buildCuda from "../../../../scripts/build_cuda.sh?raw";

export const RAW = {
  ops,
  embeddings,
  ffn,
  attention,
  encoder,
  decoder,
  model,
  optimizer,
  scheduler,
  bpe,
  data,
  dataLoader,
  traces,
  checkpointing,
  deviceLayers,
  evaluate,
  matmulTiled,
  matmulBackward,
  batchedMatmul,
  softmax,
  softmaxBackward,
  layerNorm,
  layerNormBackward,
  gelu,
  crossEntropy,
  embeddingCu,
  adam,
  bias,
  transpose,
  vectorAdd,
  bindings,
  trainDevice,
  trainCpu,
  buildCuda,
} as const;

export type RawKey = keyof typeof RAW;
