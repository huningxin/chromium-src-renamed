// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/ml/model_impl_ngraph.h"

#include <string>
#include <utility>

#include "base/strings/string_number_conversions.h"
#include "services/ml/compilation_impl_ngraph.h"
#include "services/ml/public/interfaces/constants.mojom.h"
#include "third_party/ngraph/include/ngraph/ngraph.hpp"

namespace ml {

ModelImplNgraph::ModelImplNgraph() {
}

ModelImplNgraph::~ModelImplNgraph() {
}

bool ModelImplNgraph::Init() {
  return true;
}

void ModelImplNgraph::Finish(mojom::ModelInfoPtr model_info,
                            FinishCallback callback) {
  DLOG(INFO) << "ModelImplNgraph::Finish";
  int32_t result = mojom::NOT_ERROR;
  DLOG(INFO) << "operands(" << model_info->operands.size() << ")";
  for (size_t i = 0; i < model_info->operands.size(); ++i) {
    DLOG(INFO) << "  operand[" << i << "]";
    const mojom::OperandPtr& operand = model_info->operands[i];
    result = AddOperand(operand->type, operand->dimensions, operand->scale,
                        operand->zeroPoint);
    if (result != mojom::NOT_ERROR) {
      std::move(callback).Run(result);
      return;
    }
  }
  DLOG(INFO) << "values(" << model_info->values.size() << ")";
  memory_size_ = model_info->memory_size;
  auto mapping = model_info->memory->Map(memory_size_);
  const int8_t* base = static_cast<const int8_t*>(mapping.get());
  memory_.reset(new int8_t[memory_size_]);
  memcpy(memory_.get(), base, memory_size_);
  for (size_t i = 0; i < model_info->values.size(); ++i) {
    DLOG(INFO) << "  values[" << i << "]";
    const mojom::OperandValueInfoPtr& value_info = model_info->values[i];
    result = SetOperandValue(
        value_info->index,
        static_cast<const void*>(memory_.get() + value_info->offset),
        value_info->length);
    if (result != mojom::NOT_ERROR) {
      std::move(callback).Run(result);
      return;
    }
    ValueInfo value;
    value.index = value_info->index;
    value.offset = value_info->offset;
    value.length = value_info->length;
    values_[value_info->index] = value;
  }
  DLOG(INFO) << "inputs(" << model_info->inputs.size() << ")";
  DLOG(INFO) << "outputs(" << model_info->outputs.size() << ")";
  result = IdentifyInputsAndOutputs(model_info->inputs, model_info->outputs);
  if (result != mojom::NOT_ERROR) {
    std::move(callback).Run(result);
    return;
  }
  DLOG(INFO) << "operations(" << model_info->operations.size() << ")";
  for (size_t i = 0; i < model_info->operations.size(); ++i) {
    DLOG(INFO) << "  operation[" << i << "]";
    const mojom::OperationPtr& operation = model_info->operations[i];
    result =
        AddOperation(operation->type, operation->inputs, operation->outputs);
    if (result != mojom::NOT_ERROR) {
      std::move(callback).Run(result);
      return;
    }
  }

  std::move(callback).Run(mojom::NOT_ERROR);
}

void ModelImplNgraph::CreateCompilation(CreateCompilationCallback callback) {
  DLOG(INFO) << "ModelImplNgraph::CreateCompilation";
  auto init_params = mojom::CompilationInitParams::New();
  mojom::CompilationPtrInfo ptr_info;
  mojo::MakeStrongBinding(std::make_unique<CompilationImplNgraph>(this),
                          mojo::MakeRequest(&ptr_info));
  init_params->compilation = std::move(ptr_info);

  std::move(callback).Run(mojom::NOT_ERROR, std::move(init_params));
}

int32_t ModelImplNgraph::AddOperand(int32_t type,
                                   const std::vector<uint32_t>& dimensions,
                                   float scale,
                                   int32_t zeroPoint) {
  DLOG(INFO) << "  ModelImplNgraph::AddOperand";
  DLOG(INFO) << "    "
             << "type: " << type;
  DLOG(INFO) << "    "
             << "dimensions(" << dimensions.size()
             << "): " << VectorToString(dimensions.data(), dimensions.size());
  DLOG(INFO) << "    "
             << "scale: " << scale;
  DLOG(INFO) << "    "
             << "zeroPoint: " << zeroPoint;
  Operand operand;
  operand.type = type;
  operand.dimensions = dimensions;
  operand.scale = scale;
  operand.zeroPoint = zeroPoint;
  operands_.push_back(operand);

  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::SetOperandValue(uint32_t index,
                                        const void* buffer,
                                        uint32_t length) {
  DLOG(INFO) << "  ModelImplNgraph::SetOperandValue";
  DLOG(INFO) << "    "
             << "index: " << index;
  DLOG(INFO) << "    "
             << "length: " << length;
  if (index > operands_.size()) {
    return mojom::BAD_DATA;
  }
  auto operand = operands_[index];
  if (operand.type == mojom::TENSOR_FLOAT32 || operand.type == mojom::FLOAT32) {
    const float* value = static_cast<const float*>(buffer);
    uint32_t size = length / 4;
    DLOG(INFO) << "    "
               << "buffer(" << size << "): " << VectorToString(value, size);
  } else if (operand.type == mojom::TENSOR_INT32 ||
             operand.type == mojom::INT32) {
    const int32_t* value = static_cast<const int32_t*>(buffer);
    uint32_t size = length / 4;
    DLOG(INFO) << "    "
               << "buffer(" << size << "): " << VectorToString(value, size);
  } else if (operand.type == mojom::TENSOR_QUANT8_ASYMM) {
    const int8_t* value = static_cast<const int8_t*>(buffer);
    uint32_t size = length;
    DLOG(INFO) << "    "
               << "buffer(" << size << "): " << VectorToString(value, size);
  } else if (operand.type == mojom::UINT32) {
    const uint32_t* value = static_cast<const uint32_t*>(buffer);
    uint32_t size = length;
    DLOG(INFO) << "    "
               << "buffer(" << size << "): " << VectorToString(value, size);
  }

  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::AddOperation(int32_t type,
                                     const std::vector<uint32_t>& inputs,
                                     const std::vector<uint32_t>& outputs) {
  DLOG(INFO) << "  ModelImplNgraph::AddOperation";
  DLOG(INFO) << "    "
             << "type: " << type;
  DLOG(INFO) << "    "
             << "inputs(" << inputs.size()
             << "): " << VectorToString(inputs.data(), inputs.size());
  DLOG(INFO) << "    "
             << "outputs(" << outputs.size()
             << "): " << VectorToString(outputs.data(), outputs.size());
  Operation operation;
  operation.type = type;
  operation.inputs = inputs;
  operation.outputs = outputs;
  operations_.push_back(operation);

  if (operation.outputs.size() != 1) {
    DLOG(ERROR) << "Only 1 output is supported";
    return mojom::BAD_DATA;
  }

  int32_t result = mojom::NOT_ERROR;
  if (type == mojom::ADD || type == mojom::MUL) {
    result = NgraphAddElementwise(type, inputs, outputs);
  } else if (type == mojom::CONV_2D || type == mojom::DEPTHWISE_CONV_2D ||
             type == mojom::ATROUS_CONV_2D ||
             type == mojom::ATROUS_DEPTHWISE_CONV_2D) {
    result = NgraphAddConvolution(type, inputs, outputs);
  } else if (type == mojom::AVERAGE_POOL_2D || type == mojom::MAX_POOL_2D) {
    result = NgraphAddPooling(type, inputs, outputs);
  } else if (type == mojom::SOFTMAX) {
    result = NgraphAddSoftmax(type, inputs, outputs);
  } else if (type == mojom::RESHAPE) {
    result = NgraphAddReshape(type, inputs, outputs);
  } else if (type == mojom::CONCATENATION) {
    result = NgraphAddConcatenation(type, inputs, outputs);
  } else if (type == mojom::FULLY_CONNECTED) {
    result = NgraphAddFullyConnected(type, inputs, outputs);
  } else if (type == mojom::RESIZE_BILINEAR) {
    result = NgraphAddResizeBilinear(type, inputs, outputs);
  } else {
    DLOG(ERROR) << "Operation type " << type << " is not supported.";
    return mojom::BAD_DATA;
  }
  return result;
}

int32_t ModelImplNgraph::IdentifyInputsAndOutputs(
    const std::vector<uint32_t>& inputs,
    const std::vector<uint32_t>& outputs) {
  DLOG(INFO) << "  ModelImplNgraph::IdentifyInputsAndOutputs";
  DLOG(INFO) << "    "
             << "inputs(" << inputs.size()
             << "): " << VectorToString(inputs.data(), inputs.size());
  DLOG(INFO) << "    "
             << "outputs(" << outputs.size()
             << "): " << VectorToString(outputs.data(), outputs.size());
  inputs_ = inputs;
  outputs_ = outputs;

  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::NgraphAddElementwise(
    int32_t type,
    const std::vector<uint32_t>& inputs,
    const std::vector<uint32_t>& outputs) {
  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::NgraphAddConvolution(
    int32_t type,
    const std::vector<uint32_t>& inputs,
    const std::vector<uint32_t>& outputs) {
  const bool depthwise = (type == mojom::DEPTHWISE_CONV_2D ||
                          type == mojom::ATROUS_DEPTHWISE_CONV_2D)
                             ? true
                             : false;
  const bool atrous =
      (type == mojom::ATROUS_CONV_2D || type == mojom::ATROUS_DEPTHWISE_CONV_2D)
          ? true
          : false;
  const uint32_t output_index = outputs[0];
  const Operand& output = operands_[output_index];
  const int32_t output_batch = output.dimensions[0];
  const int32_t output_height = output.dimensions[1];
  const int32_t output_width = output.dimensions[2];
  const int32_t output_channel = output.dimensions[3];
  uint32_t index = 0;
  const uint32_t input_index = inputs[index++];
  const Operand& input = operands_[input_index];
  const int32_t input_height = input.dimensions[1];
  const int32_t input_width = input.dimensions[2];

  const uint32_t filter_idx = inputs[index++];
  Operand& filter = operands_[filter_idx];
  int32_t depth_out, depth_in;
  if (depthwise) {
    depth_out = filter.dimensions[3];
  } else {
    depth_out = filter.dimensions[0];
    depth_in = filter.dimensions[3];
  }
  const int32_t filter_height = filter.dimensions[1];
  const int32_t filter_width = filter.dimensions[2];

  // const uint32_t bias_idx = inputs[index++];

  bool implicit_padding;
  int32_t padding_left, padding_right, padding_top, padding_bottom,
      padding_code;
  if ((!depthwise && inputs.size() == 10) ||
      (depthwise && inputs.size() == 11)) {
    implicit_padding = false;
    padding_left = getScalarInt32(values_[inputs[index++]], memory_.get());
    padding_right = getScalarInt32(values_[inputs[index++]], memory_.get());
    padding_top = getScalarInt32(values_[inputs[index++]], memory_.get());
    padding_bottom = getScalarInt32(values_[inputs[index++]], memory_.get());
  } else if ((!depthwise && inputs.size() == 7) ||
             (depthwise && inputs.size() == 8)) {
    implicit_padding = true;
    padding_code = getScalarInt32(values_[inputs[index++]], memory_.get());
  } else {
    DLOG(ERROR) << "Inputs size is incorrect";
    return mojom::BAD_DATA;
  }
  int32_t stride_width, stride_height;
  int32_t dilation_width, dilation_height;
  if (!atrous) {
    stride_width = getScalarInt32(values_[inputs[index++]], memory_.get());
    stride_height = getScalarInt32(values_[inputs[index++]], memory_.get());
    dilation_width = 1;
    dilation_height = 1;
  } else {
    dilation_width = getScalarInt32(values_[inputs[index++]], memory_.get());
    dilation_height = getScalarInt32(values_[inputs[index++]], memory_.get());
    stride_width = 1;
    stride_height = 1;
  }
  int32_t depthwise_multiplier;
  if (depthwise) {
    depthwise_multiplier =
        getScalarInt32(values_[inputs[index++]], memory_.get());
    if (depthwise_multiplier != 1) {
      DLOG(ERROR) << "  depthwise_multiplier " << depthwise_multiplier
                  << " is not supported.";
      return mojom::BAD_DATA;
    }
    depth_in = depth_out / depthwise_multiplier;
  }
  const int32_t fuse_code =
      getScalarInt32(values_[inputs[index++]], memory_.get());

  DLOG(INFO) << "  input_height: " << input_height;
  DLOG(INFO) << "  input_width: " << input_width;
  DLOG(INFO) << "  output_batch: " << output_batch;
  DLOG(INFO) << "  output_height: " << output_height;
  DLOG(INFO) << "  output_width: " << output_width;
  DLOG(INFO) << "  output_channel: " << output_channel;
  DLOG(INFO) << "  filter_height: " << filter_height;
  DLOG(INFO) << "  filter_width: " << filter_width;
  DLOG(INFO) << "  depth_in: " << depth_in;
  DLOG(INFO) << "  depth_out: " << depth_out;
  DLOG(INFO) << "  implicit_padding: " << implicit_padding;
  if (implicit_padding) {
    DLOG(INFO) << "  padding_code: " << padding_code;
  } else {
    DLOG(INFO) << "  padding_left: " << padding_left;
    DLOG(INFO) << "  padding_right: " << padding_right;
    DLOG(INFO) << "  padding_top: " << padding_top;
    DLOG(INFO) << "  padding_bottom: " << padding_bottom;
  }
  DLOG(INFO) << "  stride_width: " << stride_width;
  DLOG(INFO) << "  stride_height: " << stride_height;
  DLOG(INFO) << "  dilation_width: " << dilation_width;
  DLOG(INFO) << "  dilation_height: " << dilation_height;
  if (depthwise) {
    DLOG(INFO) << "  depthwise_multiplier: " << depthwise_multiplier;
  }
  DLOG(INFO) << "  fuse_code: " << fuse_code;

  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::NgraphAddPooling(int32_t type,
                                        const std::vector<uint32_t>& inputs,
                                        const std::vector<uint32_t>& outputs) {
  const uint32_t output_index = outputs[0];
  const Operand& output = operands_[output_index];
  const int32_t output_batch = output.dimensions[0];
  const int32_t output_height = output.dimensions[1];
  const int32_t output_width = output.dimensions[2];
  const int32_t output_channel = output.dimensions[3];
  int32_t i = 0;
  const int32_t input_index = inputs[i++];
  const Operand& input = operands_[input_index];
  const int32_t input_height = input.dimensions[1];
  const int32_t input_width = input.dimensions[2];

  DLOG(INFO) << "  input_height: " << input_height
             << "  input_width: " << input_width;
  DLOG(INFO) << "  output_batch: " << output_batch
             << "  output_height: " << output_height
             << "  output_width: " << output_width
             << "  output_channel: " << output_channel;

  bool implicit_padding;
  int32_t padding_left, padding_right, padding_top, padding_bottom,
      padding_code;
  if (inputs.size() == 10) {
    implicit_padding = false;
    padding_left = getScalarInt32(values_[inputs[i++]], memory_.get());
    padding_right = getScalarInt32(values_[inputs[i++]], memory_.get());
    padding_top = getScalarInt32(values_[inputs[i++]], memory_.get());
    padding_bottom = getScalarInt32(values_[inputs[i++]], memory_.get());
  } else if (inputs.size() == 7) {
    implicit_padding = true;
    padding_code = getScalarInt32(values_[inputs[i++]], memory_.get());
  } else {
    DLOG(ERROR) << "  inputs size is incorrect";
    return mojom::BAD_DATA;
  }
  const int32_t stride_width =
      getScalarInt32(values_[inputs[i++]], memory_.get());
  const int32_t stride_height =
      getScalarInt32(values_[inputs[i++]], memory_.get());
  const int32_t filter_width =
      getScalarInt32(values_[inputs[i++]], memory_.get());
  const int32_t filter_height =
      getScalarInt32(values_[inputs[i++]], memory_.get());
  const int32_t fuse_code = getScalarInt32(values_[inputs[i++]], memory_.get());

  DLOG(INFO) << "  implicit_padding: " << implicit_padding;
  if (implicit_padding) {
    DLOG(INFO) << "  padding_code: " << padding_code;
  } else {
    DLOG(INFO) << "  padding_left: " << padding_left;
    DLOG(INFO) << "  padding_right: " << padding_right;
    DLOG(INFO) << "  padding_top: " << padding_top;
    DLOG(INFO) << "  padding_bottom: " << padding_bottom;
  }
  DLOG(INFO) << "  stride_width: " << stride_width;
  DLOG(INFO) << "  stride_height: " << stride_height;
  DLOG(INFO) << "  filter_height: " << filter_height;
  DLOG(INFO) << "  filter_width: " << filter_width;
  DLOG(INFO) << "  fuse_code: " << fuse_code;

  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::NgraphAddSoftmax(int32_t type,
                                        const std::vector<uint32_t>& inputs,
                                        const std::vector<uint32_t>& outputs) {
  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::NgraphAddReshape(int32_t type,
                                        const std::vector<uint32_t>& inputs,
                                        const std::vector<uint32_t>& outputs) {
  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::NgraphAddConcatenation(
    int32_t type,
    const std::vector<uint32_t>& inputs,
    const std::vector<uint32_t>& outputs) {
  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::NgraphAddFullyConnected(
    int32_t type,
    const std::vector<uint32_t>& inputs,
    const std::vector<uint32_t>& outputs) {
  // The output tensor, of shape [batch_size, num_units]
  const uint32_t output_index = outputs[0];
  const Operand& output = operands_[output_index];
  const int32_t output_batch_size = output.dimensions[0];
  const int32_t output_num_units = output.dimensions[1];

  uint32_t index = 0;
  const uint32_t input_index = inputs[index++];
  const Operand& input = operands_[input_index];
  // A tensor of at least rank 2, specifying the input.
  if (input.dimensions.size() < 2) {
    DLOG(ERROR) << "A tenosr of least rank 2.";
    return mojom::BAD_DATA;
  }

  const uint32_t weights_idx = inputs[index++];
  const Operand& weights = operands_[weights_idx];
  const uint32_t num_units = weights.dimensions[0];
  const uint32_t input_size = weights.dimensions[1];

  // According to Android NN API doc:
  // If rank is greater than 2, then it gets flattened to a 2-D Tensor.
  // The (flattened) 2-D Tensor is reshaped (if necessary) to
  // [batch_size, input_size], where "input_size" corresponds to the number of
  // inputs to the layer, matching the second dimension of weights, and
  // "batch_size" is calculated by dividing the number of elements by
  // "input_size".
  uint32_t input_batch_size;
  if (input.dimensions.size() > 2) {
    input_batch_size = product(input.dimensions) / input_size;
  } else {
    if (input.dimensions[1] != input_size) {
      DLOG(ERROR) << "input.dimensions[1] (" << input.dimensions[1] << ") "
                  << "!= input_size (" << input_size << ")";
      return mojom::BAD_DATA;
    }
    input_batch_size = input.dimensions[0];
  }

  // A 1-D tensor, of shape [num_units]
  const uint32_t bias_idx = inputs[index++];
  const Operand& bias = operands_[bias_idx];
  const uint32_t bias_num_units = bias.dimensions[0];
  if (bias_num_units != num_units) {
    DLOG(ERROR) << "bias_num_units (" << bias_num_units << ") != "
                << "num_units (" << num_units << ")";
    return mojom::BAD_DATA;
  }

  const int32_t fuse_code =
      getScalarInt32(values_[inputs[index++]], memory_.get());

  DLOG(INFO) << "  input_batch_size: " << input_batch_size;
  DLOG(INFO) << "  num_units: " << num_units;
  DLOG(INFO) << "  input_size: " << input_size;
  DLOG(INFO) << "  bias_num_units: " << bias_num_units;
  DLOG(INFO) << "  output_batch_size: " << output_batch_size;
  DLOG(INFO) << "  output_num_units: " << output_num_units;
  DLOG(INFO) << "  fuse_code: " << fuse_code;

  return mojom::NOT_ERROR;
}

int32_t ModelImplNgraph::NgraphAddResizeBilinear(
    int32_t type,
    const std::vector<uint32_t>& inputs,
    const std::vector<uint32_t>& outputs) {
  const Operand& input = operands_[inputs[0]];
  if (input.dimensions.size() != 4) {
    DLOG(ERROR) << "Input must be a 4-D tensor";
    return mojom::BAD_DATA;
  }
  const uint32_t height = input.dimensions[1];
  const uint32_t width = input.dimensions[2];
  const uint32_t channel = input.dimensions[3];
  const uint32_t new_height = getScalarInt32(values_[inputs[1]], memory_.get());
  const uint32_t new_width = getScalarInt32(values_[inputs[2]], memory_.get());
  const float y_scale = new_height / height;
  const float x_scale = new_width / width;
  const uint32_t scale = std::floor(x_scale);

  DLOG(INFO) << "  height: " << height;
  DLOG(INFO) << "  width: " << width;
  DLOG(INFO) << "  channel: " << channel;
  DLOG(INFO) << "  new_height: " << new_height;
  DLOG(INFO) << "  new_width: " << new_width;
  DLOG(INFO) << "  y_scale: " << y_scale;
  DLOG(INFO) << "  x_scale: " << x_scale;
  DLOG(INFO) << "  scale: " << scale;

  return mojom::NOT_ERROR;
}

}  // namespace ml
