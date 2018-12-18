// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/ml/execution_impl_ngraph.h"

#include <utility>

#include "base/strings/string_number_conversions.h"
#include "services/ml/compilation_impl_ngraph.h"
#include "services/ml/model_impl_ngraph.h"

namespace ml {

ExecutionImplNgraph::ExecutionImplNgraph(const CompilationImplNgraph* compilation,
                                         mojo::ScopedSharedBufferHandle memory) {
  operands_ = compilation->operands_;
  operations_ = compilation->operations_;
  inputs_ = compilation->inputs_;
  outputs_ = compilation->outputs_;
  memory_ = std::move(memory);
  uint32_t total_length = 0;
  inputs_info_.reserve(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    const uint32_t offset = total_length;
    const uint32_t length = operands_[inputs_[i]].requiredSize();
    inputs_info_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
    total_length += length;
  }
  outputs_info_.reserve(outputs_.size());
  for (size_t i = 0; i < outputs_.size(); ++i) {
    const uint32_t offset = total_length;
    const uint32_t length = operands_[outputs_[i]].requiredSize();
    outputs_info_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
    total_length += length;
  }
}

ExecutionImplNgraph::~ExecutionImplNgraph() {
}

void ExecutionImplNgraph::StartCompute(StartComputeCallback callback) {
  DLOG(INFO) << "ExecutionImplNgraph::StartCompute";

  for (size_t i = 0; i < inputs_.size(); ++i) {
    DLOG(INFO) << "inputs[" << i << "]:";
    PrintOperand(operands_[inputs_[i]], inputs_info_[i]);
  }

  for (size_t i = 0; i < outputs_.size(); ++i) {
    DLOG(INFO) << "outputs[" << i << "]:";
    PrintOperand(operands_[outputs_[i]], outputs_info_[i]);
  }

  std::move(callback).Run(mojom::NOT_ERROR);
}

}  // namespace ml
