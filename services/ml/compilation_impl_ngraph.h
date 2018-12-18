// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_COMPILATION_IMPL_NGRAPH_H_
#define SERVICES_ML_COMPILATION_IMPL_NGRAPH_H_

#include <vector>

#include "base/macros.h"
#include "services/ml/common.h"
#include "services/ml/public/interfaces/compilation.mojom.h"

namespace ml {

class ModelImplNgraph;

class CompilationImplNgraph : public mojom::Compilation {
 public:
  explicit CompilationImplNgraph(const ModelImplNgraph*);
  ~CompilationImplNgraph() override;

  void Finish(int32_t preference, FinishCallback callback) override;
  void CreateExecution(CreateExecutionCallback callback) override;

 private:
  friend class ExecutionImplNgraph;
  const ModelImplNgraph* model_;

  std::vector<Operand> operands_;
  std::vector<Operation> operations_;
  std::vector<uint32_t> inputs_;
  std::vector<uint32_t> outputs_;

  DISALLOW_COPY_AND_ASSIGN(CompilationImplNgraph);
};

}  // namespace ml

#endif  // SERVICES_ML_COMPILATION_IMPL_NGRAPH_H_