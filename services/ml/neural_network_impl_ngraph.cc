// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/ml/neural_network_impl_ngraph.h"

#include <utility>

#include "services/ml/model_impl_ngraph.h"

namespace ml {

void NeuralNetworkImplNgraph::Create(ml::mojom::NeuralNetworkRequest request) {
  auto impl = std::make_unique<NeuralNetworkImplNgraph>();
  auto* impl_ptr = impl.get();
  impl_ptr->binding_ =
      mojo::MakeStrongBinding(std::move(impl), std::move(request));
}

NeuralNetworkImplNgraph::NeuralNetworkImplNgraph() = default;

NeuralNetworkImplNgraph::~NeuralNetworkImplNgraph() = default;

void NeuralNetworkImplNgraph::CreateModel(CreateModelCallback callback) {
  LOG(INFO) << "createModel";
  auto model_impl_ngraph = std::make_unique<ModelImplNgraph>();
  if (!model_impl_ngraph->Init()) {
    std::move(callback).Run(mojom::INCOMPLETE, nullptr);
    return;
  }

  auto init_params = mojom::ModelInitParams::New();
  mojom::ModelPtrInfo model_ptr_info;
  mojo::MakeStrongBinding(std::move(model_impl_ngraph),
                          mojo::MakeRequest(&model_ptr_info));
  init_params->model = std::move(model_ptr_info);

  std::move(callback).Run(mojom::NOT_ERROR, std::move(init_params));
}

}  // namespace ml
