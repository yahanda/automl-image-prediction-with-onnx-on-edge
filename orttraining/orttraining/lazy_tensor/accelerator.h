#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include "core/session/onnxruntime_cxx_api.h"
#include "orttraining/core/session/training_session.h"

using CompiledCode = std::function<std::vector<c10::IValue>(
    at::ArrayRef<c10::IValue>&)>;

class Accelerator {
 public:
  Accelerator(const torch::jit::Node* node)
      : subgraph_(node->g(torch::jit::attr::Subgraph)) { std::cout << "JIT see\n" << *node << std::endl; }
  void run(torch::jit::Stack& stack);
  static bool supported(const torch::jit::Node* node);

 private:

  void check_inputs(const at::ArrayRef<c10::IValue>& inputs);
  void propagate_input_types(const at::ArrayRef<c10::IValue>& inputs);
  std::string export_to_onnx();
  std::unique_ptr<onnxruntime::training::TrainingSession> create_session();
  CompiledCode compile(
    torch::jit::CompleteArgumentSpec spec, at::ArrayRef<c10::IValue>& inputs);
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, CompiledCode> cache_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, std::unique_ptr<onnxruntime::training::TrainingSession>> cached_sess_;
};
