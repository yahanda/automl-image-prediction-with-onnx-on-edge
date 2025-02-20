// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"

#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace QDQ {
namespace {
// adjust for an optional input/output that has an entry but does not exist
int NumActualValues(const Node& node, bool input) {
  const auto& defs = input ? node.InputDefs() : node.OutputDefs();
  return gsl::narrow_cast<int>(std::count_if(defs.cbegin(), defs.cend(),
                                             [](const NodeArg* def) { return def && def->Exists(); }));
}
}  // namespace

static std::vector<const Node*> FindQDQNodes(const GraphViewer& graph_viewer, const Node& node, bool find_dq_nodes) {
  // First get all the upstream (DQ) or downstream (Q) nodes
  std::vector<const Node*> nodes =
      find_dq_nodes ? graph_utils::FindParentsByType(node, QDQ::DQOpName)
                    : graph_utils::FindChildrenByType(node, QDQ::QOpName);

  // Remove all the nodes which are not in the graph_viewer
  nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                             [&graph_viewer](const Node* _node) {
                               return _node == nullptr || graph_viewer.GetNode(_node->Index()) == nullptr;
                             }),
              nodes.end());

  return nodes;
}

bool NodeGroupSelector::CheckQDQNodes(const GraphViewer& graph_viewer, const Node& node,
                                      const std::vector<const Node*>& dq_nodes,
                                      const std::vector<const Node*>& q_nodes,
                                      int num_dq_inputs) const {
  if (num_dq_inputs == -1) {
    num_dq_inputs = NumActualValues(node, true);
  }

  int num_outputs = NumActualValues(node, false);  // number of outputs that exist

  // The input is a Graph Viewer, so cannot use graph_utils or optimizer_utils
  return num_dq_inputs == gsl::narrow_cast<int>(dq_nodes.size()) &&
         num_outputs == gsl::narrow_cast<int>(q_nodes.size()) &&
         q_nodes.size() == node.GetOutputEdgesCount() &&
         !graph_viewer.NodeProducesGraphOutput(node);
}

std::optional<NodeGroup> NodeGroupSelector::GetQDQSelection(const GraphViewer& graph_viewer, const Node& node) const {
  std::vector<const Node*> dq_nodes = FindQDQNodes(graph_viewer, node, true);
  std::vector<const Node*> q_nodes = FindQDQNodes(graph_viewer, node, false);
  if (!Check(graph_viewer, node, dq_nodes, q_nodes)) {
    return std::nullopt;
  }

  NodeGroup node_group;
  node_group.dq_nodes.reserve(dq_nodes.size());
  node_group.q_nodes.reserve(q_nodes.size());
  node_group.target_node = node.Index();
  auto get_node_idx = [&](const Node* n) { return n->Index(); };
  std::transform(dq_nodes.begin(), dq_nodes.end(), std::back_inserter(node_group.dq_nodes), get_node_idx);
  std::transform(q_nodes.begin(), q_nodes.end(), std::back_inserter(node_group.q_nodes), get_node_idx);
  return node_group;
}

std::optional<NodesToOptimizeIndices> BaseSelector::Select(const GraphViewer& graph_viewer, const Node& node) const {
  const auto qdq_group = node_group_selector_->GetQDQSelection(graph_viewer, node);
  if (!qdq_group.has_value()) {
    return std::nullopt;
  }

  NodesToOptimizeIndicesBuilder builder;
  builder.input_nodes = qdq_group->dq_nodes;
  builder.output_nodes = qdq_group->q_nodes;
  builder.target_node = qdq_group->target_node;

  UpdateBuilder(builder);
  return builder.Build();
}

bool DropQDQNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     const std::vector<const Node*>& dq_nodes,
                                     const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  const Node& dq_node = *dq_nodes.front();
  const Node& q_node = *q_nodes.front();

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  return IsQDQPairSupported(q_node, dq_node, get_const_initializer, graph_viewer.ModelPath());
}

bool DropDQNodeGroupSelector::CheckDQNodes(const Node& node, const std::vector<const Node*>& dq_nodes) const {
  int num_dq_inputs = NumActualValues(node, true);

  return num_dq_inputs == gsl::narrow_cast<int>(dq_nodes.size());
}

bool DropDQNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  if (!CheckDQNodes(node, dq_nodes)) {
    return false;
  }

  (void)q_nodes;
  const Node& dq_node = *dq_nodes.front();

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  return IsDQSupported(dq_node, get_const_initializer);
}

bool UnaryNodeGroupSelector::Check(const GraphViewer& graph_viewer, const Node& node,
                                   const std::vector<const Node*>& dq_nodes,
                                   const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  return dt_input == dt_output;
}

bool BinaryNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
    return false;
  }

  int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_input_1 == dt_input_2 &&
         dt_input_1 == dt_output;
}

bool VariadicNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                      const Node& node,
                                      const std::vector<const Node*>& dq_nodes,
                                      const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
    return false;
  }

  // All DQs' inputs and Q's output should have same data type
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  for (size_t dq_idx = 1; dq_idx < dq_nodes.size(); dq_idx++) {
    if (dt_input != dq_nodes[dq_idx]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      return false;
    }
  }

  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_input == dt_output;
}

void VariadicSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  builder.num_input_defs = 1;  // set to 1 as the first input is variadic
}

bool ConvNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  const std::vector<const Node*>& dq_nodes,
                                  const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
    return false;
  }

  // input and output types need to be same
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_weight = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dt_input != dt_output) {
    return false;
  }

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (!int8_allowed_ || dt_weight != dt_input) {
      return false;
    }
  }

  if (dq_nodes.size() < 3) {  // no bias
    return true;
  }

  int32_t dt_bias = dq_nodes[2]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_bias == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
}

void ConvSelector::UpdateBuilder(NodesToOptimizeIndicesBuilder& builder) const {
  builder.input_nodes.resize(3, NodesToOptimizeIndices::kEmptyNodeIndex);
}

bool MatMulNodeGroupSelector::Check(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
  if (dq_nodes.size() != 2) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_weight = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (!int8_allowed_ || dt_weight != dt_input) {
      return false;
    }
  }

  // potential match for QLinearMatMul or MatMulIntegerToFloat
  bool qlinear = !q_nodes.empty();

  if (qlinear) {
    // QLinearMatMul
    if (!CheckQDQNodes(graph_viewer, node, dq_nodes, q_nodes)) {
      return false;
    }

    int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    return dt_input == dt_output;
  } else {
    // can be converted to MatMulIntegerToFloat if EP supports that.
    return matmulintegertofloat_allowed_;
  }
}

}  // namespace QDQ
}  // namespace onnxruntime

#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
