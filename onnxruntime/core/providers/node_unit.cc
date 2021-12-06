// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_unit.h"

#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace {

std::vector<NodeUnit::IODef> DefsFromNode(const ConstPointerContainer<std::vector<NodeArg*>>& node_defs) {
  std::vector<NodeUnit::IODef> defs;
  defs.reserve(node_defs.size());

  for (const auto entry : node_defs) {
    defs.push_back(NodeUnit::IODef{entry, std::nullopt});
  }

  return defs;
}

std::vector<NodeUnit::IODef> DefsFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group,
                                              bool input) {
  const Node& target_node = *graph_viewer.GetNode(qdq_group.target_node);
  const auto node_defs = input ? target_node.InputDefs() : target_node.OutputDefs();

  // initialize with original NodeArg entries
  std::vector<NodeUnit::IODef> defs{DefsFromNode(input ? target_node.InputDefs() : target_node.OutputDefs())};

  // for any DQ inputs, replace the original with the DQ input and add the scale/zp metadata
  auto cur = input ? target_node.InputEdgesBegin() : target_node.OutputEdgesBegin();
  auto end = input ? target_node.InputEdgesEnd() : target_node.OutputEdgesEnd();

  for (; cur != end; ++cur) {
    const Node& node = cur->GetNode();  // src node if input, dst node if output
    const auto& dq_or_q_nodes = input ? qdq_group.dq_nodes : qdq_group.q_nodes;

    auto qdq_iter = std::find(dq_or_q_nodes.cbegin(), dq_or_q_nodes.cend(), node.Index());
    if (qdq_iter != dq_or_q_nodes.cend()) {
      const auto dq_or_q_defs = input ? node.InputDefs() : node.OutputDefs();
      const NodeArg* zp = dq_or_q_defs.size() > 2 ? dq_or_q_defs[2] : nullptr;

      // replace original def.
      // if coming from DQ node we're replacing the destination of the edge - the input index on the target node
      // if going to the Q node we're replacing the source of the edge - the output index on the target node
      auto idx = input ? cur->GetDstArgIndex() : cur->GetSrcArgIndex();
      defs[idx] = NodeUnit::IODef{dq_or_q_defs[0], NodeUnit::IODef::QDQMetadata{dq_or_q_defs[1], zp}};
    }
  }

  return defs;
}

std::vector<NodeUnit::IODef> InputDefsFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
  return DefsFromQDQGroup(graph_viewer, qdq_group, true);
}

std::vector<NodeUnit::IODef> OutputDefsFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
  return DefsFromQDQGroup(graph_viewer, qdq_group, false);
}

std::vector<const Node*> GetQDQNodes(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
  std::vector<const Node*> nodes(qdq_group.dq_nodes.size() + 1 + qdq_group.q_nodes.size());
  auto insert = [&](const NodeIndex idx) { nodes.push_back(graph_viewer.GetNode(idx)); };

  std::for_each(qdq_group.dq_nodes.cbegin(), qdq_group.dq_nodes.cend(), insert);
  insert(qdq_group.target_node);
  std::for_each(qdq_group.q_nodes.cbegin(), qdq_group.q_nodes.cend(), insert);

  return nodes;
}
}  // namespace

NodeUnit::NodeUnit(const Node& node)
    : type_{Type::Node},
      input_defs_{DefsFromNode(node.InputDefs())},
      output_defs_{DefsFromNode(node.InputDefs())},
      node_{node},
      nodes_{{&node}} {
}

NodeUnit::NodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group)
    : type_{Type::QDQ},
      input_defs_{InputDefsFromQDQGroup(graph_viewer, qdq_group)},
      output_defs_{OutputDefsFromQDQGroup(graph_viewer, qdq_group)},
      node_{*graph_viewer.GetNode(qdq_group.target_node)},
      nodes_{GetQDQNodes(graph_viewer, qdq_group)} {
}
}  // namespace onnxruntime
