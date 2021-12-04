// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_unit.h"
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
}

std::vector<NodeUnit::IODef> DefsFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group,
                                              bool input) {
  const Node& target_node = *graph_viewer.GetNode(qdq_group.target_node);
  const auto node_defs = input ? target_node.InputDefs() : target_node.OutputDefs();

  // initialize with original NodeArg entries
  std::vector<NodeUnit::IODef> defs;
  defs.reserve(node_defs.size());
  for (const auto* def : node_defs) {
    defs.push_back(NodeUnit::IODef{def});
  }

  // for any DQ inputs, replace the original with the DQ input and add the scale/zp metadata
  auto cur = input ? target_node.InputEdgesBegin() : target_node.OutputEdgesBegin();
  auto end = input ? target_node.InputEdgesEnd() : target_node.OutputEdgesEnd();

  for (; cur != end; ++cur) {
    const Node& node = cur->GetNode();  // src node if input, dst node if output

    const std::vector<NodeIndex>& dq_or_q_nodes = input ? qdq_group.dq_nodes : qdq_group.q_nodes;

    auto qdq_iter = std::find(dq_or_q_nodes.cbegin(), dq_or_q_nodes.cend(), node.Index());
    if (qdq_iter != dq_or_q_nodes.cend()) {
      const auto dq_or_q_defs = input ? node.InputDefs() : node.OutputDefs();
      const NodeArg* zp = dq_or_q_defs.size() > 2 ? dq_or_q_defs[2] : nullptr;

      // replace original def. if coming from DQ node we're replacing the destination of the edge (the consumer node)
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

// std::vector<NodeUnit::IODef> InputDefsFromQDQGroup2(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
//   std::vector<NodeUnit::IODef> defs;
//   // std::vector<graph_utils::GraphEdge> edges;
//
//   const Node& target_node = *graph_viewer.GetNode(qdq_group.target_node);
//   const auto node_defs = target_node.InputDefs();
//   defs.reserve(node_defs.size());
//
//   auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(target_node);
//
//   // iterate target node inputs. for DQ inputs, we use the orginal input to the DQ node.
//   // for non-DQ inputs we use the info as is
//   //
//   int idx = 0;
//   for (const auto entry : node_defs) {
//     const Node* dq_node{nullptr};
//     const graph_utils::GraphEdge* edge{nullptr};
//
//     for (auto it = input_edges.cbegin(), end = input_edges.cend(); it != end; ++it) {
//       if (it->dst_arg_index == idx) {
//         edge = &*it;
//
//         // check if DQ or other
//         const Node& src_node = *graph_viewer.GetNode(it->src_node);
//
//         auto dq_node_iter = std::find(qdq_group.dq_nodes.cbegin(), qdq_group.dq_nodes.cend(), src_node.Index());
//         if (dq_node_iter != qdq_group.dq_nodes.cend()) {
//           dq_node = &src_node;
//         } else {
//         }
//
//         break;  // input can only have one edge
//       }
//     }
//
//     ++idx;
//
//     if (dq_node) {
//       // use input from DQ and add scale and optional zero point metadata
//       const auto dq_inputs = dq_node->InputDefs();
//       const NodeArg* zp = dq_inputs.size() > 2 ? dq_inputs[2] : nullptr;
//       defs.push_back(NodeUnit::IODef{dq_inputs[0], NodeUnit::IODef::QDQMetadata{dq_inputs[1], zp}});
//
//       // create edge from DQ input to correct target node src idx
//
//       //
//       // TODO:!!! Need to have a simple, consistent approach to getting, checking, updating edges
//       // Should we use graph_utils or Node?
//       //
//
//       const auto dq_input_edge = graph_utils::GraphEdge::GetN
//                                      edges.push_back(graph_utils::GraphEdge())
//
//     } else {
//       // value from non-DQ node, graph input or initializer, so use original input def
//       defs.push_back(NodeUnit::IODef{entry, std::nullopt});
//       edges.push_back(*edge);
//     }
//
//     return defs;
//   }
//
//   std::vector<NodeUnit::IODef> OutputDefsFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
//     std::vector<NodeUnit::IODef> defs;
//
//     const Node& target_node = *graph_viewer.GetNode(qdq_group.target_node);
//     const auto node_defs = target_node.OutputDefs();
//     defs.reserve(node_defs.size());
//
//     int idx = 0;
//
//     // iterate target node inputs. for DQ inputs, we use the orginal input to the DQ node.
//     // for non-DQ inputs we use the info as is
//     //
//     for (const auto entry : node_defs) {
//       const auto edge = graph_utils::GetOutputEdge(target_node, idx);
//       ++idx;
//
//       if (edge != nullptr) {
//         // check if Q
//         const Node& dst_node = edge->GetNode();
//         auto q_node_iter = std::find(qdq_group.q_nodes.cbegin(), qdq_group.q_nodes.cend(), dst_node.Index());
//
//         if (q_node_iter != qdq_group.q_nodes.cend()) {
//           // use output from Q, and add scale and optional zero point metadata, and 'axis' attribute if set
//           const auto q_inputs = dst_node.InputDefs();
//           const auto* axis_attr = graph_utils::GetNodeAttribute(dst_node, "axis");
//           int axis = 1;
//           if (axis_attr) {
//             axis = axis_attr->i();
//           }
//
//           const NodeArg* zp = q_inputs.size() > 2 ? q_inputs[2] : nullptr;
//           defs.push_back(NodeUnit::IODef{dst_node.OutputDefs()[0],
//                                          NodeUnit::IODef::QDQMetadata{q_inputs[1], zp, axis}});
//           continue;
//         }
//       }
//
//       // value is not consumed by a Q node, so keep original
//       defs.push_back(NodeUnit::IODef{entry, std::nullopt});
//     }
//
//     return defs;
//   }
//
//   std::vector<graph_utils::GraphEdge> InputEdgesFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
//     // get input edges for target node
//     const Node& target_node = *graph_viewer.GetNode(qdq_group.target_node);
//     auto node_edges = graph_utils::GraphEdge::GetNodeInputEdges(target_node);
//
//     // now for each value, replace the source with the DQ node info, and add the metadata
//
//     return node_edges;
//   }
//
//   std::vector<graph_utils::GraphEdge> OutputEdgesFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {}
// }

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
      // input_edges_{graph_utils::GraphEdge::GetNodeInputEdges(node)},
      // output_edges_{graph_utils::GraphEdge::GetNodeOutputEdges(node)},
      node_{node},
      nodes_{{&node}} {
}

NodeUnit::NodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group)
    : type_{Type::QDQ},
      input_defs_{InputDefsFromQDQGroup(graph_viewer, qdq_group)},
      output_defs_{OutputDefsFromQDQGroup(graph_viewer, qdq_group)},
      // input_edges_{},
      // output_edges_{},
      node_{*graph_viewer.GetNode(qdq_group.target_node)},
      nodes_{GetQDQNodes(graph_viewer, qdq_group)} {
}
}  // namespace onnxruntime
