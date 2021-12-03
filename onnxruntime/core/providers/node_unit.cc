// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_unit.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"

namespace onnxruntime {

// class NodeUnitOrig : public INodeUnitOrig {
//  public:
//   NodeUnitOrig(const Node& node)
//       : node_(node),
//         all_nodes_{&node} {}
//
//   const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept override {
//     return node_.InputDefs();
//   }
//
//   const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept override {
//     return node_.OutputDefs();
//   }
//
//   const std::string& OpType() const noexcept override { return node_.OpType(); }
//   int SinceVersion() const noexcept override { return node_.SinceVersion(); }
//   const std::string& Domain() const noexcept override { return node_.Domain(); }
//   const Path& ModelPath() const noexcept override { return node_.ModelPath(); }
//   const std::string& Name() const noexcept override { return node_.Name(); }
//
//   const Node& GetNode() const noexcept override { return node_; }
//
//   // size_t GetInputEdgesCount() const noexcept override { return node_.GetInputEdgesCount(); }
//   NodeIndex Index() const noexcept override { return node_.Index(); }
//
//   ProviderType GetExecutionProviderType() const noexcept override { return node_.GetExecutionProviderType(); }
//
//   // Node::NodeConstIterator OutputNodesBegin() const noexcept override { return node_.OutputNodesBegin(); }
//
//   // Node::NodeConstIterator OutputNodesEnd() const noexcept override { return node_.OutputNodesEnd(); }
//
//   const std::vector<const Node*> GetAllNodes() const noexcept override { return all_nodes_; }
//
//   INodeUnit::Type UnitType() const noexcept override { return INodeUnit::Type::Node; }
//
//  private:
//   const Node& node_;
//   std::vector<const Node*> all_nodes_;
// };
//
// class QDQNodeUnit : public INodeUnitOrig {
//  public:
//   QDQNodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group);
//
//   const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept override {
//     return ConstPointerContainer<std::vector<NodeArg*>>(input_defs_);
//   }
//
//   const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept override {
//     return ConstPointerContainer<std::vector<NodeArg*>>(output_defs_);
//   }
//
//   const std::string& OpType() const noexcept override { return node_.OpType(); }
//   int SinceVersion() const noexcept override { return node_.SinceVersion(); }
//   const std::string& Domain() const noexcept override { return node_.Domain(); }
//   const Path& ModelPath() const noexcept override { return node_.ModelPath(); }
//   const std::string& Name() const noexcept override { return node_.Name(); }
//
//   const Node& GetNode() const noexcept override { return node_; }
//   NodeIndex Index() const noexcept override { return node_.Index(); }
//   ProviderType GetExecutionProviderType() const noexcept override { return node_.GetExecutionProviderType(); }
//
//   const std::vector<const Node*> GetAllNodes() const noexcept override { return all_nodes_; }
//
//   INodeUnit::Type UnitType() const noexcept override { return INodeUnit::Type::QDQ; }
//
//  private:
//   void init();
//   const GraphViewer& graph_viewer_;
//   const QDQ::NodeGroup qdq_group_;
//   const Node& node_;
//   std::vector<NodeArg*> input_defs_;
//   std::vector<NodeArg*> output_defs_;
//   std::vector<const Node*> all_nodes_;
// };
//
//// QUESTION/TODO, do we want to embed the graph_viewer into the QDQNodeGroup?
// QDQNodeUnit::QDQNodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group)
//     : graph_viewer_(graph_viewer),
//       qdq_group_(qdq_group),
//       node_(*graph_viewer_.GetNode(qdq_group_.target_node)) {
//   init();
// }
//
// void QDQNodeUnit::init() {
//   for (auto node_index : qdq_group_.dq_nodes) {
//     all_nodes_.push_back(graph_viewer_.GetNode(node_index));
//   }
//
//   for (auto node_index : qdq_group_.q_nodes) {
//     auto& node = const_cast<Node&>(*graph_viewer_.GetNode(node_index));
//     for (auto* def : node.MutableOutputDefs()) {
//       output_defs_.push_back(def);
//     }
//     all_nodes_.push_back(&node);
//   }
//
//   all_nodes_.push_back(&node_);
//
//   auto get_input = [&](std::vector<NodeArg*>& defs, NodeIndex node_index) {
//     // This is a bit hacky, but seems there is not other way to get a non-const NodeArg* from a const Node
//     auto& node = const_cast<Node&>(*graph_viewer_.GetNode(node_index));
//     defs.push_back(node.MutableInputDefs()[0]);
//   };
//
//   auto get_scale_zp = [&](std::vector<NodeArg*>& defs, NodeIndex node_index) {
//     // This is a bit hacky, but seems there is not other way to get a non-const NodeArg* from a const Node
//     auto& node = const_cast<Node&>(*graph_viewer_.GetNode(node_index));
//     defs.push_back(node.MutableInputDefs()[1]);
//     defs.push_back(node.MutableInputDefs()[2]);
//   };
//
//   auto get_all_input = [&](std::vector<NodeArg*>& defs, NodeIndex node_index) {
//     // This is a bit hacky, but seems there is not other way to get a non-const NodeArg* from a const Node
//     get_input(defs, node_index);
//     get_scale_zp(defs, node_index);
//   };
//
//   // Conv only, other may also need special handling
//   if (node_.OpType() == "Conv") {
//     get_all_input(input_defs_, qdq_group_.dq_nodes[0]);
//     get_all_input(input_defs_, qdq_group_.dq_nodes[1]);
//     get_scale_zp(input_defs_, qdq_group_.q_nodes[0]);
//     get_input(input_defs_, qdq_group_.dq_nodes[2]);
//   }
// }
//
// const std::unique_ptr<INodeUnit> CreateNodeUnit(const Node& node) {
//   return std::make_unique<NodeUnit>(node);
// }
//
// const std::unique_ptr<INodeUnit> CreateQDQNodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
//   return std::make_unique<QDQNodeUnit>(graph_viewer, qdq_group);
// }

// std::vector<IODef> input_defs_;
// std::vector<IODef> output_defs_;
// Node::EdgeSet input_edges_;  // TODO: maybe use a pointer and optional container to avoid copy if this is a plain node
// Node::EdgeSet output_edges_;
//
// const Node& node_;                // single node or target of QDQ
// std::vector<const Node*> nodes_;  // single node or all nodes in QDQ group

namespace {
std::vector<NodeUnit::IODef> DefsFromNode(const ConstPointerContainer<std::vector<NodeArg*>>& node_defs) {
  std::vector<NodeUnit::IODef> defs;
  defs.reserve(node_defs.size());

  for (const auto entry : node_defs) {
    defs.push_back(NodeUnit::IODef{entry, std::nullopt});
  }
}

std::vector<NodeUnit::IODef> InputDefsFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
  std::vector<NodeUnit::IODef> defs;

  const Node& target_node = *graph_viewer.GetNode(qdq_group.target_node);
  const auto node_defs = target_node.InputDefs();
  defs.reserve(node_defs.size());

  // TODO: for each real input, get DQ node, populate defs using DQ input def, zp and scale.
  int idx = 0;
  for (const auto entry : node_defs) {
    if (entry != nullptr && entry->Exists()) {
      // get 'real' input via edge.
      // could potentially have no edge if indices from initializer so leave unchanged in that case (TODO: validate assumption)
      const auto edge = graph_utils::GetInputEdge(target_node, idx);

      if (edge != nullptr) {
        // check if DQ or other
        const Node& node = edge->GetNode();
        NodeUnit::IODef new_def{node.InputDefs()[edge->GetSrcArgIndex()], std::nullopt};
        assert(new_def.nodearg != nullptr);  // TEMPORARY sanity check

        auto dq_node_iter = std::find(qdq_group.dq_nodes.cbegin(), qdq_group.dq_nodes.cend(), node.Index());
        if (dq_node_iter != qdq_group.dq_nodes.cend()) {
          // need additional metadata

        } else {
        }

      } else {
        // value from graph input or initializer so maintain original input def
        defs.push_back(NodeUnit::IODef{entry, std::nullopt});
      }
    } else {
      defs.push_back(NodeUnit::IODef{nullptr, std::nullopt});
    }
  }
  return defs;
}

std::vector<NodeUnit::IODef> OutputDefsFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
  std::vector<NodeUnit::IODef> defs;
  // TODO: for each output, get DQ node, populate defs using DQ input def, zp and scale.
  // ??? Do we need to handle outputs that may not go to a DQ node
  return defs;
}

std::vector<graph_utils::GraphEdge> InputEdgesFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {
  // get input edges for target node
  const Node& target_node = *graph_viewer.GetNode(qdq_group.target_node);
  auto node_edges = graph_utils::GraphEdge::GetNodeInputEdges(target_node);

  // now for each value, replace the source with the DQ node info, and add the metadata

  return node_edges;
}

std::vector<graph_utils::GraphEdge> OutputEdgesFromQDQGroup(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group) {}
}

NodeUnit::NodeUnit(const Node& node)
    : type_{Type::Node},
      input_defs_{DefsFromNode(node.InputDefs())},
      output_defs_{DefsFromNode(node.InputDefs())},
      input_edges_{graph_utils::GraphEdge::GetNodeInputEdges(node)},
      output_edges_{graph_utils::GraphEdge::GetNodeOutputEdges(node)},
      node_{node},
      nodes_{{&node}} {
}

NodeUnit::NodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group)
    : type_{Type::QDQ},
      input_defs_{InputDefsFromQDQGroup(graph_viewer, qdq_group)},
      output_defs_{OutputDefsFromQDQGroup(graph_viewer, qdq_group)},
      input_edges_{},
      output_edges_{},
      node_{node},
      nodes_{{&node}} {
}
};

}  // namespace onnxruntime
