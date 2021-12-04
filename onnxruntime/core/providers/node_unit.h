// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <optional>

// #include "core/graph/basic_types.h"
// Need move Node::NodeConstIterator out of Node for forward declaration
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

// template <typename Container>
// class ConstPointerContainer;
// class Node;
// class NodeArg;
// class Path;
class GraphViewer;

namespace QDQ {
struct NodeGroup;
}

// class INodeUnitOrig {
//  public:
//   enum class Type : uint8_t {
//     Node,
//     QDQ
//   };
//
//   virtual ~INodeUnitOrig() = default;
//
//   virtual const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept = 0;
//   virtual const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept = 0;
//
//   virtual const std::string& OpType() const noexcept = 0;
//   virtual int SinceVersion() const noexcept = 0;
//   virtual const std::string& Domain() const noexcept = 0;
//   virtual const Path& ModelPath() const noexcept = 0;
//   virtual const std::string& Name() const noexcept = 0;
//
//   virtual const Node& GetNode() const noexcept = 0;
//
//   // virtual size_t GetInputEdgesCount() const noexcept = 0;
//   virtual NodeIndex Index() const noexcept = 0;
//
//   virtual ProviderType GetExecutionProviderType() const noexcept = 0;
//
//   // virtual Node::NodeConstIterator OutputNodesBegin() const noexcept = 0;
//   // virtual Node::NodeConstIterator OutputNodesEnd() const noexcept = 0;
//
//   virtual const std::vector<const Node*> GetAllNodes() const noexcept = 0;
//
//   virtual Type UnitType() const noexcept = 0;
// };

class NodeUnit {
 public:
  NodeUnit(const Node& node);
  NodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& qdq_group);

  enum class Type : uint8_t {
    Node,
    QDQ
  };

  ~NodeUnit() = default;

  struct IODef {
    struct QDQMetadata {
      const NodeArg* scale{nullptr};
      const NodeArg* zero_point{nullptr};
      // int q_axis{1};  // QuantizeLinear 'axis' attribute. Ignore for DQ
    };

    const NodeArg* nodearg{nullptr};
    std::optional<QDQMetadata> qdq_metadata;
  };

  Type UnitType() const noexcept { return type_; }

  const std::vector<IODef>& InputDefs() const noexcept { return input_defs_; }
  const std::vector<IODef>& OutputDefs() const noexcept { return output_defs_; }

  // const std::vector<graph_utils::GraphEdge>& InputEdges() const noexcept { return input_edges_; }
  // const std::vector<graph_utils::GraphEdge>& OutputEdges() const noexcept { return output_edges_; }

  const std::string& Domain() const noexcept { return node_.Domain(); }
  const std::string& OpType() const noexcept { return node_.OpType(); }
  const std::string& Name() const noexcept { return node_.Name(); }
  int SinceVersion() const noexcept { return node_.SinceVersion(); }

  const Node& GetNode() const noexcept { return node_; }
  NodeIndex Index() const noexcept { return node_.Index(); }

  ProviderType GetExecutionProviderType() const noexcept { return node_.GetExecutionProviderType(); }

  // virtual Node::NodeConstIterator OutputNodesBegin() const noexcept = 0;
  // virtual Node::NodeConstIterator OutputNodesEnd() const noexcept = 0;

  // single node if Type is Node, or all nodes in QDQ group if Type is QDQ
  const std::vector<const Node*> GetAllNodes() const noexcept { return nodes_; }

  const Path& ModelPath() const noexcept { return node_.ModelPath(); }

 private:
  Type type_;

  std::vector<IODef> input_defs_;
  std::vector<IODef> output_defs_;
  // std::vector<graph_utils::GraphEdge> input_edges_;
  // std::vector<graph_utils::GraphEdge> output_edges_;

  const Node& node_;                // single node or target of QDQ
  std::vector<const Node*> nodes_;  // single node or all nodes in QDQ group - TODO: Do we need Node* or is NodeIndex fine? Latter is simpler to setup.
};

}  // namespace onnxruntime
