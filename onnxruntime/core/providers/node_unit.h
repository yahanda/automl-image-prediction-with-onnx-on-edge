// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <optional>

#include "core/graph/graph.h"

namespace onnxruntime {

class GraphViewer;

namespace QDQ {
struct NodeGroup;
}

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
    };

    const NodeArg* nodearg{nullptr};
    std::optional<QDQMetadata> qdq_metadata;
  };

  Type UnitType() const noexcept { return type_; }

  const std::vector<IODef>& InputDefs() const noexcept { return input_defs_; }
  const std::vector<IODef>& OutputDefs() const noexcept { return output_defs_; }

  const std::string& Domain() const noexcept { return node_.Domain(); }
  const std::string& OpType() const noexcept { return node_.OpType(); }
  const std::string& Name() const noexcept { return node_.Name(); }
  int SinceVersion() const noexcept { return node_.SinceVersion(); }

  const Node& GetNode() const noexcept { return node_; }
  NodeIndex Index() const noexcept { return node_.Index(); }

  ProviderType GetExecutionProviderType() const noexcept { return node_.GetExecutionProviderType(); }

  // single node if Type is Node, or all nodes in QDQ group if Type is QDQ
  // TODO : Do we need Node* or is NodeIndex fine ? Latter is simpler to setup with QDQ as the QDQ group is node indexes
  const std::vector<const Node*> GetAllNodes() const noexcept { return nodes_; }

  const Path& ModelPath() const noexcept { return node_.ModelPath(); }

 private:
  Type type_;

  std::vector<IODef> input_defs_;
  std::vector<IODef> output_defs_;

  const Node& node_;                // single node or target of QDQ
  std::vector<const Node*> nodes_;  // single node or all nodes in QDQ group
};

}  // namespace onnxruntime
