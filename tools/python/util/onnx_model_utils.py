import logging
import onnx
import onnxruntime as ort
import pathlib
from onnx import version_converter


def iterate_graph_per_node_func(graph, per_node_func, **func_args):
    '''
    Iterate the graph including subgraphs calling the per_node_func for each node.
    :param graph: Graph to iterate
    :param per_node_func: Function to call for each node. Signature is fn(node: onnx:NodeProto, **kwargs)
    :param func_args: The keyword args to pass through.
    '''

    for node in graph.node:
        per_node_func(node, **func_args)
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                iterate_graph_per_node_func(attr.g, per_node_func, **func_args)


def iterate_graph_per_graph_func(graph, per_graph_func, **func_args):
    '''
    Iterate the graph including subgraphs calling the per_graph_func for each Graph.
    :param graph: Graph to iterate
    :param per_graph_func: Function to call for each graph. Signature is fn(node: onnx:GraphProto, **kwargs)
    :param func_args: The keyword args to pass through.
    '''

    per_graph_func(graph, **func_args)

    for node in graph.node:
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                iterate_graph_per_graph_func(attr.g, per_graph_func, **func_args)


def update_onnx_opset(model_path: pathlib.Path, opset: int, out_path: pathlib.Path = None,
                      logger: logging.Logger = None):
    """
    Helper to update the opset of a model using onnx version_converter. Target opset must be greater than current opset.
    Model is saved to the original location with the '.onnx' extension replaced with '.opset<opset>.onnx'.
    :param model_path: Path to model to update
    :param opset: Opset to update model to
    :param out_path: Optional output path for updated model. If not provided will write to the model_path with the
                     '.onnx' extension replaced by '.opset<opset>.onnx'. e.g. model.onnx updated to opset 13 will be
                     written to model.opset13.onnx.
    :param logger: Optional logger for diagnostic output
    """

    if logger:
        logger.info("Updating %s to opset %d", model_path, opset)

    model = onnx.load(str(model_path))
    new_model = version_converter.convert_version(model, opset)
    # save with .onnx -> .opsetX.onnx
    if not out_path:
        out_path = str(model_path.with_suffix(f'.opset{opset}.onnx'))

    onnx.save(new_model, out_path)

    if logger:
        logger.info("Saved updated model to %s", model_path)


def optimize_model(model_path: pathlib.Path,
                   output_path: pathlib.Path = None,
                   level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC):
    '''
    Optimize an ONNX model using ONNX Runtime to the specified level
    :param model_path: Path to ONNX model
    :param output_path: Optional output path. If not specified the '.onnx' extention of model_path will be replaced
                        with '.optimized.onnx'.
    :param level: onnxruntime.GraphOptimizationLevel to use. Default is ORT_ENABLE_BASIC.
    :return: output_path that was used.
    '''

    if not output_path:
        output_path = model_path.with_suffix(f'.optimized.onnx')

    so = ort.SessionOptions()
    so.optimized_model_filepath = str(output_path)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    # create session to optimize
    _ = ort.InferenceSession(str(model_path), so)

    return output_path


def _replace_symbolic_dim_value(graph: onnx.GraphProto, **kwargs):
    '''
    Iterate all values in the graph, replacing dim_param in a tensor shape with the provided value.
    :param graph: GraphProto to update
    :param dim_param: dim_param to set
    :param value: value to use
    '''

    param_to_replace = kwargs['dim_param']
    value = kwargs['value']

    def update_dim_values(value_infos):
        for vi in value_infos:
            if vi.type.HasField("tensor_type"):
                shape = vi.type.tensor_type.shape
                if shape:
                    for dim in shape.dim:
                        if dim.HasField('dim_param') and dim.dim_param == param_to_replace:
                            dim.Clear()
                            dim.dim_value = value

    update_dim_values(graph.input)
    update_dim_values(graph.output)
    update_dim_values(graph.value_info)


def make_dim_param_fixed(model: onnx.ModelProto, param_name: str, value: int):
    iterate_graph_per_graph_func(model.graph, _replace_symbolic_dim_value, dim_param=param_name, value=value)


def _create_producer_consumer_link(node_to_producers: dict, node_to_consumers: dict,
                                   producer: onnx.NodeProto, consumer: onnx.NodeProto):
    '''
    Create links between two nodes for a value produced by one and consumed by the other.
    :param node_to_producers: Map of NodeProto to set of nodes that produce values the node consumes as inputs.
    :param node_to_consumers: Map of NodeProto to set of nodes that consume values the node produces as outputs.
    :param producer: Producer node
    :param consumer: Consumer node
    '''

    if consumer not in node_to_producers:
        node_to_producers[consumer] = set()

    if producer not in node_to_consumers:
        node_to_consumers[producer] = set()

    # add entry mapping this node to the producer of this input
    node_to_producers[consumer].add(producer)
    node_to_consumers[producer].add(consumer)


def _map_node_dependencies(graph: onnx.GraphProto, node_to_producers: dict, node_to_consumers: dict):
    graph_inputs = set([i.name for i in graph.input])
    initializers = set([i.name for i in graph.initializer])

    # map of value name to node that creates it. copy parent values but override if values get shadowed
    producers = {}

    implicit_inputs = set()

    def is_local_value(value):
        return value in producers or value in initializers or value in graph_inputs

    for node in graph.node:
        inputs = [i for i in node.input]

        for attr in node.attribute:
            if attr.HasField('g'):
                subgraph_implicit_inputs = _map_node_dependencies(attr.g, node_to_producers, node_to_consumers)
                inputs += subgraph_implicit_inputs

        for i in inputs:
            if is_local_value(i):
                if i in producers:
                    producer = producers[i]
                    _create_producer_consumer_link(node_to_producers, node_to_consumers, producer, node)
            else:
                # not produced above us, not in initializers for this graph. may be graph input or initializer
                # in parent graph
                implicit_inputs.add(i)

        for o in node.output:
            producers[o] = node

    return implicit_inputs


def get_producer_consumer_maps(graph: onnx.GraphProto):
    '''
    Get maps for connections between the nodes that produces each value and the nodes that consumer the value.
    Processing includes subgraphs. As the map key is a Node instance from the Graph there should be no ambiguity.
    :param graph: Graph to process.
    :return: Tuple with two maps.
             First is node_to_producers map of a node to set of all nodes producing input it consumes.
             Second is node_to_consumers map of a node to set of all nodes consuming output it creates.
             e.g. NodeA and NodeB provide inputs to NodeC. NodeC provides input to NodeD
             node_to_consumers[NodeA] = set([NodeC])
             node_to_consumers[NodeB] = set([NodeC])
             node_to_producers[NodeC] = set([NodeA, NodeB])
             node_to_consumers[NodeC] = set([NodeD])
             node_to_producers[NodeD] = set([NodeC])
    '''

    # use a hash of the object id for NodeProto.
    # we need this for the partitioning checker where we keep maps with nodes as the key.
    onnx.NodeProto.__hash__ = lambda self: id(self)

    node_to_producers = {}  # map of node instance to nodes producing input values it consumes
    node_to_consumers = {}  # map of node instance to nodes consuming output values it produces

    implicit_inputs = _map_node_dependencies(graph, node_to_producers, node_to_consumers)

    # top level graph should have no implicit inputs
    assert(not implicit_inputs)

    return node_to_producers, node_to_consumers
