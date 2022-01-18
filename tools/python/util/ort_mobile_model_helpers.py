import logging
import onnx
import onnx.shape_inference
import os
import pathlib
import torch
import torchvision

from collections import deque

FUNC_NAME_WIDTH = 24
FORMAT = '%(funcName)' + str(FUNC_NAME_WIDTH) + 's %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('default')
logger.setLevel(logging.INFO)


def update_onnx_opset(model_path, opset):
    """
    Helper to update the opset of a model using onnx version_converter. Target opset must be greater than current opset.
    Model is saved to the original location with the '.onnx' extension replaced with '.opset<opset>.onnx'.
    :param model_path: Path to model to update
    :param opset: Opset to update model to
    """
    from onnx import version_converter

    logger.info("Updating %s to opset %d", model_path, opset)
    model = onnx.load(model_path)
    new_model = version_converter.convert_version(model, opset)
    # save with .onnx -> .opsetX.onnx
    onnx.save(new_model, str(pathlib.Path(model_path).with_suffix(f'.opset{opset}.onnx')))
    logger.info("Saved updated model to %s", model_path)


def map_node_dependencies(model):
    # initializers = set()
    producers = {}  # map of value to node that creates it.
    node_to_producers = {}  # map of node instance to nodes producing input values it consumes
    node_to_consumers = {}  # map of node instance to nodes consuming output values it produces

    for node in model.graph.node:
        # TODO add support for subgraphs by recursing into Graph attribute
        # Neither NNAPI or CoreML EPs support currently so ignore for now
        # for attr in node.attribute:
        #     if attr.HasField('g'):
        #        process_graph(attr.g, ...)

        for i in node.input:
            if i and i in producers:
                # input is valid and is creatd by an upstream node

                producer = producers[i]

                if node not in node_to_producers:
                    node_to_producers[node] = set()

                if producer not in node_to_consumers:
                    node_to_consumers[producer] = set()

                # add entry mapping this node to the producer of this input
                node_to_producers[node].add(producer)
                node_to_consumers[producer].add(node)

        for o in node.output:
            producers[o] = node

    return node_to_producers, node_to_consumers


class SupportedOpsChecker:
    '''
    Class to process the md file with list of supported ops and caveats for an execution provider.
    e.g. /tools/ci_build/github/android/nnapi_supported_ops.md
         /tools/ci_build/github/apple/coreml_supported_ops.md
    '''

    def __init__(self, filename):
        self._filename = filename
        self._ops = {}  # op to caveat
        self._ops_seen = set()

        with open(filename, 'r') as f:
            for line in f.readlines():
                # we're looking for a markdown table with 2 columns. first is op name. second is caveats
                # op name is domain:op
                if line.startswith('|'):
                    pieces = line.strip().split('|')
                    if len(pieces) == 4:  # pre-first '|'. op, caveat, post-last '|'
                        domain_op = pieces[1]
                        caveat = pieces[2]
                        caveat = caveat.replace('<br/>', ' ')  # remove some HTML tags
                        # skip lines that don't have the ':' which separates the domain and op
                        # e.g. the table header will fail this check
                        if ':' in domain_op:
                            self._ops[domain_op] = caveat

    def is_op_supported(self, node):
        domain = node.domain if node.domain else 'ai.onnx'
        domain_op = domain + ':' + node.op_type

        is_supported = domain_op in self._ops
        if is_supported:
            self._ops_seen.add(domain_op)

        return is_supported

    def get_caveats(self):
        caveats = []
        for op in sorted(self._ops_seen):
            caveat = self._ops[op]
            if caveat:
                caveats.append(f'{op}:{caveat}')

        return caveats


def is_fixed_size_tensor(value: onnx.ValueInfoProto):
    is_fixed = False
    if value.type.HasField("tensor_type"):
        shape = value.type.tensor_type.shape
        if shape:
            is_fixed = True  # scalar has no dims so set to True and unset if we hit a dim without a valid value
            for dim in shape.dim:
                if dim.HasField('dim_value') and dim.dim_value > 0:
                    continue

                # anything else means it's a dynamic value
                is_fixed = False
                break

    return is_fixed


def check_partitioning(model: onnx.ModelProto, supported_ops_checker: SupportedOpsChecker, ep_name: str,
                       require_fixed_input_sizes: bool = False, value_info=None):
    '''
    Estimate the partitions the graph will be split into for nodes that is_node_supported_fn returns true for.

    The check on whether a node is supported is purely based on the operator type. Additional limitations
    (e.g. NNAPI EP only supports 2D Conv) are not checked, so partitions may not be 100% accurate. The limitations
    for operators in the partitions are printed so the user can manually check.
    :param model: Model to process
    :param supported_ops_checker: Checker with info on supported ops.
    :param require_fixed_input_sizes: If True, require that the inputs to a potentially supported node are
                                       fixed size tensors for it to be considered as supported.
                                       If True, onnx.shape_inference.infer_shapes should have been run on the model
                                       to populate the shape information.
    :param value_info: Map of value name to ValueInfoProto. Required if require_fixed_input_sizes is True to lookup
                       the shape of a value.
    '''

    if require_fixed_input_sizes and not value_info:
        raise ValueError("value_info must be provided if require_fixed_input_sizes is True.")

    node_to_producers, node_to_consumers = map_node_dependencies(model)

    #
    # Replicate logic from /onnxruntime/core/providers/partitioning_utils.cc:CreateSupportedPartitionNodeGroups
    # to roughly estimate number of partitions for nodes that is_node_supported_fn returns true for.
    #

    # we don't currently support a callback for additional group closure checks
    on_group_closed_fn = None

    supported_groups = []
    # number of inputs from unprocessed nodes (in-degree) per node
    in_degree = {}
    # nodes that are ready to process
    nodes_to_process = deque()  # deque of Node instances
    # nodes that will be processed when considering the next partition node group
    nodes_to_process_with_next_group = deque()

    # initialize in-degrees and find root nodes
    for node in model.graph.node:
        node_input_edge_count = len(node_to_producers[node]) if node in node_to_producers else 0
        in_degree[node] = node_input_edge_count
        if node_input_edge_count == 0:
            # node is only dependent on graph input or initializers
            nodes_to_process.append(node)

    supported_group = []
    # the partition node group's border is the aggregate of its nodes' output nodes
    supported_group_border = set()
    num_supported_nodes = 0
    num_unsupported_nodes_due_to_op = 0
    num_unsupported_nodes_due_to_dynamic_input = 0
    unsupported_ops = set()

    def close_group():
        if supported_group:
            keep_partition = not on_group_closed_fn or on_group_closed_fn(supported_group)

            if keep_partition:
                supported_groups.append(supported_group.copy())

            supported_group.clear()
            supported_group_border.clear()

    while nodes_to_process or nodes_to_process_with_next_group:
        if not nodes_to_process:
            close_group()
            nodes_to_process = nodes_to_process_with_next_group
            nodes_to_process_with_next_group = deque()
            continue

        node = nodes_to_process.popleft()

        is_op_supported = supported_ops_checker.is_op_supported(node)
        is_input_shape_supported = not require_fixed_input_sizes or \
                                   all(is_fixed_size_tensor(value_info[i]) for i in node.input)
        is_node_supported = is_op_supported and is_input_shape_supported

        if not is_node_supported:
            if node in supported_group_border:
                # an unsupported node on the border will be processed after the current partition node group
                # so skip any additional processing/counting here
                nodes_to_process_with_next_group.append(node)
                continue

            if not is_op_supported:
                unsupported_ops.add(f'{node.domain if node.domain else "ai.onnx"}:{node.op_type}')
                num_unsupported_nodes_due_to_op += 1
            else:
                num_unsupported_nodes_due_to_dynamic_input += 1

        if is_node_supported:
            num_supported_nodes += 1

            # add node to the partition node group
            supported_group.append(node)

            # remove node from the border and add its outputs to the border
            if node in supported_group_border:
                supported_group_border.remove(node)

            # for each consumer node add to supported_group_border
            if node in node_to_consumers:
                for consumer in node_to_consumers[node]:
                    supported_group_border.add(consumer)

        # adjust in-degrees of the node outputs and add any new nodes to process
        if node in node_to_consumers:
            for consumer in node_to_consumers[node]:
                consumer_node_in_degree = in_degree[consumer]
                consumer_node_in_degree -= 1
                if consumer_node_in_degree == 0:
                    nodes_to_process.append(consumer)

                in_degree[consumer] = consumer_node_in_degree

    close_group()

    num_nodes = len(model.graph.node)
    num_partitions = len(supported_groups)
    logger.info(f'{num_partitions} partitions with a total of {num_supported_nodes}/{num_nodes} '
                'nodes can be handled by this EP.')
    if supported_groups:
        logger.info(f'Partition sizes: {", ".join([str(len(partition)) for partition in supported_groups])}')
        logger.info(f'Unsupported nodes due to operator=%s', num_unsupported_nodes_due_to_op)
        if num_unsupported_nodes_due_to_dynamic_input:
            logger.info('Unsupported nodes due to input having a dynamic shape=%d',
                        num_unsupported_nodes_due_to_dynamic_input)

    if logger.getEffectiveLevel() >= logging.DEBUG:
        # Enable this manually if you need to look at specific partitions.
        # for group in supported_groups:
        #     logger.debug(f'Nodes in group: {",".join([f"{node.name}:{node.op_type}" for node in group])}')
        if unsupported_ops:
            logger.debug(f'Unsupported ops: {",".join(sorted(unsupported_ops))}')
        caveats = supported_ops_checker.get_caveats()
        if caveats:
            indent = ' ' * (FUNC_NAME_WIDTH + 5)
            logger.debug('Caveats that have not been checked and may result in a node not being supported:  '
                         f'{"".join([os.linesep + indent + caveat for caveat in caveats])}')

    pct_nodes_using_ep = num_supported_nodes / num_nodes
    if num_partitions == 0:
        logger.info(f"{ep_name} can not run any nodes in this model.")
    elif num_partitions == 1:
        if pct_nodes_using_ep > 50:
            logger.info(f"{ep_name} should work well for this model.")
        else:
            logger.info(f"{ep_name} may work well for this model, however the majority of nodes will use the CPU EP. "
                        "Performance testing is required to validate.")
    elif num_partitions == 2:
        logger.info(f"{ep_name} can be considered for this model. Performance testing is required to validate.")
    else:
        logger.info(f"{ep_name} is not recommended with this model as the number of partitions will most likely "
                    "result in poor performance.")


def replace_symbolic_dim_value(graph, **kwargs):
    '''
    Iterate all values in the model, replacing dim_param in a tensor shape with the provided value.
    :param model: ModelProto to update
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
                            dim_param = dim.dim_param  # save before Clear
                            dim.Clear()
                            dim.dim_value = value

    update_dim_values(graph.input)
    update_dim_values(graph.output)
    update_dim_values(graph.value_info)


def iterate_graph_per_node_func(graph, per_node_func, **func_args):

    for node in graph.node:
        per_node_func(node, **func_args)
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                iterate_graph_per_node_func(attr.g, per_node_func, **func_args)


def iterate_graph_per_graph_func(graph, per_graph_func, **func_args):

    per_graph_func(graph, **func_args)

    for node in graph.node:
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                iterate_graph_per_graph_func(attr.g, per_graph_func, **func_args)


def check_nnapi_partitions(model, value_info=None):
    logger.info(f'Checking NNAPI EP partitions...')
    supported_ops = SupportedOpsChecker(r'D:\src\github\ort\tools\ci_build\github\android\nnapi_supported_ops.md')
    check_partitioning(model, supported_ops, "NNAPI", value_info is not None, value_info)


def check_coreml_partitions(model, value_info=None):
    logger.info(f'Checking CoreML EP partitions...')
    supported_ops = SupportedOpsChecker(r'D:\src\github\ort\tools\ci_build\github\apple\coreml_supported_ops.md')
    check_partitioning(model, supported_ops, "CoreML", value_info is not None, value_info)


def check_shapes(model):
    # it's OK if the input is dynamically sized and we do a Resize early to a fixed size.
    # it's not good if lots of ops have dynamic inputs

    num_fixed_values = 0
    num_dynamic_values = 0

    dynamic_inputs = []
    for i in model.graph.input:
        if not is_fixed_size_tensor(i):
            dynamic_inputs.append(i)
            # split/join to remove repeated whitespace and newlines from str(i)
            logger.info(f"Input is not a fixed size tensor: {' '.join(str(i).split())}")
            num_dynamic_values += 1
        else:
            num_fixed_values += 1

    dynamic_outputs = []
    for o in model.graph.output:
        if not is_fixed_size_tensor(o):
            dynamic_outputs.append(o)
            logger.info(f"Output is not a fixed size tensor: {' '.join(str(o).split())}")
            num_dynamic_values += 1
        else:
            num_fixed_values += 1

    for vi in model.graph.value_info:
        if is_fixed_size_tensor(vi):
            num_fixed_values += 1
        else:
            num_dynamic_values += 1

    logger.info(f"Num values with fixed shape={num_fixed_values}. Num values with dynamic shape={num_dynamic_values}")

    if dynamic_inputs:
        if dynamic_outputs:
            logger.info("Model has dynamic inputs and outputs. "
                        "Consider re-exporting model with fixed sizes if NNAPI or CoreML can be used with this model.")
        else:
            logger.info("Model has dynamically sized inputs but fixed sized outputs. Depending on where ")

    return num_dynamic_values > 1


def make_dim_param_fixed(model: onnx.ModelProto, param_name: str, value: int):
    iterate_graph_per_graph_func(model.graph, replace_symbolic_dim_value, dim_param=param_name, value=value)


def checker(model_path):
    logger.info(f'Performing checks for {model_path}')
    model = onnx.load(model_path)
    model_with_shape_info = onnx.shape_inference.infer_shapes(model)

    # create lookup map for efficiency
    value_to_shape = {}
    for v in model_with_shape_info.graph.input:
        value_to_shape[v.name] = v
    for v in model_with_shape_info.graph.output:
        value_to_shape[v.name] = v
    for v in model_with_shape_info.graph.value_info:
        value_to_shape[v.name] = v

    has_dynamic_shapes = check_shapes(model_with_shape_info)

    if has_dynamic_shapes:
        make_dim_param_fixed(model_with_shape_info, "unk__610", 1)
        has_dynamic_shapes = check_shapes(model_with_shape_info)
        print(f'New value for has_dynamic_shapes={has_dynamic_shapes}')

    logger.info("Checking NNAPI and CoreML")
    check_nnapi_partitions(model_with_shape_info)
    check_coreml_partitions(model_with_shape_info)

    logger.info("Checking NNAPI and CoreML with fixed shapes being required...")
    check_nnapi_partitions(model_with_shape_info, value_to_shape)
    check_coreml_partitions(model_with_shape_info, value_to_shape)

    logger.info('---------------\n')


def get_model():
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    # model.eval()
    qmodel = torchvision.models.quantization.mobilenet_v3_large(pretrained=True, progress=True, quantize=True)
    qmodel.eval()

    return qmodel


def run_with_pt(model, image_filename):
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(image_filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

        torch.onnx.export(model, input_batch, 'mobilenet_v3_large_quant.onnx',
                          input_names=['image'], output_names=['scores'],
                          opset_version=13,
                          do_constant_folding=False,
                          training=False,
                          export_params=True,
                          keep_initializers_as_inputs=False
                          )

    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    return probabilities


def download_file(url, filename):
    import urllib
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)


def download_data():
    # download_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    download_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")


def test_run():
    qmodel = get_model()
    image = r'D:\mlperf\imagenet2012\val\ILSVRC2012_val_00000001.JPEG'

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    probabilities = run_with_pt(qmodel, image)

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


# download_data()
# test_run()

# use a hash of the object id for NodeProto.
# we need this for the partitioning checker where we keep maps with nodes as the key.
onnx.NodeProto.__hash__ = lambda self: id(self)

logger.setLevel(logging.DEBUG)

checker(r'D:\MobileBuildPackageModels\Converted\IndividualModels\resnet50_v1\resnet50_v1.onnx')
# checker(r'C:\Users\scmckay\Downloads\mlperf_models_202103\mobile\mobilenet_edgetpu\mobilenet_edgetpu_224_1.0_float.onnx')
# checker(r'C:\Users\scmckay\Downloads\mlperf_models_202103\mobile\mobilenet_edgetpu\mobilenet_edgetpu_224_1.0_float-int8.onnx')
# checker(r'C:\Users\scmckay\Downloads\mlperf_models_202103\mobile\mobilenet_edgetpu\mobilenet_edgetpu_224_1.0-qdq.onnx')

