import unittest
from onnx import helper
from onnx import TensorProto

from .onnx_model_utils import get_producer_consumer_maps


class TestGetProducerConsumerMaps(unittest.TestCase):

    @staticmethod
    def _create_model():
        body = helper.make_graph(
            [
                # shadow a1 in main graph.
                # LoopAdd_SubgraphOutput should be linked to this and a1 should not be an implicit input
                helper.make_node("Add", ["loop_state_in", "iter"], ["a1"], "LoopAdd_Shadows"),

                # main_graph_initializer should be handled (implicit input but no producer node)
                # graph input 'x' from main graph should also be handled
                helper.make_node("Add", ["main_graph_initializer", "x"], ["a2"], "LoopAdd_OuterScopeInitializer"),

                # implicit input should be handled - 'z' can be accessed from outside scope
                # Add2 in main graph should be implicit input of the Loop node
                helper.make_node("Add", ["z", "a1"], ["a3"], "LoopAdd_ImplicitInput"),

                # create subgraph output
                helper.make_node("Add", ["a2", "a3"], ["loop_state_out"], "LoopAdd_SubgraphOutput"),
            ],
            "Loop_body",
            [
                helper.make_tensor_value_info('iter', TensorProto.INT64, [1]),
                helper.make_tensor_value_info('cond', TensorProto.BOOL, [1]),
                helper.make_tensor_value_info('loop_state_in', TensorProto.FLOAT, [1])
            ],
            [
                helper.make_tensor_value_info('cond', TensorProto.BOOL, [1]),
                helper.make_tensor_value_info('loop_state_out', TensorProto.FLOAT, [1]),
            ],
            [
            ]
        )

        # Create the main graph
        graph_proto = helper.make_graph(
            [
                # create 'a1' which is shadowed in the subgraph. node should not be joined to Loop1
                helper.make_node("Add", ["x", "y"], ["a1"], "Add1"),
                # create 'z' which is an implicit input to subgraph. node should be joined to Loop1
                helper.make_node("Add", ["a1", "main_graph_initializer"], ["z"], "Add2"),
                # rename 'z' to use as explicit input to Loop
                helper.make_node("Identity", ["z"], ["state_var_in"], "RenameZ"),
                helper.make_node("Loop", ["max_trip_count", "keep_going", "state_var_in"], ["state_var_out"], "Loop1",
                                 body=body),
                helper.make_node("Sub", ["a1", "state_var_out"], ["graph_output"], "sub_1")
            ],
            "Main_graph",
            [
                helper.make_tensor_value_info('x', TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info('y', TensorProto.FLOAT, [1]),
            ],
            [
                helper.make_tensor_value_info('graph_output', TensorProto.FLOAT, [1]),
            ],
            [
                helper.make_tensor('max_trip_count', TensorProto.INT64, [1], [2]),
                helper.make_tensor('main_graph_initializer', TensorProto.FLOAT, [1], [1.]),
                helper.make_tensor('keep_going', TensorProto.BOOL, [1], [True]),
            ]
        )

        return helper.make_model(graph_proto)

    def test_model_with_subgraph(self):
        '''
        Test a manually created model that has a subgraph and implicit inputs of all possible types.
        '''

        model = self._create_model()
        node_to_producers, node_to_consumers = get_producer_consumer_maps(model.graph)

        main_graph_add_create_a1 = model.graph.node[0]
        main_graph_add_create_z = model.graph.node[1]
        main_graph_rename_z = model.graph.node[2]
        main_graph_loop = model.graph.node[3]
        main_graph_sub = model.graph.node[4]

        subgraph = main_graph_loop.attribute[0].g
        loop_add_shadow = subgraph.node[0]
        loop_add_outer_scope_init = subgraph.node[1]
        loop_add_implicit_input = subgraph.node[2]
        loop_add_subgraph_output = subgraph.node[3]

        def node_name(node):
            return f'{node.name}:{node.op_type}'

        def check_linked(producer, consumer):
            self.assertTrue(producer in node_to_producers[consumer],
                            f'{node_name(producer)} not in producers for {node_name(consumer)}')
            self.assertTrue(consumer in node_to_consumers[producer],
                            f'{node_name(consumer)} not in consumers for {node_name(producer)}')

        def check_not_linked(producer, consumer):
            self.assertFalse(producer in node_to_producers[consumer],
                             f'{node_name(producer)} in producers for {node_name(consumer)}')
            self.assertFalse(consumer in node_to_consumers[producer],
                             f'{node_name(consumer)} not in consumers for {node_name(producer)}')

        check_linked(main_graph_add_create_a1, main_graph_add_create_z)
        # a1 in main graph shouldn't be implicit input to loop as it is shadowed
        check_not_linked(main_graph_add_create_a1, main_graph_loop)
        # z is implicit input
        check_linked(main_graph_add_create_z, main_graph_loop)
        check_linked(main_graph_rename_z, main_graph_loop)
        check_linked(main_graph_loop, main_graph_sub)

        # check subgraph
        check_linked(loop_add_shadow, loop_add_implicit_input)
        check_linked(loop_add_outer_scope_init, loop_add_subgraph_output)
        check_linked(loop_add_implicit_input, loop_add_subgraph_output)
