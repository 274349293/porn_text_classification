#coding:utf-8
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os

def freeze_graph(model_dir, output_graph_pb, output_node_names):
    '''
    对于freeze操作,我们需要定义输出结点的名字.因为网络其实是比较复杂的,定义了输出结点的名字,
    那么freeze的时候就只把输出该结点所需要的子图都固化下来,其他无关的就舍弃掉.
    因为我们freeze模型的目的是接下来做预测.所以,一般情况下,output_node_names就是我们预测的目标.
    '''
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if not ckpt or not ckpt.model_checkpoint_path:
        raise Exception('model_path error:%s' % model_dir)

    ckpt_file = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
    meta_file = ckpt_file + '.meta'
    saver = tf.train.import_meta_graph(meta_file, clear_devices=True)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(','))

        pb_file = os.path.join(model_dir, output_graph_pb)
        with tf.gfile.GFile(pb_file, 'wb') as wf:
            wf.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

def load_graph(freeze_graph_file, op_prefix_name='', is_debug=False):
    with tf.gfile.GFile(freeze_graph_file, 'rb') as rf:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(rf.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=op_prefix_name,
            op_dict=None,
            producer_op_list=None)

    if is_debug:
        for op in graph.get_operations():
            print(op.name, op.values())
    return graph

def get_graph_op(graph, name):
    return graph.get_operation_by_name(name).outputs[0]

if __name__ == '__main__':
    output_node_names = 'output/pred_proba'
    freeze_graph(r"/root/text_classifier/word_cnn/wordcnn/bin/model/occupation", "occupation.pb", output_node_names)