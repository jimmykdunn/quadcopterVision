# -*- coding: utf-8 -*-
"""
tf_utilities.py
DESCRIPTION:
    Tensorflow utility function wrappers

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""

import tensorflow as tf
import os
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.tools import freeze_graph

"""
ckpt_to_protobuf()
DESCRIPTION:
    Converts a tensorflow saved checkpoint file into a protobuf file capable
    of being deployed on the quadcopters without needing to install tensorflow,
    just cv2.
    
INPUTS: 
    ckptFile: file to the checkpoint that we want to convert to a protobuf.
    
OUTPUTS: 
    Protobuf (.pb and .pbtxt) files with the same name as ckptFile

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
REFERENCES:
    This function is based on
    https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/
"""
def ckpt_to_protobuf(ckptFile):
    baseName,ext = os.path.splitext(os.path.basename(ckptFile))
    directory = os.path.dirname(ckptFile)
    
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(ckptFile)) # the ENTIRE session is now in saver
        saver.restore(sess,ckptFile)
        graphInputs = ['inputs/b_images'] # ['inputs/b_images','inputs/b_masks']
        graphOutputs = ['heatmaps/b_heatmaps']
        
        #names = [i.name for i in sess.graph.get_operations()]
                    
        # Setup protobuf filenames
        pbtxt_filename = baseName+'.pbtxt'
        pbtxt_filepath = os.path.join(directory, baseName + '.pbtxt')
        pb_filepath = os.path.join(directory, baseName + '.pb')
        pb_opt_filepath = os.path.join(directory, baseName + '_opt.pb')
        pbtxt_opt_filepath = os.path.join(directory, baseName + '_opt.pbtxt')
        
        '''
        # Freeze graph. This saves all the actual weights to the file
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_node_names = ['heatmaps/b_heatmaps'] #['cnn/output']
        frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess, input_graph_def, output_node_names)
        '''
        
        # This will only save the graph; the variables (weights) will 
        # not be saved. Writes to the ".pbtxt" file.
        tf.train.write_graph(
            graph_or_graph_def=sess.graph_def, 
            logdir=directory, name=pbtxt_filename, as_text=True)
        
        freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='',
            input_binary=False, input_checkpoint=ckptFile, 
            output_node_names=graphOutputs[0], 
            restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', 
            output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
        
        
        
    
    with tf.gfile.GFile(pb_filepath,'rb') as in_f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(in_f.read())
        # Downsize the graph keeping only what we need for forward passes            
        optimized_graph = optimize_for_inference_lib.optimize_for_inference(
            graph_def, graphInputs, graphOutputs, tf.float32.as_datatype_enum)
    
        # OPTIMIZED VERSIONS
        # This saves the variables (weights) to a ".pb" file
        with tf.gfile.FastGFile(pb_opt_filepath, 'wb') as ff:
            ff.write(optimized_graph.SerializeToString())
            
        tf.train.write_graph(
            graph_or_graph_def=optimized_graph.as_graph_def(), 
            logdir=directory, name=baseName + '_opt.pbtxt', as_text=True)
    

    '''
    #@@@
    inference_graph = tf.graph_util.extract_sub_graph(input_graph_def, output_node_names)

    for node in inference_graph.node:
        print(node.name + " is a " + node.op)
        if hasattr(node.attr, 'value'):
            stophere=1
    #@@@
    '''

            
# end ckpt_to_protobuf
            
# Run with defaults if at highest level
if __name__ == "__main__":
    
    ckpt_to_protobuf(os.path.join('homebrew_hourglass_nn_save','model_at10.ckpt'))