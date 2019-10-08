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
        print("Importing graph from checkpoint file")
        saver = tf.train.import_meta_graph("{}.meta".format(ckptFile)) # the ENTIRE session is now in saver
        print("Restoring session from checkpoint file")
        saver.restore(sess,ckptFile)
        graphOutputs = ['heatmaps/b_heatmaps']
        
        #names = [i.name for i in sess.graph.get_operations()]
                    
        # Setup protobuf filenames
        pbtxt_filename = baseName+'.pbtxt'
        pb_filepath = os.path.join(directory, baseName + '.pb')
        #pb_opt_filepath = os.path.join(directory, baseName + '_opt.pb')
        
        # Write the .pb file
        print("Freezing the graph in protobuf format")
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, graphOutputs)
        with tf.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
            
            
        print("Writing frozen graph definition in protobuf txt format")
        tf.train.write_graph(
            graph_or_graph_def=output_graph_def, 
            logdir=directory, name=pbtxt_filename, as_text=True)
        
    '''
    print("FULL NODE/VARIABLE LIST")
    for node in graph_def.node:
        print(node.name + " is a " + node.op)

    # Artifically extract a subgraph
    inference_graph = tf.graph_util.extract_sub_graph(output_graph_def, graphOutputs)
  
    # This saves the variables (weights) to a ".pb" file
    with tf.gfile.FastGFile(pb_opt_filepath, 'wb') as ff:
        ff.write(inference_graph.SerializeToString())
    '''
            
# end ckpt_to_protobuf
    
def trim_to_output(graph_def, graphOutputs, outfile):
    inference_graph = tf.graph_util.extract_sub_graph(graph_def, graphOutputs)
    
    # This saves the variables (weights) to a ".pb" file
    with tf.gfile.FastGFile(outfile, 'wb') as ff:
        ff.write(inference_graph.SerializeToString())
# end trim_to_output
            
# Run with defaults if at highest level
if __name__ == "__main__":
    
    ckpt_to_protobuf(os.path.join('homebrew_hourglass_nn_save','model_at10.ckpt'))