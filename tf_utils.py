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
    
    graph = tf.Graph()
    
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(ckptFile)) # the ENTIRE session is now in saver
            saver.restore(sess,ckptFile)
            
            #names = [i.name for i in sess.graph.get_operations()]
                        
            # Setup protobuf filenames
            pbtxt_filename = baseName+'.pbtxt'
            pbtxt_filepath = os.path.join(directory, pbtxt_filename)
            pb_filepath = os.path.join(directory, baseName + '.pb')
            
            # This will only save the graph but the variables (weights) will not be 
            # saved. This saves the ".pbtxt" file.
            tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir=directory, 
                name=pbtxt_filename, as_text=True)
        
            # Freeze graph. This saves all the actual weights to the file
            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()
            output_node_names = ['heatmaps/b_heatmaps'] #['cnn/output']
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess, input_graph_def, output_node_names)
    
            with tf.gfile.GFile(pb_filepath, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
    
    
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
    
    ckpt_to_protobuf(os.path.join('homebrew_hourglass_nn_save_GOOD','model_at1000.ckpt'))