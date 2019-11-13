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
    
    sess = tf.Session()
    with sess.as_default():
        print("Importing graph from checkpoint file")
        saver = tf.train.import_meta_graph("{}.meta".format(ckptFile)) # the ENTIRE session is now in saver
        print("Restoring session from checkpoint file")
        saver.restore(sess,ckptFile)
        graphOutputs = ['heatmaps/b_heatmaps']
        
        #names = [i.name for i in sess.graph.get_operations()]
                    
        # Setup protobuf filenames
        pb_filepath = os.path.join(directory, baseName + '_full.pb')
        pbtxt_filename = baseName+'_full.pbtxt'
        pb_trimmed_filebase = os.path.join(directory, baseName + '_trim')
        
        # Write the .pb file
        print("Freezing the graph in protobuf format")
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, graphOutputs)
        with tf.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
            
            
        print("Writing frozen graph definition in protobuf txt format")
        tf.train.write_graph(graph_or_graph_def=output_graph_def, 
            logdir=directory, name=pbtxt_filename, as_text=True)
        
        print("Saved NN graph to " + pb_filepath)
        print("Saved NN graph (text version) to " + pbtxt_filename)
        
    '''
    print("FULL NODE/VARIABLE LIST")
    for node in output_graph_def.node:
        print(node.name + " is a " + node.op)
    '''    
    
    print("Trimming out variables from the graph that aren't used for " + \
          "the forward pass")
    trim_to_output(output_graph_def, graphOutputs, pb_trimmed_filebase)
            
# end ckpt_to_protobuf
    
"""
trim_to_output()
DESCRIPTION:
    Uses extract_sub_graph from tensorflow library to trim away all parts
    of the graph not involved in calculating graphOutputs. Saves the resulting
    graph to outname as both a .pb and .pbtxt.
    
INPUTS: 
    graphDef: Tensorflow graph to trim
    graphOutputs: list of all output nodes desired from the forward pass
    outname: directory + name (without extension) to save the .pb and .pbtxt to
    
OUTPUTS: 
    Protobuf (.pb and .pbtxt) files written to outname

EXAMPLE:
    trim_to_ouptut(sess.graph_def, ['heatmaps/b_heatmap'], 'somedir/modelTrim')

INFO:
    Author: James Dunn, Boston University
    Thesis work for MS degree in ECE
    Advisor: Dr. Roberto Tron
    Email: jkdunn@bu.edu
    Date: October 2019
"""
def trim_to_output(graphDef, graphOutputs, outname):
    directory = os.path.dirname(outname)
    filename = os.path.basename(outname)
    
    # Trim the graph
    inference_graph = tf.graph_util.extract_sub_graph(graphDef, graphOutputs)
    
    # This saves the variables (weights) to a ".pb" file
    with tf.gfile.FastGFile(outname+'.pb', 'wb') as ff:
        ff.write(inference_graph.SerializeToString())
        
    # This saves the graph def to a pbtxt file
    tf.train.write_graph(graph_or_graph_def=inference_graph, 
        logdir=directory, name=filename+'.pbtxt', as_text=True)
    
    print("Saved trimmed NN graph to " + outname + '.pb')
    print("Saved trimmed NN graph (text version) to " + outname + '.pbtxt')
# end trim_to_output
            
        
# Run with defaults if at highest level
if __name__ == "__main__":    
    ckpt_to_protobuf(os.path.join('savedNetworks','mirror60k_sW00p00_1M00p00_2M00p00','model_at16000.ckpt'))
