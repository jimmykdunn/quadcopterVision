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
        graphInputs = ['inputs/b_images'] # ['inputs/b_images','inputs/b_masks']
        graphOutputs = ['heatmaps/b_heatmaps']
        
        names = [i.name for i in sess.graph.get_operations()]
                    
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
        #print("Writing graph in protobuf txt format")
        #tf.train.write_graph(
        #    graph_or_graph_def=sess.graph_def, 
        #    logdir=directory, name=pbtxt_filename, as_text=True)
        
        # Write the .pb file
        print("Freezing the graph in protobuf format")
        #from tensorflow.python.tools import freeze_graph
        #freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='',
        #    input_binary=False, input_checkpoint=ckptFile, 
        #    output_node_names=graphOutputs[0], 
        #    restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', 
        #    output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
        
        
        #graph = tf.get_default_graph()
        #input_graph_def = graph.as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, graphOutputs)

        with tf.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
            
            
        print("Writing frozen graph definition in protobuf txt format")
        tf.train.write_graph(
            graph_or_graph_def=output_graph_def, 
            logdir=directory, name=pbtxt_filename, as_text=True)
        
        
    
    print("Initial optimization")
    with tf.gfile.FastGFile(pb_filepath,'rb') as in_f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(in_f.read())
        # Downsize the graph keeping only what we need for forward passes            
        optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
            graph_def, graphInputs, graphOutputs, tf.float32.as_datatype_enum)
 
    '''
        names = [i.name for node in optimized_graph_def.node]
        
        # OPTIMIZED VERSIONS
        # This saves the variables (weights) to a ".pb" file
        with tf.gfile.FastGFile(pb_opt_filepath, 'wb') as ff:
            ff.write(optimized_graph_def.SerializeToString())
            
        tf.train.write_graph(
            graph_or_graph_def=optimized_graph_def, 
            logdir=directory, name=baseName + '_opt.pbtxt', as_text=True)
    

    '''
    
    print("FULL NODE/VARIABLE LIST")
    for node in graph_def.node:
        print(node.name + " is a " + node.op)
        
    #@@@
    #!!!WE CAN ARTIFICAILLY CHOP OUT NODES HERE BY SELECTING EARLIER OUTPUTS
    #THAN HEATMAP. EVENTUALLY WE SHOULD FIND A SPECIFIC NODE THAT IS CAUSING
    #THE error: (-2:Unspecified error) Const kernel input not found in function 'cv::dnn::experimental_dnn_v5::`anonymous-namespace'::TFImporter::getConstBlob'
    #ERROR !!!
    #graphOutputs = ['heatmaps/b_heatmaps'] # original output
    #graphOutputs = ['heatmaps/firstConv/Relu'] # ok to here (cv2 can read & eval)
    graphOutputs = ['heatmaps/secondPool/MaxPool'] # 
    
    inference_graph = tf.graph_util.extract_sub_graph(graph_def, graphOutputs)
  
    # This saves the variables (weights) to a ".pb" file
    with tf.gfile.FastGFile(pb_opt_filepath, 'wb') as ff:
        ff.write(inference_graph.SerializeToString())
    
    
    print("Trimming to selected levels and saving off")
    # ok to here
    trim_to_output(graph_def,['heatmaps/secondPool/MaxPool'],os.path.join(directory, baseName+'_mp2.pb'))
    # fails at tensorflowNet.forward()
    trim_to_output(graph_def,['heatmaps/secondUpconv/Shape'],os.path.join(directory, baseName+'_up2Shape.pb'))
    # fails at tensorflowNet.forward()
    trim_to_output(graph_def,['heatmaps/secondUpconv/strided_slice'],os.path.join(directory, baseName+'_up2SS.pb'))
    # fails at tensorflowNet.forward()
    trim_to_output(graph_def,['heatmaps/secondUpconv/stack'],os.path.join(directory, baseName+'_up2stack.pb'))
    
    # fails at cv2.dnn.readNet(modelPath+'.pb') (original fail point)
    # THIS IS WHERE cv2 BARFS! At conv2d_transpose!
    #!!!The error we are seeing here indicates that the prior layer ([heatmaps/secondUpconv/stack])
    #IS NOT A CONSTANT FOR SOME REASON!!!
    trim_to_output(graph_def,['heatmaps/secondUpconv/conv2d_transpose'],os.path.join(directory, baseName+'_up2conv2t.pb'))
    
    # fails at cv2.dnn.readNet(modelPath+'.pb') (original fail point)
    trim_to_output(graph_def,['heatmaps/secondUpconv/Relu'],os.path.join(directory, baseName+'_up2relu.pb'))
    # fails at cv2.dnn.readNet(modelPath+'.pb') (original fail point)
    trim_to_output(graph_def,['heatmaps/firstUpconv/strided_slice'],os.path.join(directory, baseName+'_up1SS.pb'))
    #
    trim_to_output(graph_def,['heatmaps/b_heatmaps/shape'],os.path.join(directory, baseName+'_hmShape.pb'))
    # Fail here
    trim_to_output(graph_def,['heatmaps/b_heatmaps'],os.path.join(directory, baseName+'_hm.pb'))
    
    #@@@
            
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