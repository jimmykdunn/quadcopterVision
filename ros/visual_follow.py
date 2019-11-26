#!/usr/bin/env python
import rospy
import cv2
import socket
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
#from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import RegionOfInterest
from geometry_msgs.msg import PoseStamped

from videoUtilities import find_centerOfMass

# Define global variables
global pub_Image
global pub_BoxesImage
global pub_Boxes
global tensorflowNet
global bridge
global current_pose
global boxes_msg

# Function to call upon receipt of an image from the image node
def callback(image_msg):
    global pub_Image
    global pub_BoxesImage
    global pub_Boxes
    global tensorflowNet
    global bridge
    global boxes_msg
    global local_position_pub
    global goal_pose
    global current_pose
    
    # Adjustable parameters
    commandQuadFlag = False # set to true to actually steer quad
    heatmapThresh = 0.5 # threshold heatmap by this before centroid calculation
    nnFramesize = (64,48) # must match trained NN expectation
    scaleY = 0.1 # horizontal image fractional offset to meters conversion
    scaleZ = 0.1 # vertical image fractional offset to meters conversion
    
    # Pull image from topic and convert ros image into a cv2 image
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to appropriate size for neural network input
    nnFrame = cv2.resize(cv_gray,nnFramesize)
    nnFrame = nnFrame[:,:,0] * float(1.0/255.0)
    nnFrame = np.squeeze(nnFrame)
    
    # Execute a forward pass of the neural network on the frame to get a
    # heatmap of target likelihood
    tensorflowNet.setInput(nnFrame)
    heatmap = tensorflowNet.forward()
    heatmap = np.squeeze(heatmap)
    
    # Find the centroid
    heatmap = heatmap*255.0 # scale appropriately
    centroid = find_centerOfMass(heatmap, minThresh=heatmapThresh*255)
    
    # Pull the actual pose
    actual_pose = current_pose
    
    # Convert the centroid to a setpoint position
    centroid_frac = [(centroid[0] - nnFramesize[0])/nnFramesize[0], \
                     (centroid[1] - nnFramesize[1])/nnFramesize[1]]
    setpoint_x = actual_pose.pose.position.x # Don't move in x just yet
    setpoint_y = centroid_frac[0]*scaleY + actual_pose.pose.position.y
    setpoint_z = centroid_frac[1]*scaleZ + actual_pose.pose.position.z
    
    # Change setpoint position
    if commandQuadFlag:
        # DON'T ACTUALLY DO THIS UNTIL READY!
        # Use the new setpoint
        goal_pose.pose.position.x = setpoint_x
        goal_pose.pose.position.y = setpoint_y
        goal_pose.pose.position.z = setpoint_z
        goal_pose.pose.orientation = actual_pose.pose.orientation
        
    else:
        # Don't change the setpoint position - just copy the pose
        goal_pose.pose.position.x = actual_pose.pose.position.x
        goal_pose.pose.position.y = actual_pose.pose.position.y
        goal_pose.pose.position.z = actual_pose.pose.position.z
        goal_pose.pose.orientation = actual_pose.pose.orientation
    
    # Publish resized image as input to the NN
    mod_img = bridge.cv2_to_imgmsg(nnFrame, encoding="bgr8") # this may not be right...
    pub_Image.publish(mod_img)
    
    # Draw a little rectangle at the quad centroid and publish. Do full size rectangle later
    imagePlusBox = cv2.rectangle(nnFrame, (centroid[0]-1,centroid[1]-1), (centroid[0]+1,centroid[1]+1), (255,0,0), 2)
    boxes_img = bridge.cv2_to_imgmsg(imagePlusBox, encoding="bgr8")
    pub_BoxesImage.publish(boxes_img)
    
    # Publish the centroid box coordinates themselves
    boxes_msg = RegionOfInterest()
    boxes_msg.x_offset = centroid[0]-1
    boxes_msg.y_offset = centroid[1]-1
    boxes_msg.height = 2
    boxes_msg.width = 2
    boxes_msg.do_rectify = False
    pub_Boxes.publish(boxes_msg)
    
    # Publish the actual setpoint position
    # Make sure this is nice and smooth and what we want in stationary testing
    # before actually using it as a flight control. Can even have everything but
    # the actual command running without actually flying the quad.
    if commandQuadFlag:
        local_position_pub.publish(goal_pose)
            
    
# Update the current pose whenever the topic is written to
def pos_sub_callback(pose_sub_data):
    global current_pose
    current_pose = pose_sub_data
    
    
# Everything in the main function gets executed at startup, and any callbacks
# from subscribed topics will get called as messages are recieved.
def main():
    global pub_Image
    global pub_BoxesImage
    global pub_Boxes
    global tensorflowNet
    global bridge
    global current_pose
    global boxes_msg
    global local_position_pub
    global goal_pose

    # Hostname is (for example) "quad_delorian"
    hostname = socket.gethostname()

    # Create a node
    rospy.init_node(hostname+'_visual_follow', anonymous='True')
    
    # Setup the quad detector
    # Import the trained neural network
    print("Loading trained neural network from trainedQuadDetector.pb")
    tensorflowNet = cv2.dnn.readNetFromTensorflow('trainedQuadDetector.pb')
    print("Neural network sucessfully loaded")
    
    # Setup the bridge (ros-to-cv2 image converter)
    bridge = CvBridge()
    
    # Create publishers and subscribers
    pub_Image = rospy.Publisher(hostname+'/quad_tracking/image', Image, queue_size=1) # raw image
    pub_BoxesImage = rospy.Publisher(hostname+'/quad_tracking/boxes_image', Image, queue_size=1) # image with boxes
    pub_Boxes = rospy.Publisher(hostname+'/quad_tracking/boxes', RegionOfInterest, queue_size=1) # box definitions
    rospy.Subscriber(hostname+'/vidstream_node/image', Image, callback, queue_size=1, buff_size=2**18) # raw image read from camera stream node
    local_position_subscribe = rospy.Subscriber(hostname+'/mavros/mocap/pose', PoseStamped, pos_sub_callback) # current position of the quad
    local_position_pub = rospy.Publisher(hostname+'/mavros/setpoint_position/local', PoseStamped, queue_size = 1) # current setpoint of the quad
    
    # Initialize current and goal poses as the current default
    current_pose = PoseStamped()
    goal_pose = PoseStamped()
    
    # Set default waypoint (approximately the middle of the lab on the ground)
    goal_pose.pose.position.x = 1.53
    goal_pose.pose.position.y = -4.00
    goal_pose.pose.position.z = 0.0
    goal_pose.pose.orientation.x = 0
    goal_pose.pose.orientation.z = 0
    goal_pose.pose.orientation.y = 0
    goal_pose.pose.orientation.w = 1

    # Keep program alive until we stop it
    rospy.spin()

if __name__ == "__main__":
    main()
