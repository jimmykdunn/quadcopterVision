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

# Define global variables
global pub_Image
global pub_BoxesImage
global pub_Boxes
global face_cascade
global bridge
global current_pose
global boxes_msg

def callback(image_msg):
    global pub_Image
    global pub_BoxesImage
    global pub_Boxes
    global face_cascade
    global bridge
    global boxes_msg
    global faces
    global local_position_pub
    global goal_pose
    # Convert ros image into a cv2 image
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Find the faces and draw a rectangle around them
    faces = face_cascade.detectMultiScale(cv_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv_img = cv2.rectangle(cv_img, (x,y), (x+w,y+h), (255,0,0), 2)
        # For each image, we also want to publish the croped image
        crop_img = cv_img[y:y+h, x:x+w]
        crop_msg = bridge.cv2_to_imgmsg(crop_img, encoding="bgr8")
        # And the region of interest information
        pub_BoxesImage.publish(crop_msg)
        boxes_msg = RegionOfInterest()
        boxes_msg.x_offset = x
        boxes_msg.y_offset = y
        boxes_msg.height = h
        boxes_msg.width = w
        boxes_msg.do_rectify = False
        pub_Boxes.publish(boxes_msg)
    # Publish the original image with the rectangles
    mod_img = bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
    pub_Image.publish(mod_img)
    
    #Follow face
    pose_msg = PoseStamped()
  
    fig_x = image_msg.width/2
    fig_y = image_msg.height/2
    #kx = .001
    #ky = .001
    kx = .003
    ky = .003
    #xmax_m = 0.70
    #xmin_m = -1.0
    ymax_m = -3.00
    ymin_m = -5.00
    zmin_m = 0.5
    zmax_m = 1.5
    
    if len(faces) > 0:
        face_x = boxes_msg.x_offset + (boxes_msg.width/2)
        face_y = boxes_msg.y_offset + (boxes_msg.height/2)
        x_error = fig_x - face_x
        y_error = fig_y - face_y
        pose_msg.pose.position.y = current_pose.pose.position.y + (x_error*kx)
        pose_msg.pose.position.z = current_pose.pose.position.z + (y_error*ky)
    else:
	# No face to follow, remain stationary
        pose_msg = goal_pose
    if pose_msg.pose.position.y > ymax_m:
        pose_msg.pose.position.y = ymax_m
    if pose_msg.pose.position.y < ymin_m:
	pose_msg.pose.position.y = ymin_m
    if pose_msg.pose.position.z > zmax_m:
	pose_msg.pose.position.z = zmax_m
    if pose_msg.pose.position.z < zmin_m:
	pose_msg.pose.position.z = zmin_m

    pose_msg.pose.position.x = goal_pose.pose.position.x
    #pose_msg.pose.position.y = goal_pose.pose.position.y
    pose_msg.pose.orientation = goal_pose.pose.orientation
    local_position_pub.publish(goal_pose)

    # Update goal pose to be pose_msg
    goal_pose = pose_msg

def pos_sub_callback(pose_sub_data):
    global current_pose
    current_pose = pose_sub_data
    
def main():
    global pub_Image
    global pub_BoxesImage
    global pub_Boxes
    global face_cascade
    global bridge
    global current_pose
    global boxes_msg
    global faces
    global local_position_pub
    global goal_pose

    hostname = socket.gethostname()

    # Create a node
    rospy.init_node(hostname+'_face_dection', anonymous='True')
    # Setup the face detector and bridge
    face_cascade = cv2.CascadeClassifier('/opt/ros/kinetic/share/OpenCV-3.3.1-dev/haarcascades/haarcascade_frontalface_default.xml')
    bridge = CvBridge()
    # Create publishers and subscribers
    pub_Image = rospy.Publisher(hostname+'/face_detection/image', Image, queue_size=1)
    pub_BoxesImage = rospy.Publisher(hostname+'/face_detection/boxes_image', Image, queue_size=1)
    pub_Boxes = rospy.Publisher(hostname+'/face_detection/boxes', RegionOfInterest, queue_size=1)
    rospy.Subscriber(hostname+'/vidstream_node/image', Image, callback, queue_size=1, buff_size=2**18)
    local_position_subscribe = rospy.Subscriber(hostname+'/mavros/mocap/pose', PoseStamped, pos_sub_callback)
    local_position_pub = rospy.Publisher(hostname+'/mavros/setpoint_position/local', PoseStamped, queue_size = 1)
    current_pose = PoseStamped()
    goal_pose = PoseStamped()
    # Set default waypoint
    goal_pose.pose.position.x = 1.53
    goal_pose.pose.position.y = -4.00
    goal_pose.pose.position.z = 0.7
    goal_pose.pose.orientation.x = 0
    goal_pose.pose.orientation.z = 0
    goal_pose.pose.orientation.y = 0
    goal_pose.pose.orientation.w = 1

    # Keep program alive until we stop it
    rospy.spin()

if __name__ == "__main__":
    main()
