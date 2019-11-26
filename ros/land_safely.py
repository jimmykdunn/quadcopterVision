#!/usr/bin/env python
import rospy
import socket
from geometry_msgs.msg import PoseStamped

# Define global variables
global current_pose


# Function to call every time that we receive the actual pose from the 
# mavros (motive camera system ROS topic) (/mavros/mocap/pose)
def pos_sub_callback(pose_sub_data):
    global local_position_pub
    global goal_pose
    global current_pose
    
    current_pose = pose_sub_data # update current pose
    
    # Initialize goal pose to be the current pose
    goal_pose = current_pose
    
    # Update the z (vertical) part of the goal pose to be zero, leaving
    # all else untouched.
    goal_pose.pose.position.z = 0.0
        
    # Publish "land safely" command to the setpoint topic
    local_position_pub.publish(goal_pose)
    
def main():
    global local_position_pub
    global goal_pose
    global current_pose

    hostname = socket.gethostname() # name of the quad (i.e. "quad_delorian")

    # Create a node
    rospy.init_node(hostname+'_land_safely', anonymous='True')
    
    # Subscribe to the pose and publish to the setpoint position
    local_position_subscribe = rospy.Subscriber(hostname+'/mavros/mocap/pose', PoseStamped, pos_sub_callback)
    local_position_pub = rospy.Publisher(hostname+'/mavros/setpoint_position/local', PoseStamped, queue_size = 1)
    
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
