import os
import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge, CvBridgeError
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, BoolStamped, Twist2DStamped
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

class Duckiebot(DTROS):
    def __init__(self,node_name):
        
        # initialize the DTROS parent class
        super(Duckiebot, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # Variables
        self.img_count = 0
        self.name = os.environ['VEHICLE_NAME']
        self.right_encoder_tics = 0
        self.left_encoder_tics = 0
        self.bridge = CvBridge()

        # Setup Parameters
        self.rez_h = self.setupParam(f"/{self.name}/camera_node/res_h", 308)
        self.rez_w = self.setupParam(f"/{self.name}/camera_node/res_w", 408)
        
        # To dynamically change the param bash shell into one of the docker containers of the duckie 
        # and type rosparam set /emergency_stop 2
        self.emergency_stop = self.setupParam("/emergency_stop",1)
        # Always get the param before using it
        self.emergency_stop = rospy.get_param("/emergency_stop")

        # Publications
        self.pub_wheel = rospy.Publisher(f'/{self.name}/wheels_driver_node/wheels_cmd',WheelsCmdStamped, queue_size=1)
        self.pub_car = rospy.Publisher(f'/{self.name}/car_cmd_switch_node/cmd',Twist2DStamped, queue_size=1)

        # Subscriptions
        self.sub_right_encoder = rospy.Subscriber(f"/{self.name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.right_encoder, queue_size=1)
        self.sub_left_encoder =  rospy.Subscriber(f"/{self.name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.left_encoder, queue_size=1)
    
    def right_encoder(self,data):
        """ Callback function to save right encoder tics. """
        self.right_encoder_tics = data.data
        #rospy.loginfo("I heard that the right wheel is %s tics and resolution of %s", str(data.data),str(data.resolution))

    def left_encoder(self,data):
        """ Callback function to save left encoder tics. """
        self.left_encoder_tics = data.data
        #rospy.loginfo("I heard that the left wheel is %s tics and resolution of %s", str(data.data),str(data.resolution))

    def connect_camera(self,callback):
        """
        Initialize and connect to the camera. \n
        Arguments:
            callback: Function where you process the image.
        """
        self.sub_image = rospy.Subscriber(f"/{self.name}/camera_node/image/compressed", CompressedImage, callback, queue_size=1)

    def decode_image(self,image): 
        """
        Decode the compressed image. \n
        Returns:
            decoded_img: Image in format BGR if is decoded else returns nothing.
        """ 
        try:
            self.img_count +=1
            decoded_img = self.bridge.compressed_imgmsg_to_cv2(image)
        except CvBridgeError as e:
            print(e)
            return None
        return decoded_img

    def save_image(self,location,image):
        """
        Save image for debugging to a desired location. If the location doesn't exist it creates it.\n
        Arguments:
            location: Name of the folder.
            image: The image to save to the folder.
        """ 
        if not os.path.isdir(f"/data/{location}/"):
            os.mkdir(f"/data/{location}/")
        cv2.imwrite(f"/data/{location}/"+str(self.img_count)+".jpg",image)
        
    def publish_car_cmd(self,velocity,omega):
        """
        Publish target velocity and heading for the car.\n
        Arguments:
            velocity: Accepts range of (-1,1).
            omega: Accepts  range of (-8,8).
        """ 
        cmd_msg = Twist2DStamped()
        cmd_msg.v = velocity
        cmd_msg.omega = omega
        self.pub_car.publish(cmd_msg)
        rospy.loginfo("Published speed for car (%s) and direction (%s)", str(cmd_msg.v), str(cmd_msg.omega))

    def publish_wheel_cmd(self,velocity_left,velocity_right):
        """
        Publish target velocity for each wheel.\n
        Arguments:
            velocity_left:  Accepts range of (-1,1).
            velocity_right: Accepts  range of (-1,1).
        """ 
        cmd_msg = WheelsCmdStamped()
        cmd_msg.vel_left = velocity_left
        cmd_msg.vel_right = velocity_right
        self.pub_wheel.publish(cmd_msg)
        rospy.loginfo("Published speed for left (%s) and right (%s) wheel", str(cmd_msg.vel_left), str(cmd_msg.vel_right))
    
    def setupParam(self,param_name,value):
        """
        Set parameter for the robot.\n
        Arguments:
            param_name:  Name of the parameter.
            value: Value to set.
        """ 
        rospy.set_param(param_name,value)
        rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value
