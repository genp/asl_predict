'''

Sets up global variables for ASL Predict.

'''
import os,sys,socket
aslroot = os.getenv('ASLROOT')
if not kairoot:
    aslroot = os.path.dirname(os.path.abspath(__file__))
sys.path.append(kairoot)

cafferoot = os.getenv('CAFFEROOT')
if not cafferoot:
    cafferoot = '~/caffe'
sys.path.append(cafferoot)
DEVELOPMENT = False

# set logging level to 2 to suppress caffe output
os.environ['GLOG_minloglevel'] = '2'
USE_GPU = False
instance_type = ec2_metadata('instance-type')
EC2 = instance_type != ''
if instance_type.startswith("g"):
    print "Using GPU"
    USE_GPU = True
    GPU_DEVICE_ID = 0
