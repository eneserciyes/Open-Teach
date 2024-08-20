import pyrealsense2 as rs


for dev in rs.context().query_devices():
    print("Resetting device.")
    dev.hardware_reset()
