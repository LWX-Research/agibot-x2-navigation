"""Motion control clients for the isolated X2 simulation ROS domain."""

import time

import rclpy
from aimdk_msgs.msg import McLocomotionVelocity
from aimdk_msgs.srv import GetMcAction, SetMcAction, SetMcInputSource


class X2Control:
    def __init__(self, node, check):
        self.node, self.check = node, check
        self.source = "slam_sim"
        self.forward = 0.
        self.registered = False
        prefix = "/aimdk_5Fmsgs/srv/"
        self.action = node.create_client(SetMcAction, prefix + "SetMcAction")
        self.query = node.create_client(GetMcAction, prefix + "GetMcAction")
        self.input = node.create_client(SetMcInputSource, prefix + "SetMcInputSource")
        self.publisher = node.create_publisher(McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10)
        self.timer = node.create_timer(.02, self.publish)

    def publish(self):
        if not self.registered:
            return
        message = McLocomotionVelocity()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.source = self.source
        message.forward_velocity = self.forward
        self.publisher.publish(message)

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check()
            rclpy.spin_once(self.node, timeout_sec=.005)

    def call(self, client, request, timeout=5.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check()
            if not client.service_is_ready():
                self.pump(.1)
                continue
            future = client.call_async(request)
            attempt_end = min(deadline, time.monotonic() + .5)
            while not future.done() and time.monotonic() < attempt_end:
                self.check()
                rclpy.spin_once(self.node, timeout_sec=.005)
            if future.done():
                return future.result()
            client.remove_pending_request(future)
        raise RuntimeError(f"Timed out calling {client.srv_name}; check AIMDK interface versions")

    def register(self):
        request = SetMcInputSource.Request()
        request.request.header.stamp = self.node.get_clock().now().to_msg()
        request.action.value = 1001
        request.input_source.name = self.source
        request.input_source.priority = 40
        request.input_source.timeout = 1000
        response = self.call(self.input, request)
        if response.response.header.code != 0:
            raise RuntimeError(f"X2 input registration failed: {response}")
        request.action.value = 2001
        request.request.header.stamp = self.node.get_clock().now().to_msg()
        response = self.call(self.input, request)
        if response.response.header.code != 0:
            raise RuntimeError(f"X2 input enable failed: {response}")
        self.registered = True
        self.publish()

    def current_action(self):
        request = GetMcAction.Request()
        request.request.header.stamp = self.node.get_clock().now().to_msg()
        return self.call(self.query, request).info.action_desc

    def set_mode(self, mode):
        request = SetMcAction.Request()
        request.header.stamp = self.node.get_clock().now().to_msg()
        if not hasattr(request, "source"):
            raise RuntimeError("Rebuild aimdk_msgs: SetMcAction must contain the SDK source field")
        request.source = self.source
        request.command.action_desc = mode
        deadline = time.monotonic() + 8
        response = None
        while time.monotonic() < deadline:
            response = self.call(self.action, request)
            self.pump(.15)
            if self.current_action() == mode:
                return
            if response.response.status.value not in (0,1,100,200,300,400):
                raise RuntimeError(f"X2 rejected {mode}: {response}")
        raise RuntimeError(f"X2 did not enter {mode}: {response}")
