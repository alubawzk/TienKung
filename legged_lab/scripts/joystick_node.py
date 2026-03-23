#!/usr/bin/env python3
"""
独立的ROS2手柄节点（方案2：直接读取/dev/input/js*设备文件）
以200Hz频率发布速度命令到vel_cmd话题
不依赖pygame，直接使用Linux Joystick API
"""

import os
import sys
import time
import threading
import struct
import glob

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    print("Error: ROS2 not available. Please install ROS2.")
    ROS2_AVAILABLE = False
    sys.exit(1)


# Linux Joystick 事件结构
# struct js_event {
#     __u32 time;     # 时间戳（毫秒）
#     __s16 value;    # 值（-32768 到 32767）
#     __u8 type;      # 事件类型
#     __u8 number;    # 轴/按钮编号
# }
JS_EVENT_BUTTON = 0x01  # 按钮按下/释放
JS_EVENT_AXIS = 0x02    # 轴移动
JS_EVENT_INIT = 0x80    # 初始化事件

JS_EVENT_SIZE = 8  # 事件结构大小（字节）


class JoystickReader:
    """直接读取 /dev/input/js* 设备文件的类"""
    
    def __init__(self, device_path=None):
        self.device_path = device_path
        self.device_file = None
        self.axes = {}  # 轴值字典 {axis_number: value}
        self.buttons = {}  # 按钮值字典 {button_number: value}
        self.running = False
        self.read_thread = None
        
    def find_joystick_device(self, target_name=None):
        """查找手柄设备文件"""
        # 查找所有 /dev/input/js* 设备
        devices = sorted(glob.glob('/dev/input/js*'))
        
        if not devices:
            return None
        
        # 如果指定了设备路径，直接使用
        if self.device_path and os.path.exists(self.device_path):
            return self.device_path
        
        # 如果指定了目标名称，尝试匹配
        if target_name:
            for device in devices:
                try:
                    # 读取设备名称（需要 root 权限或用户组权限）
                    name_path = device.replace('/dev/input/js', '/sys/class/input/js') + '/device/name'
                    if os.path.exists(name_path):
                        with open(name_path, 'r') as f:
                            device_name = f.read().strip()
                            if target_name.upper() in device_name.upper():
                                return device
                except:
                    pass
        
        # 返回第一个可用设备
        return devices[0] if devices else None
    
    def open_device(self, device_path=None, target_name="DF39"):
        """打开手柄设备文件"""
        if device_path:
            self.device_path = device_path
        else:
            self.device_path = self.find_joystick_device(target_name)
        
        if not self.device_path:
            return False
        
        try:
            self.device_file = open(self.device_path, 'rb')
            return True
        except PermissionError:
            print(f"Error: Permission denied. Try: sudo chmod 666 {self.device_path}")
            return False
        except Exception as e:
            print(f"Error opening device {self.device_path}: {e}")
            return False
    
    def read_event(self):
        """读取一个 joystick 事件"""
        if not self.device_file:
            return None
        
        try:
            # 读取8字节事件
            data = self.device_file.read(JS_EVENT_SIZE)
            if len(data) != JS_EVENT_SIZE:
                return None
            
            # 解析事件结构
            # 格式：<I (time), h (value), B (type), B (number)
            time_ms, value, event_type, number = struct.unpack('IhBB', data)
            
            return {
                'time': time_ms,
                'value': value,
                'type': event_type,
                'number': number
            }
        except Exception as e:
            return None
    
    def start_reading(self, callback=None):
        """启动读取线程"""
        if self.running:
            return
        
        if not self.device_file:
            return False
        
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, args=(callback,), daemon=True)
        self.read_thread.start()
        return True
    
    def _read_loop(self, callback):
        """读取循环（在独立线程中运行）"""
        while self.running:
            event = self.read_event()
            if event:
                # 更新轴或按钮值
                if event['type'] & JS_EVENT_AXIS:
                    # 轴值范围：-32768 到 32767，归一化到 -1.0 到 1.0
                    normalized_value = event['value'] / 32767.0
                    self.axes[event['number']] = normalized_value
                elif event['type'] & JS_EVENT_BUTTON:
                    self.buttons[event['number']] = event['value']
                
                # 调用回调函数
                if callback:
                    callback(event)
            else:
                time.sleep(0.001)  # 短暂休眠避免CPU占用过高
    
    def get_axis(self, axis_number, default=0.0):
        """获取轴值（归一化到 -1.0 到 1.0）"""
        return self.axes.get(axis_number, default)
    
    def get_button(self, button_number, default=0):
        """获取按钮值"""
        return self.buttons.get(button_number, default)
    
    def stop(self):
        """停止读取"""
        self.running = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        if self.device_file:
            self.device_file.close()
            self.device_file = None


class JoystickNode(Node):
    """ROS2手柄节点，直接读取/dev/input/js*，以200Hz发布速度命令"""
    
    def __init__(self):
        super().__init__('joystick_node')
        
        # 创建发布者
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'vel_cmd',
            10
        )
        
        # 发布频率：200Hz
        self.publish_freq = 200.0  # Hz
        self.publish_dt = 1.0 / self.publish_freq  # 秒
        
        # 初始化joystick读取器
        self.joystick = JoystickReader()
        
        # 速度命令
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0
        self.lock = threading.Lock()  # 用于保护共享数据
        
        # 命令缩放因子
        self.vx_scale = 0.3   # 前进/后退速度缩放
        self.vy_scale = 0.3   # 左右移动速度缩放
        self.dyaw_scale = 1.0  # 旋转速度缩放
        
        # 死区阈值（避免摇杆漂移）
        self.deadzone = 0.1
        
        # 初始化joystick
        if not self._init_joystick():
            self.get_logger().error("Failed to initialize joystick. Exiting.")
            sys.exit(1)
        
        # 创建定时器，以200Hz频率发布
        timer_period = self.publish_dt
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Joystick node started. Publishing at {self.publish_freq}Hz")
    
    def _init_joystick(self):
        """初始化joystick手柄"""
        try:
            # 尝试打开设备（优先查找DF39）
            if not self.joystick.open_device(target_name="DF39"):
                self.get_logger().warn("DF39 not found, trying first available device")
                if not self.joystick.open_device():
                    self.get_logger().error("No joystick device found")
                    return False
            
            self.get_logger().info(f"Opened joystick device: {self.joystick.device_path}")
            
            # 启动读取线程
            if not self.joystick.start_reading(callback=self._on_joystick_event):
                self.get_logger().error("Failed to start joystick reading thread")
                return False
            
            self.get_logger().info("Joystick read thread started")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error initializing joystick: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_joystick_event(self, event):
        """处理joystick事件（在读取线程中调用）"""
        try:
            # 读取摇杆轴值
            # 左摇杆：轴0（左右），轴1（前后）
            # 右摇杆：轴2（左右），轴3（前后）
            left_x = self.joystick.get_axis(0)
            left_y = self.joystick.get_axis(1)
            right_x = self.joystick.get_axis(2)
            right_y = self.joystick.get_axis(3)
            
            # 扳机键：轴4（左扳机），轴5（右扳机）
            # Linux joystick API 通常将扳机映射为 -32767（未按下）到 32767（按下）
            left_trigger_raw = self.joystick.get_axis(4)
            right_trigger_raw = self.joystick.get_axis(5)
            
            # 处理扳机键值：从 -1.0 到 1.0 转换为 0.0 到 1.0
            left_trigger = (left_trigger_raw + 1.0) / 2.0
            right_trigger = (right_trigger_raw + 1.0) / 2.0
            
            # 确保值在0到1范围内
            left_trigger = max(0.0, min(1.0, left_trigger))
            right_trigger = max(0.0, min(1.0, right_trigger))
            
            # 应用死区（对摇杆）
            if abs(left_x) < self.deadzone:
                left_x = 0.0
            if abs(left_y) < self.deadzone:
                left_y = 0.0
            if abs(right_x) < self.deadzone:
                right_x = 0.0
            if abs(right_y) < self.deadzone:
                right_y = 0.0
            
            # 映射到命令
            vx = -left_y * self.vx_scale      # 前进/后退（左摇杆前后）
            vy = -right_x * self.vy_scale     # 左右移动（右摇杆左右）
            
            # yaw角控制：使用扳机键
            # 左扳机：左转（负dyaw），右扳机：右转（正dyaw）
            dyaw = (right_trigger - left_trigger) * self.dyaw_scale
            
            # 线程安全更新命令值
            with self.lock:
                self.vx = vx
                self.vy = vy
                self.dyaw = dyaw
                
        except Exception as e:
            self.get_logger().error(f"Error processing joystick event: {e}")
    
    def timer_callback(self):
        """定时器回调：以200Hz频率发布速度命令"""
        if not rclpy.ok():
            return
        
        # 线程安全读取命令值
        with self.lock:
            vx = self.vx
            vy = self.vy
            dyaw = self.dyaw
        
        # 创建消息
        msg = Float32MultiArray()
        msg.data = [float(vx), float(vy), float(dyaw)]
        
        # 发布消息
        self.publisher.publish(msg)
    
    def destroy_node(self):
        """清理资源"""
        if self.joystick:
            self.joystick.stop()
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        joystick_node = JoystickNode()
        rclpy.spin(joystick_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'joystick_node' in locals():
            joystick_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

