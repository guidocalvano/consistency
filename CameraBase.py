from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3, Mat4, NodePath, Texture, GraphicsPipe, GraphicsOutput, FrameBufferProperties, WindowProperties
import sys
from direct.task.Task import Task
from pandac.PandaModules import WindowProperties as WindowPropertiesPM
import numpy as np
import scipy as sp
import scipy.linalg
import cv2


class CameraBase(ShowBase):
    def init(self):
        self.setBackgroundColor(0.0, 0.0, 0.0)
        # camera_position = render.attachNewNode('camera')
        # self.camera.reparent_to(camera_position)
        self.disable_mouse()

        self.camera.setPos(0, -10, 0)
        self.camera.lookAt(0.0, 0.0, 0.0)

        self.motion_mat4 = Mat4(Mat4.identMat())

        self.mouse_view_mode = False

        self.prevtime = None

        self.taskMgr.add(self.motion, "moveTask")

        self.accept("escape", sys.exit)

        self.accept("w", self.start_up)
        self.accept("w-up", self.stop_up_down)
        self.accept("s", self.start_down)
        self.accept("s-up", self.stop_up_down)

        self.accept("a", self.start_left)
        self.accept("a-up", self.stop_left_right)
        self.accept("d", self.start_right)
        self.accept("d-up", self.stop_left_right)

        self.accept("arrow_up", self.start_forward)
        self.accept("arrow_up-up", self.stop_forward_back)
        self.accept("arrow_down", self.start_back)
        self.accept("arrow_down-up", self.stop_forward_back)

        self.accept("mouse1", self.toggle_mouse_view_mode)

        self.accept("p", self.make_picture)


        self.dr = self.camNode.getDisplayRegion(0)

        self._init_depth_cam()

        return self

    def _init_depth_cam(self):
        winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        fbprops = FrameBufferProperties()
        fbprops.setDepthBits(1)

        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe, "depth buffer", -2,
            fbprops, winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)
        self.depthTex = Texture()
        self.depthTex.setFormat(Texture.FDepthComponent)
        self.depthBuffer.addRenderTexture(self.depthTex,
            GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
        lens = self.cam.node().getLens()
        # the near and far clipping distances can be changed if desired
        # lens.setNear(5.0)
        # lens.setFar(500.0)
        self.depthCam = self.makeCamera(self.depthBuffer,
            lens=lens,
            scene=self.render)
        self.depthCam.reparentTo(self.cam)

    def motion(self, task):
        # This method is called every frame, since it is attached to taskMgr.
        # The elapsed time is the current time minus the last saved time
        if self.prevtime is None:
            self.prevtime = task.time

            return Task.cont

        elapsed = task.time - self.prevtime
        self.prevtime = task.time
        currentMat = Mat4()
        elapsedMotion = scipy.linalg.fractional_matrix_power(np.array(self.motion_mat4), elapsed)
        currentMat.multiply(Mat4(*elapsedMotion.flatten().tolist()), self.camera.getMat())


        if self.mouse_view_mode:
            x = self.mouseWatcherNode.getMouseX()
            y = self.mouseWatcherNode.getMouseY()
            props = self.win.getProperties()
            self.win.movePointer(0,
                                 int(props.getXSize() / 2),
                                 int(props.getYSize() / 2))
            # self.win.movePointer(0, 0, 0)

            x_rotation = Mat4(Mat4.identMat())
            x_rotation.setRotateMatNormaxis(x * -15.0, Vec3(0, 0, 1))
            y_rotation = Mat4(Mat4.identMat())
            y_rotation.setRotateMatNormaxis(y * -15.0, Vec3(1, 0, 0))

            currentMat = x_rotation * currentMat
            currentMat = y_rotation * currentMat

        self.camera.setMat(currentMat)

        return Task.cont

    def start_up(self):
        print("start up")
        self.motion_mat4.setCell(3, 2, 5)

    def start_down(self):
        print("start up")
        self.motion_mat4.setCell(3, 2, -5)

    def stop_up_down(self):
        print("stop up")
        self.motion_mat4.setCell(3, 2, 0)

    def start_left(self):
        print("stop up")
        self.motion_mat4.setCell(3, 0, -5)

    def start_right(self):
        print("stop up")
        self.motion_mat4.setCell(3, 0, 5)

    def stop_left_right(self):
        print("stop up")
        self.motion_mat4.setCell(3, 0, 0)

    def start_forward(self):
        print("stop up")
        self.motion_mat4.setCell(3, 1, 5)

    def start_back(self):
        print("stop up")
        self.motion_mat4.setCell(3, 1, -5)

    def stop_forward_back(self):
        print("stop up")
        self.motion_mat4.setCell(3, 1, 0)

    def toggle_mouse_view_mode(self):
        self.mouse_view_mode = not self.mouse_view_mode

        if self.mouse_view_mode:
            self.activate_mouse_view_mode()
            return

        self.deactivate_mouse_view_mode()

    def activate_mouse_view_mode(self):
        props = WindowPropertiesPM()
        props.setCursorHidden(True)
        props.setMouseMode(WindowPropertiesPM.M_relative)

        window_properties = self.win.getProperties()
        self.win.movePointer(0,
                             int(window_properties.getXSize() / 2),
                             int(window_properties.getYSize() / 2))
        self.win.requestProperties(props)

    def deactivate_mouse_view_mode(self):
        props = WindowPropertiesPM()
        props.setCursorHidden(False)
        props.setMouseMode(WindowPropertiesPM.M_absolute)
        self.win.requestProperties(props)

    def make_picture(self):
        img = self.get_camera_image()
        d_img = self.get_camera_depth_image()

        self.show_rgbd_image(img, d_img)

    def get_camera_image(self, requested_format="RGBA"):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """
        tex = self.dr.getScreenshot()
        if requested_format is None:
            data = tex.getRamImage()
        else:
            data = tex.getRamImageAs(requested_format)
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image

    def get_camera_depth_image(self):
        """
        Returns the camera's depth image, which is of type float32 and has
        values between 0.0 and 1.0.
        """
        data = self.depthTex.getRamImage()
        depth_image = np.frombuffer(data, np.float32)
        depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
        depth_image = np.flipud(depth_image)
        return depth_image

    def show_rgbd_image(self, image, depth_image, window_name='Image window', delay=1, depth_offset=0.0, depth_scale=1.0):
        if depth_image.dtype != np.uint8:
            if depth_scale is None:
                depth_scale = depth_image.max() - depth_image.min()
            if depth_offset is None:
                depth_offset = depth_image.min()
            depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
            depth_image = (255.0 * depth_image).astype(np.uint8)
        depth_image = np.tile(depth_image, (1, 1, 3))
        if image.shape[2] == 4:  # add alpha channel
            alpha = np.full(depth_image.shape[:2] + (1,), 255, dtype=np.uint8)
            depth_image = np.concatenate([depth_image, alpha], axis=-1)
        images = np.concatenate([image, depth_image], axis=1)
        images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR) 
        cv2.imshow(window_name, images)
        key = cv2.waitKey(delay)
        key &= 255
        if key == 27 or key == ord('q'):
            print("Pressed ESC or q, exiting")
            exit_request = True
        else:
            exit_request = False
        return exit_request