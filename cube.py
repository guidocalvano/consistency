from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import *
from panda3d.core import TextNode, Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles, GeomNode, Vec3, Mat4
import sys
from direct.task.Task import Task
import numpy as np
import scipy as sp
import scipy.linalg
from pandac.PandaModules import WindowProperties


class CameraBase(ShowBase):
    def init(self):
        self.setBackgroundColor(0.0, 0.0, 0.0)
        # camera_position = render.attachNewNode('camera')
        # self.camera.reparent_to(camera_position)
        self.disable_mouse()

        self.camera.setPos(0, 0, 10)
        self.camera.lookAt(100.0, 0.0, 0.0)

        self.motion_mat4 = Mat4(Mat4.identMat())
        # self.motion_mat4.setCell(3, 1, 5)

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

        return self

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




            print(x)
            print(y)


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
        props = WindowProperties()
        props.setCursorHidden(True)
        props.setMouseMode(WindowProperties.M_relative)

        window_properties = self.win.getProperties()
        self.win.movePointer(0,
                             int(window_properties.getXSize() / 2),
                             int(window_properties.getYSize() / 2))
        self.win.requestProperties(props)

    def deactivate_mouse_view_mode(self):
        props = WindowProperties()
        props.setCursorHidden(False)
        props.setMouseMode(WindowProperties.M_absolute)
        self.win.requestProperties(props)

camera_base = CameraBase().init()


format = GeomVertexFormat.getV3n3c4t2()
vdata = GeomVertexData('name', format, Geom.UHStatic)
vdata.setNumRows(3)

# for i in range(8):
#     x = 1 if (i & 1) > 0 else -1
#     y = 1 if (i & 2) > 0 else -1
#     z = 1 if (i & 4) > 0 else -1
#
#     vertex.addData3f(x, y, z)
#

vertex = GeomVertexWriter(vdata, 'vertex')
normal = GeomVertexWriter(vdata, 'normal')
color = GeomVertexWriter(vdata, 'color')
texcoord = GeomVertexWriter(vdata, 'texcoord')

vertex.addData3f(0, 0, 0)
normal.addData3f(0, -1, 0)
color.addData4f(0, 0, 0, 1)
texcoord.addData2f(0, 0)

vertex.addData3f(100, 0, 0)
normal.addData3f(0, -1, 0)
color.addData4f(1, 0, 0, 1)
texcoord.addData2f(1, 0)

vertex.addData3f(0, 100, 0)
normal.addData3f(0, -1, 0)
color.addData4f(0, 1, 0, 1)
texcoord.addData2f(0, 1)

prim = GeomTriangles(Geom.UHStatic)
prim.addVertices(0,1,2)

geom = Geom(vdata)
geom.addPrimitive(prim)

node = GeomNode('gnode')
node.addGeom(geom)

nodePath = camera_base.render.attachNewNode(node)
earth_tex = camera_base.loader.loadTexture("models/earth_1k_tex.jpg")
# nodePath.setTexture(earth_tex, 1)

title = OnscreenText(text="Panda3D: Tutorial - Making a Cube Procedurally",
                     style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=.07,
                     parent=camera_base.a2dBottomRight, align=TextNode.ARight)

camera_base.run()
