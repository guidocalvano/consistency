from panda3d.core import TextNode
from direct.gui.DirectGui import *

from ToyShapeCreator import ToyShapeCreator
from CameraBase import CameraBase


camera_base = CameraBase().init()

nodePath = camera_base.render.attachNewNode(ToyShapeCreator.orientation_cube())

title = OnscreenText(text="Panda3D: Tutorial - Making a Cube Procedurally",
                     style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=.07,
                     parent=camera_base.a2dBottomRight, align=TextNode.ARight)

camera_base.run()
