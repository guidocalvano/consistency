from panda3d.core import \
    Geom,\
    GeomVertexFormat,\
    GeomVertexData,\
    GeomVertexWriter,\
    GeomTristrips,\
    GeomNode,\
    Vec3,\
    Mat4,\
    NodePath,\
    Texture

import numpy as np


class MatrixCapsVisualizer:

    def invert_projection(self, polygon_coordinate_system, screen_coordinates):

        # The filter weights are a projection of a polygon with pose of the capsule output. One axis of the capsule
        # pose is the normal of the polygon, the other axes correspond to the uv coordinates.
        # The projection of the polygon is a square, with each corner corresponding to to a point in the original
        # polygon. We can find those points by tracing the lines of projection back to the uv plane of the polygon.
        # In the coordinate system of the camera this is hard. In the coordinate system of the polygon this is easy.

        # the 2D screen coordinates can be converted to camera coordinates: the camera plane is at z == 1,
        # and the vectors are position vectors
        z_and_vector_type_rows = np.ones(screen_coordinates.shape)

        camera_coordinates = np.concatenate([screen_coordinates, z_and_vector_type_rows])

        camera_origin = np.array([0, 0, 0])
        from_camera_to_polygon_coordinate_system = np.linalg.inv(polygon_coordinate_system)
        # from here on we operate in the polygon coordinate system

        # the projection beams end at the origin of the camera coordinate system
        projection_beam_end = from_camera_to_polygon_coordinate_system * camera_origin

        # the location of the projected coordinates;
        #   i.e. the location of the square projection of the polygon
        #   on the screen floating in 3D space expressed in coordinates in the polygon coordinate system
        projected_coordinates = from_camera_to_polygon_coordinate_system * camera_coordinates

        reversed_direction_projection_beam = projected_coordinates - projection_beam_end  # broadcasting should duplicate beam end

        # we now have a model of the projection line from the beam end to the polygon's uv plane:
        # projection_beam_end + distance * reversed_direction_projection_beam
        # where distance is expresses the distance from projection_beam_end in lengths of
        # the vector reversed_direction_projection_beam (note that the lengths of this vector are different for each
        # screen coordinate)
        # The normal axis (i.e. z axis) of the polygon coordinate system expresses distance to the uv plane.
        # The projection beam started at the uv plane so:
        # projection_beam_end + distance * reversed_direction_projection_beam == [u, v, 0]
        # or
        # projection_beam_end.z + distance * reversed_direction_projection_beam.z == 0
        # or
        # distance = -projection_beam_end.z / reversed_direction_projection_beam.z
        distance = np.reciprocal(reversed_direction_projection_beam[:, 2]) * (-projection_beam_end[2])

        polygon_coordinates = projection_beam_end + reversed_direction_projection_beam * distance

        return polygon_coordinates

    def create_filter_polygon(self, source_coordinate_system, screen_coordinates, filter, format='RGB'):

        polygon_coordinates = self.invert_projection(source_coordinate_system, screen_coordinates)

        node = self.construct_textured_polygon(polygon_coordinates, filter, format)

        return node

    def construct_textured_polygon(self, polygon_coordinates, filter, format='RGB'):
        texture = Texture()
        texture.setRamImageAs(filter, format)

        geom = self.construct_polygon_geometry(polygon_coordinates)

        geomNode = GeomNode('gnode')
        geomNode.addGeom(geom)

        node = NodePath(geomNode)
        node.setTexture(texture, 1)

        return node

    def construct_polygon_geometry(self, polygon_coordinates):
        geom_format = GeomVertexFormat.getV3t2()
        vdata = GeomVertexData('name', geom_format, Geom.UHStatic)
        vdata.setNumRows(4)

        vertex = GeomVertexWriter(vdata, 'vertex')
        texcoord = GeomVertexWriter(vdata, 'texcoord')

        for i in range(4):
            vertex.addData3f(*polygon_coordinates[0, :])
            texcoord.addData3f(int((i & 2) > 0), int((i & 1) > 0))  # uv coordinates correspond to binary components

        prim = GeomTristrips(Geom.UHStatic)

        prim.addVertices(0, 1, 2, 3)
        prim.close_primitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        return geom