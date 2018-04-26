from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTristrips, GeomNode, Vec3, Mat4, NodePath



class ToyShapeCreator:
    @staticmethod
    def orientation_cube():

        format = GeomVertexFormat.getV3n3c4t2()
        vdata = GeomVertexData('name', format, Geom.UHStatic)
        vdata.setNumRows(8 * 3)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        texcoord = GeomVertexWriter(vdata, 'texcoord')

        for i in range(8):
            x = 1 if (i & 1) > 0 else -1
            y = 1 if (i & 2) > 0 else -1
            z = 1 if (i & 4) > 0 else -1

            vertex.addData3f(x, y, z)
            vertex.addData3f(x, y, z)
            vertex.addData3f(x, y, z)

            normal.addData3f(x, 0, 0)
            normal.addData3f(0, y, 0)
            normal.addData3f(0, 0, z)

            r = .5 + x / 2
            g = .5 + y / 2
            b = .5 + z / 2

            color.addData4f(1, r, 0, 1)
            color.addData4f(0, 1, g, 1)
            color.addData4f(b, 0, 1, 1)

            texcoord.addData2f(g, b)
            texcoord.addData2f(b, r)
            texcoord.addData2f(r, g)

        prim = GeomTristrips(Geom.UHStatic)

        for axis in range(3):
            axis_bit = 2 ** axis
            side_bit = 2 ** axis

            u_bit = axis_bit * 2
            v_bit = u_bit * 2

            print("sidebit " + str(side_bit))

            print("ubit before " + str(u_bit))
            print("vbit before " + str(v_bit))

            if v_bit >= 2 ** 3:
                v_bit = 1
            if u_bit >= 2 ** 3:
                u_bit = 1
                v_bit = 2 * u_bit

            print("ubit after " + str(u_bit))
            print("vbit after " + str(v_bit))
            for side in range(2):

                next_vertices = []
                for u in range(2):
                    for v in range(2):
                        vertex_index = (u_bit * u + v_bit * v + side_bit * side)
                        print("vi" + str(vertex_index))
                        next_vertices.append(vertex_index * 3 + axis)

                if side == 1:
                    next_vertices.append(next_vertices.pop(0))
                    next_vertices.append(next_vertices.pop(0))

                prim.addVertices(*next_vertices)
                prim.close_primitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        geomNode = GeomNode('gnode')
        geomNode.addGeom(geom)

        node = NodePath(geomNode)
        earth_tex = loader.loadTexture("models/orientation_symbol2.png")
        node.setTexture(earth_tex, 1)

        return geomNode
