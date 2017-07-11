# For license and more information see the LICENSE and README.MD files
import bpy
import bmesh
import numpy as np
from scipy.optimize import linprog
from mathutils import Vector
import math


# #========================================================# #
# |                                                        | #
# | Methods for basic mesh calculations                    | #
# |                                                        | #
# #========================================================# #

def is_a_mesh(bm):
    for edge in bm.edges:
        l = len(edge.link_faces)
        if l > 2 or l == 0:
            return False
    return True


def is_quad_mesh(bm):
    if not is_a_mesh(bm):
        return False
    for vertex in bm.verts:
        l = len(vertex.link_edges)
        if l > 4:
            return False
        if l < 4:
            if not is_boundary_vertex(vertex):
                return False
    for face in bm.faces:
        if len(face.verts) != 4:
            return False
    return True


def has_convex_faces(bm):
    for face in bm.faces:
        if not is_convex([v.co for v in face.verts]):
            return False
    return True


def is_convex(vertices):
    l = len(vertices)
    n = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[1]).normalized()
    for i in range(1, l):
        m = (vertices[(i+1)%l] - vertices[i]).cross(vertices[(i+2)%l] - vertices[(i+1)%l]).normalized()
        if (m - n).length >= error:
            return False
    return True


def is_degenerate(vertices):
    l = len(vertices)
    for i in range(0, l):
        if (vertices[i] - vertices[(i+1) % l]).length < error:
            return True
    return False


def is_boundary_vertex(vertex):
    n = 0
    for edge in vertex.link_edges:
        n += len(edge.link_faces)
    if n>= 2*len(vertex.link_edges):
        return False
    return True


def is_boundary_edge(v, u):
    return is_boundary_vertex(v) and is_boundary_vertex(u)


def is_boundary_face(face):
    for edge in face.edges:
        if is_boundary_edge(edge.verts[0], edge.verts[1]):
            return True
    return False


#anle between (v,u) and (v,w)
def angle(v, u, w):
    a = u-v
    b = w-v
    a.normalize()
    b.normalize()
    if a == b:
        return 0
    return np.arccos(a.dot(b)) * 180 / np.pi


def get_next_edge(vertex, visited):
    for face in visited[len(visited)-1].link_faces:
        for edge in face.edges:
            if (edge.verts[0] == vertex or edge.verts[1] == vertex) and not edge in visited:
                return edge


def get_ordered_adjacent_vertices(vertex):
    min = math.inf
    for e in vertex.link_edges:
        l = len(e.link_faces)
        if (l < min):
            edge = e
            min = l
    visited = [edge]
    while edge is not None:
        edge = get_next_edge(vertex, visited)
        if edge is not None:
            visited.append(edge)

    vertices = []
    for e in visited:
        vertices.append(e.other_vert(vertex))
    return(vertices)


def get_adjacent_vertices(vertex):
    vertices = []
    for e in vertex.link_edges:
        vertices.append(e.other_vert(vertex))
    return vertices


def are_coplanar(a, b, c, d):
    return (a - d).dot((a-b).cross(a-c)) == 0


def is_angle_criterion_true(w1, w2, w3, w4):
    return np.abs(w1 + w3 - w2 - w4) < error


def is_conical_mesh(bm):
    if not is_quad_mesh(bm):
        return False
    for vertex in bm.verts:
        if not is_boundary_vertex(vertex):
            others = get_ordered_adjacent_vertices(vertex)
            if not are_coplanar(others[0].co, others[1].co, others[2].co, others[3].co):
                w1 = angle(vertex.co, others[0].co, others[1].co)
                w2 = angle(vertex.co, others[1].co, others[2].co)
                w3 = angle(vertex.co, others[2].co, others[3].co)
                w4 = angle(vertex.co, others[3].co, others[0].co)
                if not is_angle_criterion_true(w1, w2, w3, w4):
                    print("Angle criterion failed at vertex: " 
                        + str(vertex.co) + "," + str(w1) + "," + str(w2) + "," + str(w3) + "," + str(w4))
                    return False
    return True


def bisector_plane(n1, n2):
    n = n1 + n2
    n.normalize()
    m = intersect_planes(n1, n2)
    m.normalize()
    return n.cross(m)


def intersect_planes(n1, n2):
    n = n1.cross(n2)
    n.normalize()
    return n


def face_containing(faces, u, v, w):
    for face in faces:
        vertices = face.verts
        if u in vertices and v in vertices and w in vertices:
            return face


def edge_between(f, g):
    for edge in f.edges:
        if edge in g.edges:
            return edge


def bisector_planes(vertex):
    vertices = get_ordered_adjacent_vertices(vertex)
    l = len(vertices)
    faces = []
    for i in range(0, l):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % l]
        face = face_containing(vertex.link_faces, vertex, v1, v2)
        if not face is None:
            faces.append(face)
    l = len(faces)
    n = []
    for i in range(0,l):
        f1 = faces[i]
        f2 = faces[(i + 1) % l]
        n1 = f1.normal
        n2 = f2.normal
        if (n1 - n2).length < error:
            edge = edge_between(f1, f2)
            tmp = n1.cross(edge.verts[0].co - edge.verts[1].co)
        else:
            tmp = bisector_plane(n1, n2)
        n.append(tmp)
    return n


# #========================================================# #
# |                                                        | #
# | Methods for Gauss-image calculations of a conical mesh | #
# |                                                        | #
# #========================================================# #

#calculates the Gauss-images for all vertices
def calculate_gauss_image(bm):
    s = []
    for vertex in bm.verts:
        if not is_boundary_vertex(vertex):
            n = calculate_normal(vertex)
            s.append((vertex.index, n))
    for vertex in bm.verts:
        if is_boundary_vertex(vertex):
            n = calculate_normal(vertex)
            s.append((vertex.index, n))
    return s


#calculates the Gauss-image for the given vertex
def calculate_normal(vertex):
    #vertex has link_faces <= 2 and an adjacent inner vertex
    if len(vertex.link_faces) <= 2:
        for e in vertex.link_edges:
            v = e.other_vert(vertex)
            if not is_boundary_vertex(v):
                return normal(v)
        for f in vertex.link_faces:
            for v in f.verts:
                if not is_boundary_vertex(v):
                    return normal(v)

        n = Vector((0,0,0))
        for f in vertex.link_faces:
            n += f.normal
        return n

    #vertex has link_faces <= 2 but no adjacent inner vertex
    if len(vertex.link_faces) == 1:
        return vertex.link_faces[0].normal

    if len(vertex.link_faces) == 2:
        fn1 = vertex.link_faces[0].normal
        fn2 = vertex.link_faces[1].normal
        n1 = bisector_plane(fn1, fn2)
        e = []
        for v in get_adjacent_vertices(vertex):
            if is_boundary_vertex(v):
                e.append(vertex.co - v.co)
        n = intersect_planes(n1, e[0].cross(e[1]))
        n.normalize()
        return n

    #vertex has link_faces >= 3
    normals = bisector_planes(vertex)
    n1 = normals[0]
    for n in normals:
        if (n-n1).length > error:
            n2 = n
            break
    n = intersect_planes(n1, n2)
    n.normalize()
    for face in vertex.link_faces:
        if face.normal.dot(n) < 0:
            return -n
    return n


def normal(vertex):
    return [e[1] for e in s if e[0] == vertex.index][0]


# #========================================================# #
# |                                                        | #
# | Methods for calculating a parallel mesh to the given   | #
# |                                                        | #
# #========================================================# #

#calculates a parallel mesh at constant face-distance d to the given bmesh and joins them
def parallel_mesh(bm, d):
    parallel = calculate_parallel_mesh(bm, d)
    join_bmeshes(bm, parallel)


#joins bm2 to bm1
def join_bmeshes(bm1, bm2):
    tmp_mesh = bpy.data.meshes.new("tmp")
    bm2.to_mesh(tmp_mesh)
    bm1.from_mesh(tmp_mesh)
    bpy.data.meshes.remove(tmp_mesh)


#constant face-offset
def calculate_parallel_mesh(bm, d):
    parallel = bm.copy()
    for vertex in parallel.verts:
        nv = normal(vertex)
        nf = vertex.link_faces[0].normal
        alpha = angle(Vector((0,0,0)), nv, nf)
        dist = d / math.cos(math.radians(alpha))
        vertex.co = vertex.co + dist*nv
    return parallel


#face to the left of directed edge (u,v)
def side_face_normal(u, v, n_v):
    n = (v - u).cross(n_v)
    return n.normalized()


# #========================================================# #
# |                                                        | #
# | Methods for local freedom calculations - conical mesh  | #
# |                                                        | #
# #========================================================# #

#Calculates local freedom of face via solving linear programs
def is_locally_free(face):
    precision = len(str(error))-2
    vertices = face.verts #assumes ordered vertices
    l = len(vertices)
    n = []
    for i in range(0, l):
        v = vertices[i]
        u = vertices[(i+1) % l]
        if not is_boundary_edge(v, u):
            ni = np.array(side_face_normal(v.co, u.co, normal(v)).to_tuple(precision))
            line = np.append(ni, -ni)
            n.append(line)
    nf = np.array(face.normal.to_tuple(precision))
    c = np.append(nf, -nf)
    n = np.array(n)
    b = [0] * len(n)
    x = linprog(c, A_ub = n, b_ub = b)
    if x.status == 3: #unbounded
        return True
    x = linprog(-c, A_ub = n, b_ub = b)
    if x.status == 3: #unbounded
        return True
    return False


def signed_area(verts, normal):
    total = Vector((0, 0, 0))
    l = len(verts)
    for i in range(0, l):
        total += verts[i].cross(verts[(i+1) % l])
    result = total.dot(normal)
    return result / 2


def gauss_curvature_parallel(face):
    s = []
    v = []
    d = 1
    for vertex in face.verts:
        nv = normal(vertex)
        alpha = angle(Vector((0,0,0)), nv, face.normal)
        if alpha < error:
            dist = d
        else:
            dist = d / math.cos(math.radians(alpha))
        s.append(dist * nv)
        v.append(vertex.co)
    return signed_area(s, face.normal) / signed_area(v, face.normal)


# #========================================================# #
# |                                                        | #
# | Main program selecting all locally free faces of a     | #
# | given conical mesh                                     | #
# |                                                        | #
# #========================================================# #

def reason(r):
    return {
        "b": "since it is a boundary face",
        "k": "detected by Gaussian curvature",
        "l": "detected by Linear Program",
    }[r]

def print_face(face, k, free, r):
    l = len(str(error))-2
    b = "" if free else "not "
    text = reason(r)
    gauss = ("{0:." + str(l) + "f}").format(k)
    print("face " + str(face.index) + " with K = " + gauss + " is " + b + "locally free, " + text)

def select_locally_free_faces(bm):
    for face in bm.faces:
        s_f = [normal(v) for v in face.verts]
        k = gauss_curvature_parallel(face)
        if is_boundary_face(face):
            face.select = True
            print_face(face, k, True, "b")
        else:
            if is_convex(s_f) and not is_degenerate(s_f):
                if k > 0:
                    face.select = True
                    print_face(face, k, True, "k")
                else:
                    face.select = False
                    print_face(face, k, False, "k")
            else:
                if is_locally_free(face):
                    face.select = True
                    print_face(face, k, True, "l")
                else:
                    face.select = False
                    print_face(face, k, False, "l")


# precision that is considered to be zero
error = 0.005

mode = bpy.context.active_object.mode
bpy.ops.object.mode_set(mode='EDIT')

obj = bpy.context.edit_object
mesh = obj.data
bm = bmesh.from_edit_mesh(mesh)
bm.transform(obj.matrix_world)
obj.matrix_world = Matrix()
bm.normal_update()
if is_conical_mesh(bm):
    if not has_convex_faces(bm):
        print("Given mesh has faces that are not convex or not planar.")
    else:
        s = calculate_gauss_image(bm)
        select_locally_free_faces(bm)
        #parallel_mesh(bm, 0.5)
    bmesh.update_edit_mesh(mesh)
else:
    print("Given mesh is not conical.")

bpy.ops.object.mode_set(mode=mode)