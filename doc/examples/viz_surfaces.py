"""
==================
Visualize surfaces
==================

Here is a simple tutorial that shows how to visualize surfaces using DIPY_. It
also shows how to load/save, get/set and update ``vtkPolyData`` and show
surfaces.

``vtkPolyData`` is a structure used by VTK to represent surfaces and other data
structures. Here we show how to visualize a simple cube but the same idea
should apply for any surface.
"""

import numpy as np

"""
Import useful functions from ``dipy.viz.utils``
"""

import dipy.io.vtk as io_vtk
import dipy.viz.utils as ut_vtk
from dipy.viz import window

# Conditional import machinery for vtk
# Allow import, but disable doctests if we don't have vtk
from dipy.utils.optpkg import optional_package
vtk, have_vtk, setup_module = optional_package('vtk')

interactive = True
thick_edges = False

"""
Create an empty ``vtkPolyData``
"""

my_polydata = vtk.vtkPolyData()

"""
Create a cube with vertices and triangles as numpy arrays
"""

my_vertices = np.array([[0.0,  0.0,  0.0],
                       [0.0,  0.0,  1.0],
                       [0.0,  1.0,  0.0],
                       [0.0,  1.0,  1.0],
                       [1.0,  0.0,  0.0],
                       [1.0,  0.0,  1.0],
                       [1.0,  1.0,  0.0],
                       [1.0,  1.0,  1.0]])
# the data type for vtk is needed to mention here, numpy.int64
my_triangles = np.array([[0,  6,  4],
                         [0,  2,  6],
                         [0,  3,  2],
                         [0,  1,  3],
                         [2,  7,  6],
                         [2,  3,  7],
                         [4,  6,  7],
                         [4,  7,  5],
                         [0,  4,  5],
                         [0,  5,  1],
                         [1,  5,  7],
                         [1,  7,  3]],dtype='i8')


"""
Set vertices and triangles in the ``vtkPolyData``
"""

ut_vtk.set_polydata_vertices(my_polydata, my_vertices)
ut_vtk.set_polydata_triangles(my_polydata, my_triangles)

"""
Save the ``vtkPolyData``
"""

file_name = "my_cube.vtk"
io_vtk.save_polydata(my_polydata, file_name)
print("Surface saved in " + file_name)

"""
Load the ``vtkPolyData``
"""

cube_polydata = io_vtk.load_polydata(file_name)

"""
add color based on vertices position
"""

cube_vertices = ut_vtk.get_polydata_vertices(cube_polydata)
colors = cube_vertices * 255
ut_vtk.set_polydata_colors(cube_polydata, colors)

print("new surface colors")
print(ut_vtk.get_polydata_colors(cube_polydata))

"""
Visualize surfaces
"""

# get vtkActor
cube_actor = ut_vtk.get_actor_from_polydata(cube_polydata)


if thick_edges:
    cube_actor.GetProperty().SetEdgeVisibility(1)
    cube_actor.GetProperty().SetEdgeColor(0.9, 0.9, 0.4)
    cube_actor.GetProperty().SetLineWidth(6)
    cube_actor.GetProperty().SetPointSize(12)
    cube_actor.GetProperty().SetRenderLinesAsTubes(1)
    cube_actor.GetProperty().SetRenderPointsAsSpheres(1)
    cube_actor.GetProperty().SetVertexVisibility(1)
    cube_actor.GetProperty().SetVertexColor(0.5, 1.0, 0.8)


gl_mapper = cube_actor.GetMapper()

gl_mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    "//VTK::ValuePass::Impl",  # replace the normal block
    False,
    "//VTK::ValuePass::Impl\n" # we still want the default
    #"gl_Position = MCDCMatrix * vertexMC;\n"
    "gl_Position = gl_Position + 10 * sin(vertexMC);\n",
    #"gl_Position = vec4(0, 0, 0, 1.0);\n",
    False)

gl_mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    "//VTK::Light::Impl",
    True,
    "//VTK::Light::Impl\n"
    "fragOutput0 = vec4(1, 0, 0, lineWidthPercentageBlack);\n abcd",
    False)


gl_mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,  # in the fragment shader
        "//VTK::Coincident::Dec",  # replace the normal block
        True,  # before the standard replacements
        "//VTK::Coincident::Dec\n"  # we still want the default
        # "  uniform vec3 mycolor;\n"
        # "  uniform vec3 cameraPos;\n"
        "uniform float lineWidthPercentageBlack;\n"
        "uniform float lineWidthDepthCueingFactor;\n"
        "uniform float lineHaloMaxDepth;\n"
        "uniform vec3 colorLine;\n"
        "uniform vec3 colorHalo;\n"
        "in vec3 positionVSOutput;\n"
        "in vec3 directionVSOutput;\n"
        "in vec2 uvVSOutput;\n",  # but we add this
        False)


# renderer and scene
renderer = window.Renderer()
renderer.add(cube_actor)
renderer.set_camera(position=(10, 5, 7), focal_point=(0.5, 0.5, 0.5))

camera = renderer.get_camera()

#uniform
@vtk.calldata_type(vtk.VTK_OBJECT)
def vtkShaderCallback(caller, event, calldata=None):
    camera = renderer.GetActiveCamera()
    cameraPos = camera.GetPosition()
    projMat = camera.GetProjectionTransformMatrix(renderer)
    viewMat = camera.GetViewTransformMatrix()
    program = calldata
    if program is not None:
        program.SetUniformf("lineTriangleStripWidth", 0.25)
        program.SetUniformf("lineWidthPercentageBlack", 0.5)
        program.SetUniformf("lineWidthDepthCueingFactor", 0.5)
        program.SetUniformf("lineHaloMaxDepth", 0.005)
        # program.SetUniformf("lineTriangleStripWidth", 0.05)
        # program.SetUniformf("lineWidthPercentageBlack", 0.3)
        # program.SetUniformf("lineWidthDepthCueingFactor", 1.0)
        # program.SetUniformf("lineHaloMaxDepth", 0.02)
        program.SetUniform3f("colorLine", [0.0, 0.0, 0.0])
        program.SetUniform3f("colorHalo", [1.0, 1.0, 1.0])
        # program.SetUniform3f("colorHalo", [0.8, 0.8, 0.8])
        program.SetUniform3f("mycolor", [0.4, 0.7, 0.6])
        program.SetUniform3f("cameraPos", cameraPos)
        program.SetUniformMatrix("projMat", projMat)
        program.SetUniformMatrix("viewMat", viewMat)

#for attributes that change per vertex see
# https://github.com/dmreagan/dipy/blob/halo-line/dipy/viz/utils.py#L458
# MapDataArrayToVertexAttribute
# https://github.com/dmreagan/dipy/blob/halo-line/dipy/viz/actor.py#L668

gl_mapper.AddObserver(vtk.vtkCommand.UpdateShaderEvent,
                      vtkShaderCallback)


renderer.zoom(3)

# display
if interactive:
    window.show(renderer, size=(600, 600), reset_camera=False)
# window.record(renderer, out_path='cube.png', size=(600, 600))

"""
.. figure:: cube.png
   :align: center

   An example of a simple surface visualized with DIPY.

.. include:: ../links_names.inc

"""
