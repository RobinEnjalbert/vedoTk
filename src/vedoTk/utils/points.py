from vedo import Points, Mesh, Arrows, Plotter
import numpy as np
from numpy.linalg import norm


def compute_normals_with_pca(obj: Points, flip: bool = False, n: int = 20, display: bool = True):

    # Compute the normals
    obj.compute_normals_with_pca(n=n, invert=flip)

    # Display result
    if display:
        plt = Plotter(title='Compute normals with PCA')
        plt.add(obj)
        plt.add(Arrows(obj.vertices, obj.vertices + obj.vertex_normals / norm(obj.vertex_normals), c='g', s=0.75))

        def button(*args, **kwargs):
            bu.switch()
            update()

        def slider(*args, **kwargs):
            sw.value = int(sw.value)
            update()

        def update():
            obj.compute_normals_with_pca(invert=bool(bu.status_idx), n=int(sw.value))
            plt.remove(plt.objects[-1])
            plt.add(Arrows(obj.vertices, obj.vertices + obj.vertex_normals / norm(obj.vertex_normals), c='g', s=0.75))
            plt.render()

        bu = plt.add_button(fnc=button, states=['Flip normals: No ', 'Flip normals: Yes'], bc=['red4', 'green4'],
                            size=20, pos=(0.2, 0.075))
        if flip:
            bu.switch()
        sw = plt.add_slider(sliderfunc=slider, xmin=5, xmax=min(50, obj.nvertices),
                            value=min(20, obj.nvertices), title='PCA neighborhood (default: 20)')
        plt.show().close()


def generate_triangles(obj: Points, filter_triangle_size: bool = True, filter_triangle_quality: bool = True) -> Mesh:

    # Generate triangles
    surface = obj.generate_delaunay2d(mode='fit')

    # Filter by triangle size
    if filter_triangle_size:
        size = np.array([[norm(surface.vertices[t[1]] - surface.vertices[t[0]]),
                          norm(surface.vertices[t[2]] - surface.vertices[t[1]]),
                          norm(surface.vertices[t[0]] - surface.vertices[t[2]])] for t in surface.cells])
        t_filter = np.argwhere(np.max(size, axis=1) < size.mean() + size.std()).flatten()
        surface = Mesh([surface.vertices, np.array(surface.cells)[t_filter]])

    # Filter by triangle quality
    if filter_triangle_quality:
        surface.compute_quality()
        t_filter = np.argwhere(surface.celldata['Quality'] > 15).flatten()
        surface = Mesh([surface.vertices, np.array(surface.cells)[t_filter]])

    return surface.compute_normals()

