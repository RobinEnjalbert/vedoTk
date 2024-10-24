from os.path import join

from vedoTk import VisiblePoints
from vedo import Plotter, Mesh, Points


class Scene:

    def __init__(self):
        self.mesh = Mesh(join('resources', 'bunny.obj'), c='y5')
        self.visible_points = Points()
        self.visible_points_extractor = VisiblePoints(mesh=self.mesh)

        self.plt = Plotter(interactive=False)
        self.plt.add_button(self.__capture, states=['Capture'], c=['w'], bc=['r3'])
        self.plt.render().add(self.mesh, self.visible_points)
        self.plt.show()
        self.plt.interactive()
        self.plt.close()

    def __capture(self, obj, ename):
        self.visible_points_extractor.set_camera(camera_position=self.plt.camera.GetPosition(),
                                                 focal_point=self.plt.camera.GetFocalPoint(),
                                                 view_angle=self.plt.camera.GetViewAngle())
        pcd, _ = self.visible_points_extractor.extract()
        self.plt.remove(self.visible_points)
        self.visible_points = Points(pcd)
        self.plt.add(self.visible_points).render()


Scene()
