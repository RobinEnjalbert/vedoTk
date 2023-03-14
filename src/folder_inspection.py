from typing import Optional
from os import getcwd, listdir
from os.path import isfile, abspath, join, sep
from vedo import Plotter, Mesh, Text2D


class FolderInspection:

    def __init__(self,
                 directory: Optional[str] = None,
                 one_by_one: bool = True):

        self.meshes = []
        self.filenames = []
        self.default_alpha = 0. if one_by_one else 0.1
        directory = getcwd() if directory is None else abspath(directory)
        for file in sorted(listdir(directory)):
            if isfile(file := join(directory, file)):
                try:
                    self.meshes.append(Mesh(file, alpha=self.default_alpha, c='y5').linecolor('y2'))
                    self.filenames.append(file.split(sep)[-1])
                except:
                    pass
        if len(self.meshes) == 0:
            raise ValueError("This directory does not contain any mesh file.")
        self.mesh_id = 0
        self.plotter = Plotter().render()
        self.plotter.add(*self.meshes)
        self.meshes[0].alpha(1.)
        self.text = Text2D(txt=self.filenames[0], pos='bottom_middle')
        self.plotter.add(self.text)
        self.plotter.add_button(fnc=self.__previous_file, states=['<'], bc=['red3'], pos=[0.1, 0.05])
        self.plotter.add_button(fnc=self.__next_file, states=['>'], bc=['red3'], pos=[0.9, 0.05])

        self.plotter.show(interactive=True).close()

    def __previous_file(self):
        self.meshes[self.mesh_id].alpha(self.default_alpha)
        self.mesh_id = (self.mesh_id - 1) % len(self.meshes)
        self.meshes[self.mesh_id].alpha(1.)
        self.text.text(self.filenames[self.mesh_id])
        self.plotter.render()

    def __next_file(self):
        self.meshes[self.mesh_id].alpha(self.default_alpha)
        self.mesh_id = (self.mesh_id + 1) % len(self.meshes)
        self.meshes[self.mesh_id].alpha(1.)
        self.text.text(self.filenames[self.mesh_id])
        self.plotter.render()
