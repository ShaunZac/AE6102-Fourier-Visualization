from mayavi import mlab
import numpy as np
from traits.api import HasTraits, Range, Instance, Enum, observe
from traitsui.api import View, Item, Group, HSplit
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
    MlabSceneModel
from traitsui.api import CancelButton, OKButton
from fft import fft2d


class plot(HasTraits):

    perc_coeffs = Range(1, 100, 100, desc='percent of coeff',
                        enter_set=True, auto_set=False)
    display = Enum('Sinc', 'Cone', 'Parabola')

    scene = Instance(MlabSceneModel, args=())

    func = np.empty((1, 1))
    x, y = np.mgrid[-20:20:256 * 1j, -20:20:256 * 1j]

    f_coeffs = np.empty((1, 1))
    f2 = np.empty((1, 1))

    view = View(HSplit(
        Group(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
              height=250, width=300, show_label=False)),
        Group(
            Item('perc_coeffs'),
            Item('display')
        ),

    ), buttons=[OKButton, CancelButton]
    )

    def disp(self):
        mlab.clf()
        x = self.x
        y = self.y

        if self.display == 'Sinc':
            self.function = np.sinc(x/5)*np.sinc(y/5)*50
        elif self.display == 'Cone':
            self.function = 5 * np.sqrt(x ** 2 + y ** 2)
        else:
            self.function = y ** 2 - 5 * x

        self.f_coeffs = np.fft.fft2(self.function)

        self.update()
        return

    def update(self):
        mlab.clf()
        x = self.x
        y = self.y
        x_idx = int(round(self.perc_coeffs * len(x) / 100))
        y_idx = int(round(self.perc_coeffs * len(y) / 100))
        f_coeffs = self.f_coeffs.copy()
        f_coeffs[:, y_idx:] = 0
        f_coeffs[x_idx:, :] = 0
        f2 = np.real(np.fft.ifft2(f_coeffs))
        mlab.surf(x, y, f2)
        return

    @observe('display, scene.activated')
    def show_plot(self, event=None):
        self.scene.mlab.clf()
        self.disp()
        self.scene.mlab.surf(self.x, self.y, self.func)
        # self.scene.mlab.show()

    @observe('perc_coeffs')
    def update_plot(self, event=None):
        self.scene.mlab.clf()
        self.update()
        self.scene.mlab.surf(self.x, self.y, self.f2)
        # self.scene.mlab.show()


def main():
    # x, y = np.mgrid[-20:20:256 * 1j, -20:20:256 * 1j]
    # f = np.sinc(x*y/70)*10
    # f = np.sinc(x/5)*np.sinc(y/5)*10
    # c = np.fft.fft2(f)
    # c[:, 64:] = 0
    # c[64:, :] = 0
    # f2 = np.real(np.fft.ifft2(c))
    # mlab.surf(x, y, f2)
    # mlab.show()
    p = plot()
    p.configure_traits()
    return


if __name__ == '__main__':
    main()
