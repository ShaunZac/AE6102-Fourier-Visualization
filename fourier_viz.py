from mayavi import mlab
import numpy as np
from traits.api import HasTraits, Range, Instance, Enum, observe
from traitsui.api import View, Item, Group, HSplit
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
    MlabSceneModel
from traitsui.api import CancelButton, OKButton
from fft import fft2d


class plot(HasTraits):

    perc_coeffs = Range(1, 100, 90, desc='percent of coeff',
                        enter_set=True, auto_set=False)
    display = Enum('Sinc', 'Cone', 'Parabola', desc='function')
    fft_pts = Enum('32', '64', '128', '256', '512', '1024')
    xrange = Range(1, 100, 20, desc='x Range',
                   enter_set=True, auto_set=False)
    yrange = Range(1, 100, 20, desc='y Range',
                   enter_set=True, auto_set=False)

    pts = 256

    if fft_pts == '32':
        pts = 32
    elif fft_pts == '64':
        pts = 64
    elif fft_pts == '128':
        pts = 128
    elif fft_pts == '256':
        pts = 256
    elif fft_pts == '512':
        pts = 512
    elif fft_pts == '1024':
        pts = 1024

    scene = Instance(MlabSceneModel, args=())

    func = np.empty((1, 1))
    # x, y = np.mgrid[-1*int(self.xrange):int(self.xrange):int(pts)
    #                 * 1j, -1*int(self.yrange):int(self.yrange):int(pts) * 1j]
    x = np.empty((1, 1))
    y = np.empty((1, 1))

    f_coeffs = np.empty((1, 1))
    f2 = np.empty((1, 1))

    view = View(HSplit(
        Group(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
              height=250, width=300, show_label=False)),
        Group(
            Item('perc_coeffs'),
            Item('display'),
            Item('fft_pts'),
            Item('xrange'),
            Item('yrange')
        ),

    ), buttons=[OKButton, CancelButton]
    )

    def axis_mod(self):
        if self.fft_pts == '32':
            self.pts = 32
        elif self.fft_pts == '64':
            self.pts = 64
        elif self.fft_pts == '128':
            self.pts = 128
        elif self.fft_pts == '256':
            self.pts = 256
        elif self.fft_pts == '512':
            self.pts = 512
        elif self.fft_pts == '1024':
            self.pts = 1024

        self.x, self.y = np.mgrid[-1 *
                                  self.xrange:self.xrange:int(self.pts) * 1j, -1*self.yrange:self.yrange:int(self.pts) * 1j]

    def disp(self):
        mlab.clf()
        x, y = self.x, self.y

        if self.display == 'Sinc':
            self.function = np.sinc(x/5)*np.sinc(y/5)*50
        elif self.display == 'Cone':
            self.function = 5 * np.sqrt(x ** 2 + y ** 2)
        else:
            self.function = y ** 2 - 5 * x

        self.f_coeffs = fft2d(self.function)

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
        mlab.outline()
        return

    @observe('fft_pts, xrange, yrange, scene.activated')
    def change_axis(self, event=None):
        self.scene.mlab.clf()
        self.axis_mod()
        self.disp()

    @observe('display')
    def show_plot(self, event=None):
        self.scene.mlab.clf()
        self.disp()

    @observe('perc_coeffs')
    def update_plot(self, event=None):
        self.scene.mlab.clf()
        self.update()


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
