from tvtk.api import tvtk
from mayavi import mlab
import numpy as np
from scipy.integrate import odeint
from tvtk.api import write_data
from os import listdir, mkdir, remove
from os.path import isfile, join, exists
from PIL import Image
from moviepy import editor
from traits.api import HasTraits, Range, Instance, Enum, observe
from traitsui.api import View, Item, Group, HSplit
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
    MlabSceneModel
from traitsui.api import CancelButton, OKButton
from fft import fft_1d, fft2d


class plot(HasTraits):

    perc_coeffs = Range(0, 100, 5, desc='percent of coeff',
                        enter_set=True, auto_set=False)
    display = Enum('Sinc', 'Cone', 'Parabola')

    scene = Instance(MlabSceneModel, args=())

    function = np.empty()

    view = View(HSplit(
        Group(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
              height=250, width=300, show_label=False)),
        Group(
            Item('perc_coeffs'),
            Item('display')
        ),

    ), buttons=[OKButton, CancelButton]
    )

    def disp():
        pass

    @observe('perc_coeffs, display, scene.activated')
    def update_plot(self, event=None):

        self.disp()


def main():
    x, y = np.mgrid[-20:20:256 * 1j, -20:20:256 * 1j]
    f = np.sinc(x*y/70)*10
    f = np.sinc(x/5)*np.sinc(y/5)*10
    c = np.fft.fft2(f)
    c[:, 64:] = 0
    c[64:, :] = 0
    f2 = np.real(np.fft.ifft2(c))
    mlab.surf(x, y, f2)
    mlab.show()
    return


if __name__ == '__main__':
    main()
