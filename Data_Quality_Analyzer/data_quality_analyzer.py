# Structure of the Data Quality Analyzer (DQA) module:

# 1. Load Input Data
# 2. Run FISCAS on Input Data (fiscas.py) -> Single-Cell Spectra, Segmentation 
# 3. Estimate diffusion parameters based on MSI data and segmentation results (diffusion_estimation.py) -> Diffusion Parameters
# 4. Simulate MSI canvas based on segmentation results and diffusion parameters (canvas_generation.py) -> Simulated MSI Canvas
# 5. Compare similarity between simulated MSI canvas and original MSI data
# 6. Calculate single-cell spectra (fiscas.py) of simulated MSI canvas and calculate error to single-cell spectra of original MSI data

from fiscas import load_imzml
import skimage
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
input_dir = os.path.join(base_dir, "input")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

msi_file = os.path.join(input_dir, "msi_data_1.imzML")
mic_file = os.path.join(input_dir, "mic_data_1.tif")
feature_list_file = os.path.join(input_dir, "feature_list_1.csv")

msi_data, feature_list = load_imzml(msi_file, feature_list_file)
mic_data = skimage.io.imread(mic_file)


import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase, ToolToggleBase, Cursors, ToolSetCursor


plt.rcParams['toolbar'] = 'toolmanager'

class PickCoords(ToolToggleBase):
    default_keymap = 'p'
    description = 'pick landmarks'
    default_toggled = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def enable(self, *args):
        self.cursor = Cursors.SELECT_REGION

    def disable(self, *args):
        print("disable")
        

class ListTools(ToolBase):
    """List all the tools controlled by the `ToolManager`."""
    default_keymap = 'm'  # keyboard shortcut
    description = 'List Tools'

    def trigger(self, *args, **kwargs):
        print('_' * 80)
        fmt_tool = "{:12} {:45} {}".format
        print(fmt_tool('Name (id)', 'Tool description', 'Keymap'))
        print('-' * 80)
        tools = self.toolmanager.tools
        for name in sorted(tools):
            if not tools[name].description:
                continue
            keys = ', '.join(sorted(self.toolmanager.get_tool_keymap(name)))
            print(fmt_tool(name, tools[name].description, keys))
        print('_' * 80)
        fmt_active_toggle = "{0!s:12} {1!s:45}".format
        print("Active Toggle tools")
        print(fmt_active_toggle("Group", "Active"))
        print('-' * 80)
        for group, active in self.toolmanager.active_toggle.items():
            print(fmt_active_toggle(group, active))


class GroupHideTool(ToolToggleBase):
    """Show lines with a given gid."""
    default_keymap = 'S'
    description = 'Show by gid'
    default_toggled = True

    def __init__(self, *args, gid, **kwargs):
        self.gid = gid
        super().__init__(*args, **kwargs)

    def enable(self, *args):
        self.set_lines_visibility(True)

    def disable(self, *args):
        self.set_lines_visibility(False)

    def set_lines_visibility(self, state):
        for ax in self.figure.get_axes():
            for line in ax.get_lines():
                if line.get_gid() == self.gid:
                    line.set_visible(state)
        self.figure.canvas.draw()


fig = plt.figure()
plt.plot([1, 2, 3], gid='mygroup')
plt.plot([2, 3, 4], gid='unknown')
plt.plot([3, 2, 1], gid='mygroup')

# Add the custom tools that we created
fig.canvas.manager.toolmanager.add_tool('List', ListTools)
fig.canvas.manager.toolmanager.add_tool('Show', GroupHideTool, gid='mygroup')
fig.canvas.manager.toolmanager.add_tool('Pick', PickCoords)

# Add an existing tool to new group `foo`.
# It can be added as many times as we want
fig.canvas.manager.toolbar.add_tool('zoom', 'foo')

# Remove the forward button
fig.canvas.manager.toolmanager.remove_tool('forward')

# To add a custom tool to the toolbar at specific location inside
# the navigation group
fig.canvas.manager.toolbar.add_tool('Show', 'navigation', 1)
fig.canvas.manager.toolbar.add_tool('Pick', 'navigation', 2)    

plt.show()
