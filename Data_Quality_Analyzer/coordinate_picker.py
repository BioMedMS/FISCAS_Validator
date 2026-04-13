import matplotlib as mpl
from matplotlib.backend_tools import ToolToggleBase
import matplotlib.pyplot as plt
import numpy as np

class PickCoords(ToolToggleBase):
    default_keymap = 'y'
    description = 'pick landmarks'
    default_toggled = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a dictionary to store lists for specific axes
        self.coords_dict = {}
        self.background = None

    def enable(self, *args):
        tm = self.figure.canvas.manager.toolmanager
        for tool_name in ['pan', 'zoom']:  # Disable pan when selection mode is on
            if tool_name in tm.tools:
                t = tm.get_tool(tool_name)
                if t.toggled:
                    print(f"Disabling {tool_name} tool.")
                    tm.trigger_tool(tool_name)    
        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        self.figure.canvas.set_cursor(mpl.backend_tools.Cursors.SELECT_REGION)
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.on_click)

    def disable(self, *args):
        self.figure.canvas.set_cursor(mpl.backend_tools.Cursors.POINTER)
        self.figure.canvas.mpl_disconnect(self.cid)

    def on_click(self, event):
        if event.inaxes and event.button == 1:
            ax = event.inaxes
            label = ax.get_title()
            
            if label not in self.coords_dict:
                self.coords_dict[label] = []
                
            self.coords_dict[label].append((event.xdata, event.ydata))
            
            # Visual feedback: Draw a point where clicked
            self.figure.canvas.restore_region(self.background)

            for active_label, pts in self.coords_dict.items():
                # Find the axis matching this label
                target_ax = [a for a in self.figure.axes if a.get_title() == active_label][0]
                for px, py in pts:
                    # Draw points directly to the axis
                    dot = target_ax.plot(px, py, 'x', markersize=8, color='red')[0]
                    target_ax.draw_artist(dot)

            # 3. Quickly push the updated pixels to the screen
            self.figure.canvas.blit(self.figure.bbox)

def pick_coordinates(ms_img, mic_img):
    # --- Setup the Plot with Two Subplots ---
    plt.rcParams['toolbar'] = 'toolmanager'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display MSI data
    ax1.imshow(ms_img, cmap='viridis', vmax=np.percentile(ms_img, 99))
    ax1.set_title("MSI_Data")

    # Display Microscopy data
    ax2.imshow(mic_img, cmap='gray', vmax=np.percentile(mic_img, 99))
    ax2.set_title("Microscopy_Data")

    # Register the Tool
    tm = fig.canvas.manager.toolmanager
    tm.add_tool('PickCoords', PickCoords)
    fig.canvas.manager.toolbar.add_tool('PickCoords', 'navigation')

    plt.tight_layout()
    plt.show()

    return tm.get_tool('PickCoords').coords_dict