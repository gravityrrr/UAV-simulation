import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout

class TrainingPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Style
        self.figure.patch.set_facecolor('#1e1e1e')
        
        self.ax_reward = self.figure.add_subplot(311)
        self.ax_loss = self.figure.add_subplot(312)
        self.ax_success = self.figure.add_subplot(313)
        
        self._format_ax(self.ax_reward, "Episode Reward")
        self._format_ax(self.ax_loss, "Training Loss")
        self._format_ax(self.ax_success, "Success Rate")
        
        self.figure.tight_layout()
        
        self.rewards = []
        self.losses = []
        self.successes = []
        
    def _format_ax(self, ax, title):
        ax.set_title(title, color='white')
        ax.set_facecolor('#2e2e2e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#555555')
            
    def update_plots(self, reward, loss, success):
        self.rewards.append(reward)
        self.losses.append(loss)
        self.successes.append(success)
        
        self.ax_reward.clear()
        self.ax_loss.clear()
        self.ax_success.clear()
        
        self._format_ax(self.ax_reward, "Episode Reward")
        self._format_ax(self.ax_loss, "Training Loss")
        self._format_ax(self.ax_success, "Success Rate (%)")
        
        self.ax_reward.plot(self.rewards, color='#00ffcc')
        self.ax_loss.plot(self.losses, color='#ff3366')
        
        # smooth success rate
        smooth_success = [sum(self.successes[max(0, i-10):i+1])/len(self.successes[max(0, i-10):i+1])*100 for i in range(len(self.successes))]
        self.ax_success.plot(smooth_success, color='#ffcc00')
        self.ax_success.set_ylim(0, 100)
        
        self.canvas.draw()
