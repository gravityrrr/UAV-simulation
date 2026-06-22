import yaml
import time
from gui.main_window import TrainingWorker
from PyQt6.QtWidgets import QApplication
import sys

def test_run():
    app = QApplication(sys.argv)
    
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    config['training']['max_episodes'] = 2
    
    worker = TrainingWorker(config, "PPO")
    worker.start()
    
    while worker.isRunning():
        time.sleep(0.1)
        
    print("PPO finished")
    
    worker2 = TrainingWorker(config, "SAC")
    worker2.start()
    
    while worker2.isRunning():
        time.sleep(0.1)
        
    print("SAC finished")

if __name__ == "__main__":
    test_run()
