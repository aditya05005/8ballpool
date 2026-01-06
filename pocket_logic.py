import numpy as np
from collections import Counter

class PocketDetector:
    def __init__(self, pocket_coords, pocket_radius=30, cooldown_frames=30):
        self.pockets = np.array(pocket_coords)
        self.radius = pocket_radius
        self.cooldown_limit = cooldown_frames
        
        self.total_pocketed_count = 0
        self.pocket_cooldowns = [0] * len(pocket_coords)
        self.ball_in_zone_last_frame = [False] * len(pocket_coords)
        
        # MEMORY: Stores the class IDs seen in each pocket zone recently
        # Structure: [ [class_id, class_id...], [], [], ... ] for 6 pockets
        self.pocket_memory = [[] for _ in range(len(pocket_coords))]

    def update(self, ball_detections, class_ids):
        """
        Args:
            ball_detections: numpy array of [x, y]
            class_ids: numpy array of class integers (0, 1, 2, etc.)
        """
        events = []
        
        for i, pocket in enumerate(self.pockets):
            # 1. Manage Cooldown
            if self.pocket_cooldowns[i] > 0:
                self.pocket_cooldowns[i] -= 1
                continue 

            # 2. Check current zone status
            ball_in_zone_now = False
            
            if len(ball_detections) > 0:
                # Find distance to all balls
                dists = np.linalg.norm(ball_detections - pocket, axis=1)
                min_dist_idx = np.argmin(dists)
                
                # If the closest ball is inside the radius...
                if dists[min_dist_idx] < self.radius:
                    ball_in_zone_now = True
                    
                    # --- CRITICAL NEW STEP: RECORD THE IDENTITY ---
                    # We add the class of this specific ball to our memory
                    detected_class = int(class_ids[min_dist_idx])
                    self.pocket_memory[i].append(detected_class)
                    
                    # Keep memory short (last 10 frames only) to avoid old data
                    if len(self.pocket_memory[i]) > 10:
                        self.pocket_memory[i].pop(0)

            # 3. Logic: Transition from Present -> Absent
            if self.ball_in_zone_last_frame[i] and not ball_in_zone_now:
                
                # A ball just fell in! Who was it?
                if self.pocket_memory[i]:
                    # Get the most common class in the memory (Majority Vote)
                    # This fixes flickering errors
                    most_common_class = Counter(self.pocket_memory[i]).most_common(1)[0][0]
                    
                    self.total_pocketed_count += 1
                    self.pocket_cooldowns[i] = self.cooldown_limit
                    
                    # Return both the pocket index AND the identified class
                    events.append((i, most_common_class))
                    
                    # Clear memory for next time
                    self.pocket_memory[i] = []
                
            # Update history
            self.ball_in_zone_last_frame[i] = ball_in_zone_now

        return self.total_pocketed_count, events
