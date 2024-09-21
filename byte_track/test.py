from typing import List, Union
from ultralytics import YOLO
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import copy

class DetObjType(Enum):
    Ball = 0
    Goalkeeper = 1
    Player = 2
    Referee = 3

class DetectionObj(object):
    def __init__(
            self, 
            bbox: np.ndarray,
            cls: DetObjType, 
            conf: float
        ):
        self.bbox: np.ndarray = bbox
        self.loc: np.ndarray = np.array([bbox[0], bbox[1]])
        self.cls: DetObjType = cls
        self.conf: float = conf

    @classmethod
    def clone(cls, instance):
        return DetectionObj(
            instance.bbox,
            instance.cls,
            instance.conf
        )

class Detector(object):
    def __init__(
        self,
        model: YOLO
    ):
        self.model: YOLO = model

    @staticmethod
    def compute_iou(det_obj1: DetectionObj, det_obj2: DetectionObj):
        box1 = det_obj1.bbox
        box2 = det_obj2.bbox

        # Transform to xyxy
        rect1 = np.array([
            box1[0] - box1[2] / 2,
            box1[1] - box1[3] / 2,
            box1[0] + box1[2] / 2,
            box1[1] + box1[3] / 2
        ])

        rect2 = np.array([
            box2[0] - box2[2] / 2,
            box2[1] - box2[3] / 2,
            box2[0] + box2[2] / 2,
            box2[1] + box2[3] / 2
        ])

        res = 0
        x_l = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
        y_l = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])

        if x_l > 0 and y_l > 0:
            int_area = (min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) * (min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
            union_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) + (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]) - int_area

            res = int_area / union_area

        return res
    
    @staticmethod
    def non_max_suppression(det_objs: List[DetectionObj], threshold: float):
        det_objs = sorted(det_objs, key=lambda x : x.conf, reverse=True)
        print([det_obj.conf for det_obj in det_objs])
        
        selected_objs: List[DetectionObj] = []

        while len(det_objs) > 0:
            # Pick the box with the highest score
            selected_obj = det_objs.pop(0)
            selected_objs.append(selected_obj)
            
            if len(det_objs) == 0: break

            # Compute IoU of the current box with the remaining boxes
            ious = np.array([Detector.compute_iou(selected_obj, det_obj) for det_obj in det_objs])

            # Only keep boxes with IoU less than the threshold
            temp = [det_objs[i] for i in range(len(det_objs)) if ious[i] < threshold]
            det_objs = temp

        return selected_objs
        
    def process(self, frame: np.ndarray):
        result = self.model.predict(frame)[0]
        boxes = result.boxes
        predicted_classes = boxes.cls
        predicted_confidence = boxes.conf

        det_objs: List[DetectionObj] = []
        for box, cls, conf in zip(result.boxes.xywhn, predicted_classes, predicted_confidence):
            box = np.array(box, dtype=np.float32)
            cls = DetObjType(int(cls))
            conf = float(conf)
            
            det_objs.append(DetectionObj(box, cls, conf))
        
        return result, det_objs
        

class TrackerObj(object):
    track_id_static = 1

    def __init__(self, det_obj: DetectionObj):
        self.track_id = TrackerObj.track_id_static
        TrackerObj.track_id_static += 1

        self.path: List[DetectionObj] = [det_obj]
        self.predicted_path: List[DetectionObj] = [det_obj]
        self.measured_path: List[DetectionObj] = [det_obj]
        self.matched = True

        # Set up kalman filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],   # dt = 1 here, can be tweaked
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[self.path[-1].loc[0]], [self.path[-1].loc[1]], [0], [0]], dtype=np.float32)

    def predict(self):
        prediction = self.kf.predict()

        det_obj = DetectionObj.clone(self.path[-1])
        det_obj.loc = np.matmul(prediction.T, self.kf.measurementMatrix.T)[0]
        det_obj.bbox[0] = det_obj.loc[0]
        det_obj.bbox[1] = det_obj.loc[1] 

        self.predicted_path.append(det_obj)

    def update(self, det_obj: DetectionObj = None):
        """
        For estimation step.
        Updates the location and bbox of the TrackerObj with
        a measurement if it exists, otherwise with a prediction
        """
        self.measured_path.append(DetectionObj.clone(det_obj))

        measurement_mat = np.array([[np.float32(det_obj.loc[0])], [np.float32(det_obj.loc[1])]], dtype=np.float32)
        self.kf.correct(measurement_mat)

        updated_state = self.kf.statePost

        det_obj = DetectionObj.clone(det_obj)
        det_obj.loc = np.matmul(updated_state.T, self.kf.measurementMatrix.T)[0]
        det_obj.bbox[0] = det_obj.loc[0]
        det_obj.bbox[1] = det_obj.loc[1]

        self.path.append(det_obj)

class Tracker:
    def __init__(
            self, 
            cls: DetObjType,
            conf_threshold_high: float,
            conf_threshold_low: float,
            max_distance_normalised: float,
            *, 
            max_objects: Union[int, None]=None
        ):
        self.cls = cls
        self.conf_threshold_high: float = conf_threshold_high
        self.conf_threshold_low: float = conf_threshold_low 
        self.max_distance_normalised: float = max_distance_normalised   # Will need to be reprojected distance in the future

        self.max_objects: Union[int, None] = max_objects

        self.tracker_objs: List[TrackerObj] = []

    def process(self, det_objs: List[DetectionObj]):
        """
        For association and estimation step.
        Two-layer association (ByteTrack)

        1) Perform association for detections with conf > conf_threshold_high
            with trackers. Set aside unassigned trackers and detections
        2) Perform association for remaining detections with conf > conf_threshold_lod
            with remaining trackers.
        3) Remaining trackers considered as MISSING
        4) Remaining detections with conf > conf_threshold_high will be used
            to generate new trackers

        Threshold will need to be adjusted according to class labels
        - Balls are small but fast. Score generally low. Maximum one ball should exist.
        - There should be a maximum number of tracked persons of each class.
        """
        for tracker in self.tracker_objs:
            # Perform prediction for all existing trackers
            tracker.predict()

            # Set all trackers to unmatched
            tracker.matched = False

        # Remove all irrelevant detection objects
        # Remember lists are passed by reference, so need to create new list to store
        filtered_det_objs = [det_obj for det_obj in det_objs if det_obj.cls == self.cls]

        ### FIRST ASSOCIATION ###
        high_conf_det_objs: List[DetectionObj] = [det_obj for det_obj in filtered_det_objs if det_obj.conf >= self.conf_threshold_high]

        # Skip if no high conf detection objects
        if len(high_conf_det_objs) == 0:
            pass

        # If no existing trackers, create them using the detection list
        elif len(self.tracker_objs) == 0:
            for det_obj in high_conf_det_objs:
                self.tracker_objs.append(TrackerObj(det_obj))

        # Association using squared distance, not IOU due to small and fast objects
        else:
            cost_matrix = np.zeros((len(self.tracker_objs), len(high_conf_det_objs)))
            for i, tracker in enumerate(self.tracker_objs):
                for j, det_obj in enumerate(high_conf_det_objs):
                    cost_matrix[i, j] = np.sum((tracker.predicted_path[-1].loc - det_obj.loc) ** 2)

            tracker_indices, det_obj_indices = linear_sum_assignment(cost_matrix)
            tracker_indices = tracker_indices.tolist()
            det_obj_indices = det_obj_indices.tolist()

            # Reverse iteration to prevent iteration invalidation
            for k in range(len(tracker_indices))[::-1]:
                i = tracker_indices[k]
                j = det_obj_indices[k]
                # Pop from list for invalid matches (exceed max distance)
                if cost_matrix[i, j] >= self.max_distance_normalised ** 2:
                    tracker_indices.pop(k)
                    det_obj_indices.pop(k)
                # Move if it is a valid match
                else:
                    self.tracker_objs[i].update(high_conf_det_objs[j])
                    self.tracker_objs[i].matched = True
            
            # Create new trackers for unmatched high confidence detection objects
            for j, det_obj in enumerate(high_conf_det_objs):
                if j not in det_obj_indices:
                    self.tracker_objs.append(TrackerObj(det_obj))
                    
        ### SECOND ASSOCIATION ###
        unmatched_tracker_indices = [i for i in range(len(self.tracker_objs)) if not self.tracker_objs[i].matched]
        low_conf_det_objs = [det_obj for det_obj in filtered_det_objs if self.conf_threshold_low <= det_obj.conf < self.conf_threshold_high]
        
        # Skip if no low conf detection objects
        if len(low_conf_det_objs) == 0:
            pass

        # Skip if no unmatched trackers
        elif len(unmatched_tracker_indices) == 0:
            pass

        # Association using squared distance, not IOU due to small and fast objects
        else:
            cost_matrix = np.zeros((len(unmatched_tracker_indices), len(low_conf_det_objs)))
            for i, tracker_idx in enumerate(unmatched_tracker_indices):
                for j, det_obj in enumerate(low_conf_det_objs):
                    cost_matrix[i, j] = np.sum((self.tracker_objs[tracker_idx].predicted_path[-1].loc - det_obj.loc) ** 2)

            tracker_indices, det_obj_indices = linear_sum_assignment(cost_matrix)
            tracker_indices = tracker_indices.tolist()
            det_obj_indices = det_obj_indices.tolist()

            # Reverse iteration to prevent iteration invalidation
            for k in range(len(tracker_indices))[::-1]:
                i = tracker_indices[k]
                j = det_obj_indices[k]
                # Pop from list for invalid matches (exceed max distance)
                if cost_matrix[i, j] >= self.max_distance_normalised ** 2:
                    tracker_indices.pop(k)
                    det_obj_indices.pop(k)
                # Move if it is a valid match
                else:
                    tracker_idx = unmatched_tracker_indices[i]
                    self.tracker_objs[tracker_idx].update(low_conf_det_objs[j])
                    self.tracker_objs[tracker_idx].matched = True

        # Remove unmatched trackers
        for i in range(len(self.tracker_objs))[::-1]:
            if not self.tracker_objs[i].matched:
                self.tracker_objs.pop(i)

def main():
    video_path = "videos/bundes_football.mp4"

    cap = cv2.VideoCapture(video_path)
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # For detection
    detector = Detector(YOLO("models/yolov8/best.pt"))
    nms_iou_thresh = 0.5

    # For tracking
    trackers: List[Tracker] = [
        Tracker(DetObjType.Ball, 0.3, 0.1, 0.2), # Ball is hard to track
        Tracker(DetObjType.Player, 0.7, 0.4, 0.05),
        Tracker(DetObjType.Goalkeeper, 0.7, 0.4, 0.05),
        Tracker(DetObjType.Referee, 0.7, 0.4, 0.05)
    ]

    # For display
    scaled_width = int(original_width * 0.5)
    scaled_height = int(original_height * 0.5)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: break

        ### 1 DETECTION ###
        results, det_objs = detector.process(frame)
        det_objs = detector.non_max_suppression(det_objs, nms_iou_thresh)

        ### 2 TRACKING ###
        for tracker in trackers:
            tracker.process(det_objs)
            
        # Display
        output_frame = results.plot()

        blue = (255, 0, 0)
        green = (0, 255, 0)
        red = (0, 0, 255)
        for det_obj in det_objs:
            p = np.array([det_obj.loc[0] * output_frame.shape[1], det_obj.loc[1] * output_frame.shape[0]]).astype(np.int32)
            output_frame = cv2.circle(output_frame, p, 10, blue, -1)

        for tracker in trackers:
            for tracker_obj in tracker.tracker_objs:
                p = np.array(tracker_obj.path[-1].loc * output_frame.shape[:2][::-1]).astype(np.int32)
                print(p)
                output_frame = cv2.circle(output_frame, p, 10, red, -1)
                output_frame = cv2.putText(output_frame, str(tracker_obj.track_id), p + np.array([10, 10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                ## Draw paths
                # Measured path
                path = [(det_obj.loc * output_frame.shape[:2][::-1]) for det_obj in tracker_obj.measured_path]
                path = np.array(path).astype(np.int32).reshape((-1, 1, 2))
                output_frame = cv2.polylines(
                    output_frame,
                    [path],
                    False, red, 3
                )
                
                # Predicted path
                path = [(det_obj.loc * output_frame.shape[:2][::-1]) for det_obj in tracker_obj.predicted_path]
                path = np.array(path).astype(np.int32).reshape((-1, 1, 2))
                output_frame = cv2.polylines(
                    output_frame,
                    [path],
                    False, blue, 3
                )

                # Estimated path
                path = [(det_obj.loc * output_frame.shape[:2][::-1]) for det_obj in tracker_obj.path]
                path = np.array(path).astype(np.int32).reshape((-1, 1, 2))
                output_frame = cv2.polylines(
                    output_frame,
                    [path],
                    False, green, 3
                )

        resized_frame = cv2.resize(output_frame, (scaled_width, scaled_height))
        cv2.imshow("Output", resized_frame)

        if cv2.waitKey() & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()