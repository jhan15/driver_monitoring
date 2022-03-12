import cv2
import mediapipe as mp
import numpy as np
import time
import facial_tracking.conf as conf


class FaceMesh:
    """
    Face mesh of a given frame using mediapipe.

    Args:
        max_num_faces (int): maximum number of faces to detect
        refine_landmarks (bool): whether to further refine the landmark coordinates around the eyes and lips,
            and output additional landmarks around the irises by applying the Attention Mesh Model.
        min_detection_confidence (float): minimum confidence value ([0.0, 1.0]) from the face detection
            model for the detection to be considered successful.
        min_tracking_confidence (float): minimum confidence value ([0.0, 1.0]) from the landmark-tracking model
            for the face landmarks to be considered tracked successfully, or otherwise face detection will be
            invoked automatically on the next input image. 
    """

    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.frame = None
        self.mesh_result = None

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence)
        
        # self._get_target_landmarks()
    
    def _get_target_landmarks(self):
        """Get landmarks of eyes, irises, and lips."""
        self.landmark_left_eye   = np.unique(np.array(list(self.mp_face_mesh.FACEMESH_LEFT_EYE)))
        self.landmark_right_eye  = np.unique(np.array(list(self.mp_face_mesh.FACEMESH_RIGHT_EYE)))
        self.landmark_left_iris  = np.unique(np.array(list(self.mp_face_mesh.FACEMESH_LEFT_IRIS)))
        self.landmark_right_iris = np.unique(np.array(list(self.mp_face_mesh.FACEMESH_RIGHT_IRIS)))
        self.landmark_lips       = np.unique(np.array(list(self.mp_face_mesh.FACEMESH_LIPS)))
        
    def process_frame(self, frame):
        """The function to mesh the frame."""
        self.frame = frame
        self._face_mesh()

    def _face_mesh(self):
        """Call the mediapipe face_mesh processor."""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.mesh_result = self.face_mesh.process(frame)
    
    def draw_mesh(self):
        """Draw the mesh result by mediapipe face_mesh processor."""

        # Possible self.mp_face_mesh.FACEMESH_<TYPES>:
        #     CONTOURS
        #     FACE_OVAL
        #     IRISES
        #     LEFT_EYE
        #     LEFT_EYEBROW
        #     LEFT_IRIS
        #     LIPS
        #     NUM_LANDMARKS
        #     NUM_LANDMARKS_WITH_IRISES
        #     RIGHT_EYE
        #     RIGHT_EYEBROW
        #     RIGHT_IRIS
        #     TESSELATION

        if self.mesh_result.multi_face_landmarks:
            for face_landmarks in self.mesh_result.multi_face_landmarks:
                # Draw mesh in tesselation format
                self.mp_drawing.draw_landmarks(
                    image=self.frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                
                # Draw the contours of face, eyes, eyebrows, and lips
                self.mp_drawing.draw_landmarks(
                    image=self.frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                
                # Draw the contours of irises
                self.mp_drawing.draw_landmarks(
                    image=self.frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
    
    def draw_mesh_eyes(self):
        """Draw the mesh of eyes."""
        if self.mesh_result.multi_face_landmarks:
            for face_landmarks in self.mesh_result.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=self.frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=conf.CT_COLOR, thickness=1, circle_radius=1))
                self.mp_drawing.draw_landmarks(
                    image=self.frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=conf.CT_COLOR, thickness=1, circle_radius=1))
    
    def draw_mesh_lips(self):
        """Draw the mesh of lips."""
        if self.mesh_result.multi_face_landmarks:
            for face_landmarks in self.mesh_result.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=self.frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=conf.CT_COLOR, thickness=1, circle_radius=1))

def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    fm = FaceMesh()
    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        fm.process_frame(frame)
        fm.draw_mesh()

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Face Mesh', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
