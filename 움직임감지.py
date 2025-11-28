import cv2
from ultralytics import YOLO
model = YOLO("yolov8m-pose.pt")
from ultralytics.utils.plotting import Annotator




def predict(frame, iou=0.7, conf=0.25):
    results = model(source=frame,
            device='cpu',
            iou=iou ,
            conf=conf ,
            verbose=False,
            )
    return results[0]




def draw_boxes(result, frame):
    for boxes in result.boxes:
        x1, y1, x2, y2, score, classes = boxes.data.squeeze().cpu().numpy()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    return frame


def draw_keypoints(result, frame):
    annotator = Annotator(frame, line_width=1)
    for kps in result.keypoints:
        kps = kps.data.squeeze()
        annotator.kpts(kps)
       
        nkps = kps.cpu().numpy()
        # nkps[:,2] = 1
        # annotator.kpts(nkps)
        for idx, (x, y, score) in enumerate(nkps):
            if score > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
       
    return frame


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = capture.read()
        result = predict(frame)
        frame = draw_boxes(result, frame)
        frame = draw_keypoints(result, frame)
        frame = cv2.flip(frame, 1)




        # cv2.putText(frame, text, position, font, scale, color, thickness)
        if not ret:
            print("카메라 오류")
            break
        # print(type(frame))
        cv2.imshow("VideoFrame", frame)




        if cv2.waitKey(10) & 0xFF == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            break
