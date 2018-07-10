import cv2 as cv


def draw_points(frame, poses):
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]

    for pose in poses:
        for i in range(18):
            if i not in pose.body_parts.keys(): continue

            body_part = pose.body_parts[i]
            center = (int(body_part.x * frame.shape[1] + 0.5), int(body_part.y * frame.shape[0] + 0.5))
            cv.circle(frame, center, 3, colors[i], thickness=3, lineType=8, shift=0)
            cv.putText(frame, body_part.uidx, center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame


def draw_fps(frame, fps):
    cv.putText(frame, fps, (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame
