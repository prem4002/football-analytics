import supervision as sv

vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    radius = 8
)

FIELD_DETECTION_MODEL = YOLO(home/prem/Documents/Work/Football/models/pose/football_field_keypoints/weights/best.pt)
SOURCE_VIDEO_PATH = '../videos/corner.mp4'

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = FIELD_DETECTION_MODEL(frame, conf=0.3)
key_points = sv.KeyPoints.from_inference(result)

annotated_frame = frame.copy()
annotated_frame = vertex_annotator.annotate(annotated_frame, key_points)

sv.plot_image(annotated_frame)