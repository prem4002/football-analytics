from teams import TeamClassifier
from ultralytics import YOLO
import supervision as sv
import numpy as np
from tqdm import tqdm

class objects:
    """
    purpose: To predict players, ball and referee positions and display

    Inputs:
        trained model
    
    Yields:
        position of players, ball and referee
    """
        def detect():
            #goalkeeper added here in this code, using the previous function
            team_classifier = TeamClassifier(device="cuda")  # or "cpu"
            team_classifier.features_model.eval()

            PLAYER_DETECTION_MODEL = YOLO("/home/prem/Documents/Work/Football/models/detect/football_objects/weights/best.pt")
            SOURCE_VIDEO_PATH = "/home/prem/Documents/Work/Football/videos/121364_0.mp4"
            TARGET_VIDEO_PATH = "/home/prem/Documents/Work/Football/outputs/121364_0_result_bubba.mp4"
            BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID = 0, 1, 2, 3

            # this block crops images and creates the centroids of the cluster ----------------
            init_crops = []
            warmup_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=30)
            for frame in tqdm(warmup_generator, desc="Collecting initial crops"):
                results = PLAYER_DETECTION_MODEL(frame, conf=0.3)
                result = results[0]
                detections = sv.Detections.from_ultralytics(result)
                players_detections = detections[detections.class0-_id == PLAYER_ID]
                init_crops += [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]

            if len(init_crops) > 0:
                team_classifier.fit(init_crops)
            # ---------------------------------------------------------------------------------

            dots_annotator = sv.DotAnnotator(
                color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                radius=6
            )
            triangle_annotator = sv.TriangleAnnotator(
                color=sv.Color.from_hex('#FFD700'),
                base=25,
                height=21,
                outline_thickness=1
            )

            tracker = sv.ByteTrack()
            tracker.reset()

            #copying the information of original video to be applied on the resultant video
            video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
            #video siink allows me to save videos to drive
            video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)
            frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
            frame = next(frame_generator)

            with video_sink:
                for frame in tqdm(frame_generator, total=video_info.total_frames): 
                    results = PLAYER_DETECTION_MODEL(frame, conf=0.3)
                    result = results[0]
                    detections = sv.Detections.from_ultralytics(result)

                    ball_detections = detections[detections.class_id == BALL_ID]
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

                    all_detections = detections[detections.class_id != BALL_ID]
                    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                    all_detections = tracker.update_with_detections(detections=all_detections)

                    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
                    players_detections = all_detections[all_detections.class_id == PLAYER_ID]
                    referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

                    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
                    players_detections.class_id = team_classifier.predict(players_crops)

                    #assigning ID
                    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                        players_detections, goalkeepers_detections)

                    referees_detections.class_id -= 1

                    #adding goalkeeps as a id which requires tracking too and a label
                    all_detections = sv.Detections.merge([
                        players_detections, goalkeepers_detections, referees_detections])

                    labels = [
                        f"#{tracker_id}"
                        for tracker_id
                        in all_detections.tracker_id
                    ]

                    all_detections.class_id = all_detections.class_id.astype(int)

                    annotated_frame = frame.copy()
                    annotated_frame = dots_annotator.annotate(
                        scene=annotated_frame,
                        detections=all_detections)
                    annotated_frame = triangle_annotator.annotate(
                        scene=annotated_frame,
                        detections=ball_detections)
                    video_sink.write_frame(annotated_frame)


        if __name__ == "__main__":
            video()