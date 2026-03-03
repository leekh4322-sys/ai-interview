import threading
import queue
import time
from modules.pose.pose_module import PoseAnalyzer
import modules.shared_flags as flags 
from modules.camera.camera_manager import shared_frame_queue

result_queue = queue.Queue(maxsize=5)


def _make_pose_score_feedback(motion: float):
    try:
        score = max(0.0, min(100.0, 100.0 - (float(motion) / 2.0) * 100.0))
    except:
        score = 0.0

    feedback = (
        "자세가 비교적 안정적입니다. 현재 자세를 유지해보세요."
        if score >= 70
        else "움직임이 많아 불안해 보일 수 있습니다. 상체를 조금 더 고정해보세요."
    )
    return float(score), feedback


def pose_worker():
    analyzer = PoseAnalyzer()
    print("💪 Pose Thread Started", flush=True)

    scores = []
    last_feedback = ""

    while flags.RUNNING:
        if shared_frame_queue.empty():
            time.sleep(0.001)
            continue

        frame = shared_frame_queue.get()
        processed_frame, motion, coords = analyzer.process_frame(frame)

        # ✅ 스레드 내부 점수화/피드백
        score, feedback = _make_pose_score_feedback(motion)
        scores.append(score)
        last_feedback = feedback

        # ✅ main.py가 그대로 받도록 tuple 유지
        result = (processed_frame, motion, coords)

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except:
                pass
        result_queue.put(result)

    # 종료 시 최종 출력
    final_avg = (sum(scores) / len(scores)) if scores else 0.0
    final_fb = last_feedback or "자세 피드백 없음"

    print("\n==============================", flush=True)
    print("      🧍 POSE 최종 결과", flush=True)
    print("==============================", flush=True)
    print(f"자세 평균 점수 : {final_avg:.1f} 점", flush=True)
    print("자세 피드백:", flush=True)
    print(f"- {final_fb}", flush=True)
    print("==============================\n", flush=True)

    print("💪 Pose Thread Stopped", flush=True)


def start_pose_thread():
    t_pose = threading.Thread(target=pose_worker, daemon=False)
    t_pose.start()
    print("🚀 pose_thread_example 실행됨! (Camera 공유 버전)", flush=True)
    return t_pose