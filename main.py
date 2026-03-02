# main.py (통합본: 질문생성 + 한글출력 + 창안정화 + Voice+Pose 평가)
import time
import cv2
import numpy as np
import os
import platform

# ===============================
# 한글 출력(PIL)
# ===============================
from PIL import ImageFont, ImageDraw, Image

# ===============================
# OPENAI 질문 생성
# ===============================
from modules.question.question_module import make_question

# ===============================
# 공유 RUNNING 플래그
# ===============================
from modules.shared_flags import RUNNING

# ===============================
# 단일 카메라 스레드
# ===============================
from modules.camera.camera_manager import start_camera_thread

# ===============================
# 모듈별 스레드 & 큐
# ===============================
from modules.pose.pose_thread_example import start_pose_thread, result_queue as pose_result_queue
from modules.gaze.gaze_thread_example import start_gaze_thread, gaze_result_queue
from modules.expression.expression_thread_example import start_expression_thread, expression_result_queue
from modules.hands.hand_thread_example import start_hands_thread, hands_result_queue
from modules.voice.voice_thread_example import start_voice_thread, voice_result_queue


# ===============================
# 최신 값만 가져오기
# ===============================
def drain_queue(q):
    latest = None
    while not q.empty():
        latest = q.get()
    return latest


# ===============================
# 한글 폰트 자동 감지 + 캐시 (Windows / macOS만)
# ===============================
FONT_CACHE = {}


def get_korean_font(font_size: int):
    if font_size in FONT_CACHE:
        return FONT_CACHE[font_size]

    system = platform.system()

    if system == "Windows":
        path = "C:/Windows/Fonts/malgun.ttf"
    elif system == "Darwin":
        path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    else:
        path = None  # 리눅스/기타는 불필요(기본 폰트 fallback)

    try:
        if path:
            font = ImageFont.truetype(path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        print("⚠ 한글 폰트 로드 실패 → 기본 폰트 사용")
        font = ImageFont.load_default()

    FONT_CACHE[font_size] = font
    return font


# ===============================
# 한글 텍스트 출력 함수
# ===============================
def put_korean_text(
    img_bgr,
    text,
    x,
    y,
    font_size=28,
    color=(0, 255, 255),
):
    if text is None:
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = get_korean_font(font_size)
    draw.text((x, y), str(text), font=font, fill=(color[2], color[1], color[0]))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ===============================
# 평가 누적 평균(음성+자세)
# ===============================
class RunningAvg:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, x):
        if x is None:
            return self.value()
        self.sum += float(x)
        self.n += 1
        return self.value()

    def value(self):
        return (self.sum / self.n) if self.n > 0 else 0.0


class EvalAggregatorVP:
    def __init__(self, w_voice=0.5, w_pose=0.5):
        self.w_voice = float(w_voice)
        self.w_pose = float(w_pose)
        self.avg_voice = RunningAvg()
        self.avg_pose = RunningAvg()
        self.avg_total = RunningAvg()

    def _to_one_line(self, fb):
        if fb is None:
            return ""
        if isinstance(fb, (list, tuple)):
            items = [str(x) for x in fb if x]
            return " | ".join(items[:2])
        return str(fb)

    def _extract_voice(self, latest_voice):
        # voice_thread는 보통 {"text": "..."} 형태
        if isinstance(latest_voice, dict):
            score = latest_voice.get("score") or latest_voice.get("total_score")
            fb = latest_voice.get("feedback") or latest_voice.get("comment")
            txt = (latest_voice.get("text") or "").strip()

            if score is None:
                score = 80 if len(txt) > 0 else 50

            return float(score), self._to_one_line(fb)

        return None, ""

    def _extract_pose(self, latest_pose):
        if latest_pose is None:
            return None, ""

        # 케이스 A) dict로 온 경우
        if isinstance(latest_pose, dict):
            score = latest_pose.get("score") or latest_pose.get("total_score")
            fb = latest_pose.get("feedback") or latest_pose.get("comment")
            if score is not None:
                return float(score), self._to_one_line(fb)
            return None, self._to_one_line(fb)

        # 케이스 B) tuple: (frame, motion, _)
        try:
            _, motion, _ = latest_pose
            pose_score = max(0.0, min(100.0, 100.0 - (float(motion) / 2.0) * 100.0))
            fb = (
                "자세가 비교적 안정적입니다. 현재 자세를 유지해보세요."
                if pose_score >= 70
                else "움직임이 많아 불안해 보일 수 있습니다. 상체를 조금 더 고정해보세요."
            )
            return pose_score, fb
        except Exception:
            return None, ""

    def update(self, latest_voice, latest_pose):
        voice_score, voice_fb = self._extract_voice(latest_voice)
        pose_score, pose_fb = self._extract_pose(latest_pose)

        total = None
        wsum = 0.0
        total_sum = 0.0

        if voice_score is not None:
            total_sum += float(voice_score) * self.w_voice
            wsum += self.w_voice
        if pose_score is not None:
            total_sum += float(pose_score) * self.w_pose
            wsum += self.w_pose

        if wsum > 0:
            total = total_sum / wsum

        avg_voice = self.avg_voice.update(voice_score) if voice_score is not None else self.avg_voice.value()
        avg_pose = self.avg_pose.update(pose_score) if pose_score is not None else self.avg_pose.value()
        avg_total = self.avg_total.update(total) if total is not None else self.avg_total.value()

        fixes = []
        if voice_fb:
            fixes.append(voice_fb)
        if pose_fb:
            fixes.append(pose_fb)
        total_fb = " / ".join(fixes[:2]) if fixes else "평가 진행 중입니다."

        return {
            "now": {"voice": voice_score, "pose": pose_score, "total": total},
            "avg": {"voice": avg_voice, "pose": avg_pose, "total": avg_total},
            "feedback": {"voice": voice_fb, "pose": pose_fb, "total": total_fb},
        }


# ===============================
# 메인 실행부
# ===============================
def main():
    emotion_detector = None

    # 반드시 이 순서로 시작
    start_camera_thread()
    start_pose_thread()
    start_gaze_thread()
    start_expression_thread(emotion_detector)
    start_hands_thread()
    start_voice_thread()

    print("\n🚀 AI Mock Interview — Main Started (q 또는 X로 종료)\n", flush=True)

    latest_pose = None
    latest_gaze = None
    latest_expr = None
    latest_hands = None
    latest_voice = None

    window_name = "AI Mock Interview - Dashboard"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, np.zeros((10, 10, 3), dtype=np.uint8))
    cv2.waitKey(1)

    # 질문 생성 상태
    last_voice_text = None
    last_question_time = 0.0
    QUESTION_COOLDOWN = 3.0

    latest_question = "간단히 자기소개 부탁드립니다."
    printed_start_question = False

    # ✅ 평가기 생성 (먼저!)
    eval_aggr = EvalAggregatorVP(w_voice=0.5, w_pose=0.5)
    eval_out = None

    while True:
        # ===== 최신 데이터 수신 =====
        pose_data = drain_queue(pose_result_queue)
        gaze_data = drain_queue(gaze_result_queue)
        expr_data = drain_queue(expression_result_queue)
        hands_data = drain_queue(hands_result_queue)
        voice_data = drain_queue(voice_result_queue)

        if pose_data is not None:
            latest_pose = pose_data
        if gaze_data is not None:
            latest_gaze = gaze_data
        if expr_data is not None:
            latest_expr = expr_data
        if hands_data is not None:
            latest_hands = hands_data
        if voice_data is not None:
            latest_voice = voice_data

        # ✅ 평가 업데이트 (매 루프)
        eval_out = eval_aggr.update(latest_voice, latest_pose)

        # ============================
        # 대시보드 화면
        # ============================
        dashboard = np.zeros((800, 1200, 3), dtype=np.uint8)

        cv2.putText(
            dashboard,
            "AI Mock Interview Dashboard",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # ========== 자세 =============
        if latest_pose is not None:
            frame, motion, _ = latest_pose
            f = cv2.resize(frame, (350, 250))
            dashboard[80:330, 20:370] = f
            cv2.putText(
                dashboard,
                f"Movement: {motion:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # ========== 시선 =============
        if latest_gaze is not None:
            frame, g = latest_gaze
            f = cv2.resize(frame, (350, 250))
            dashboard[80:330, 400:750] = f
            cv2.putText(
                dashboard,
                f"Gaze: {g['left_right']} / {g['up_down']}",
                (400, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # ========== 손 =============
        if isinstance(latest_hands, np.ndarray):
            f = cv2.resize(latest_hands, (350, 250))
            dashboard[350:600, 400:750] = f

        # ========== 표정 =============
        if latest_expr is not None and isinstance(latest_expr[1], dict):
            emo = latest_expr[1]
            cv2.putText(
                dashboard,
                f"Expression: {emo.get('dominant', '-')}",
                (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        # ========== 음성 =============
        text = None
        if latest_voice is not None and isinstance(latest_voice, dict):
            text = latest_voice.get("text")

        safe_text = text[:50] if text else "(음성 인식 실패)"
        dashboard = put_korean_text(
            dashboard,
            f"Voice: {safe_text}",
            20,
            415,
            font_size=24,
            color=(255, 255, 255),
        )

        # ============================
        # OpenAI 질문 생성(음성 들어오면)
        # ============================
        now = time.time()
        if text and (text != last_voice_text) and ((now - last_question_time) > QUESTION_COOLDOWN):
            last_voice_text = text
            last_question_time = now
            try:
                latest_question = make_question(text)
                print("🤖 생성된 질문:", latest_question)
            except Exception as e:
                latest_question = "(질문 생성 실패)"
                print("🔥 질문 생성 오류:", e)

        # ============================
        # 평가(Evaluation) - Voice+Pose
        # ============================
        now_total = eval_out["now"]["total"]
        avg_total = eval_out["avg"]["total"]

        score_disp = "-" if now_total is None else f"{now_total:.1f}"
        avg_disp = f"{avg_total:.1f}"

        cv2.putText(
            dashboard,
            f"Eval (V+P): {score_disp}   Avg: {avg_disp}",
            (20, 480),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        dashboard = put_korean_text(
            dashboard,
            f"Eval: {eval_out['feedback']['total'][:80]}",
            20,
            510,
            font_size=26,
            color=(255, 255, 255),
        )

        # ============================
        # 질문 표시(한글은 PIL로)
        # ============================
        if latest_question:
            q = latest_question[:70]
            dashboard = put_korean_text(
                dashboard,
                f"Next Q: {q}",
                20,
                600,
                font_size=30,
                color=(0, 255, 255),
            )

        # ============================
        # 창 표시
        # ============================
        try:
            cv2.imshow(window_name, dashboard)
        except cv2.error:
            print("🔥 imshow error — window closed")
            break

        if not printed_start_question:
            print("🤖 시작 질문:", latest_question)
            printed_start_question = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🔥 Window closed by user.")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("🔚 'q' pressed. Exiting.")
            break

        time.sleep(0.01)
    # ============================
    # 🔥 최종 결과 콘솔 출력 (Voice + Pose)
    # ============================
    final_voice = eval_aggr.avg_voice.value()
    final_pose = eval_aggr.avg_pose.value()

    print(f"음성 평균 점수 : {final_voice:.1f} 점", flush=True)
    print(f"자세 평균 점수 : {final_pose:.1f} 점", flush=True)

    if eval_out:
        print("\n📌 마지막 피드백:", flush=True)
        print("-", eval_out["feedback"]["total"], flush=True)

    print("==============================\n", flush=True)
    # ============================ㅎㅎ
    # 전체 스레드 종료
    # ============================
    import modules.shared_flags as flags
    flags.RUNNING = False

    cv2.destroyAllWindows()
    print("🧹 Threads stopped.")
    return


if __name__ == "__main__":
    main()