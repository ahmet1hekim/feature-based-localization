import queue
import threading
import dearpygui.dearpygui as dpg

from fbl.core.engine import SimEngine
from fbl.vo.tracker import VoNode
from fbl.vo.matchers import SuperGlueMatcher
from fbl.navigation.planner_node import NavigationNode
from fbl.ui.app import Application

def main():
    frame_queue = queue.Queue(maxsize=1)
    match_queue = queue.Queue(maxsize=4)
    pose_state: dict = {}
    pose_lock = threading.Lock()
    stop_event = threading.Event()

    engine = SimEngine("colsehir.jpg", frame_queue)
    init_x = engine.bg_w / 2.0
    init_y = engine.bg_h / 2.0

    app = Application(engine, pose_state, pose_lock, match_queue)
    app.setup()

    vo_t = VoNode(
        matcher=SuperGlueMatcher(),
        frame_queue=frame_queue,
        pose_state=pose_state,
        pose_lock=pose_lock,
        match_queue=match_queue,
        stop_event=stop_event,
        start_x=init_x,
        start_y=init_y,
    )
    
    plan_t = NavigationNode(
        pose_state=pose_state,
        pose_lock=pose_lock,
        get_waypoints_callback=lambda: engine.waypoints,
        remove_waypoint_callback=engine.remove_waypoint,
        is_running_callback=lambda: engine.is_running,
        apply_cmd_callback=engine.apply_autopilot_cmd,
        stop_event=stop_event
    )

    vo_t.start()
    plan_t.start()
    print("[main] Threads started.")

    try:
        while dpg.is_dearpygui_running():
            app.run_frame()
    except KeyboardInterrupt:
        print("\n[main] Ctrl-C.")
    finally:
        stop_event.set()
        vo_t.join(timeout=3)
        plan_t.join(timeout=3)
        app.teardown()
        print("[main] Done.")

if __name__ == "__main__":
    main()
