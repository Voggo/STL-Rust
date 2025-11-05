#!/usr/bin/env python3
"""
Real-time MQTT plotter for:
 - stl/signals/<name>  (JSON: {"signal":"x","value":..., "timestamp_secs":...})
 - stl/robustness      (JSON: {"outputs":[{"signal":"output","value":..., "timestamp_secs":...}, ...], "timestamp_secs":...})

Usage:
    python mqtt_plot.py --host localhost --port 1883

Dependencies:
    pip install paho-mqtt matplotlib
"""
import argparse
import json
import threading
import time
from queue import Queue, Empty
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec

import paho.mqtt.client as mqtt

# max number of points kept per series
MAX_POINTS = 1000

# Queue for messages from MQTT thread -> plotting thread
msg_q = Queue()


def on_connect(client, userdata, flags, rc, *args):
    # Accept extra args to be compatible with both paho callback API versions
    print("Connected to MQTT broker, rc=", rc)
    # Subscribe to relevant topics
    client.subscribe("stl/signals/+")
    client.subscribe("stl/robustness")


def on_message(client, userdata, msg, *args):
    # Put raw payload + topic into queue for the main thread to parse/process
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print("Invalid JSON on topic", msg.topic, e)
        return
    msg_q.put((msg.topic, payload))


def mqtt_thread(broker_host, broker_port, client_id="mqtt_plotter"):
    # Create client robustly across multiple paho-mqtt versions. Try several
    # callback_api_version values until one constructs successfully. We also
    # set callbacks that accept variable args so they work with both API
    # versions.
    def try_create(api_version):
        try:
            if api_version is None:
                return mqtt.Client(client_id=client_id)
            return mqtt.Client(client_id=client_id)
        except TypeError:
            # older paho versions may use positional args
            if api_version is None:
                return mqtt.Client(client_id)
            return mqtt.Client(client_id=client_id)

    client = None
    last_exc = None
    for api in (2, 1, 0, None):
        try:
            client = try_create(api)
            # best-effort set internal flag too
            try:
                setattr(client, "_callback_api_version", api if api is not None else 1)
            except Exception:
                pass
            break
        except Exception as e:
            last_exc = e
            client = None
            continue

    if client is None:
        print("Failed to create MQTT client:", last_exc)
        return

    client.on_connect = on_connect
    client.on_message = on_message

    # Debug: print paho version and client's callback api value
    try:
        print("paho-mqtt version:", getattr(mqtt, "__version__", "unknown"))
        print("client._callback_api_version:", getattr(client, "_callback_api_version", None))
    except Exception:
        pass

    try:
        client.connect(broker_host, broker_port, keepalive=5)
    except Exception as e:
        print("Failed to connect to MQTT broker:", e)
        return

    # Blocking loop that handles reconnects and callbacks. If we hit a
    # RuntimeError due to callback API mismatch, try once more forcing
    # the older/newer version and then give up.
    # Try forcing API v1 (legacy) before starting loop to maximize compatibility.
    try:
        setattr(client, "_callback_api_version", 1)
    except Exception:
        pass

    try:
        client.loop_forever()
    except RuntimeError as e:
        print("MQTT loop_runtime error:", e)
        # attempt a single retry with the alternate flag
        try:
            setattr(client, "_callback_api_version", 0)
            client.loop_forever()
        except Exception as e2:
            print("Retry failed, giving up:", e2)


def make_plot(window_seconds=None):
    """
    Create matplotlib figure and axes.
    Returns (fig, ax_signals, ax_robustness)
    """
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)
    ax_signals = fig.add_subplot(gs[0])
    ax_rob = fig.add_subplot(gs[1], sharex=ax_signals)
    ax_signals.set_title("Signals (real-time)")
    ax_signals.set_ylabel("Value")
    ax_rob.set_title("Robustness (real-time)")
    ax_rob.set_ylabel("Robustness")
    ax_rob.set_xlabel("Time (s)")
    return fig, ax_signals, ax_rob


def run_plot(broker_host="localhost", broker_port=1883, window_seconds=None, refresh_ms=500):
    # storage: dynamic per-signal
    signals = defaultdict(lambda: {"t": deque(maxlen=MAX_POINTS), "v": deque(maxlen=MAX_POINTS)})
    # robustness: could contain entries for different output names, keep per signal name
    robustness = defaultdict(lambda: {"t": deque(maxlen=MAX_POINTS), "v": deque(maxlen=MAX_POINTS)})

    fig, ax_signals, ax_rob = make_plot(window_seconds)

    # lines storage for updating (signal_name -> Line2D)
    signal_lines = {}
    robustness_lines = {}
    # single fill for area <= 0.0 on robustness axis
    robustness_zero_fill = None

    # start MQTT in background thread
    thr = threading.Thread(target=mqtt_thread, args=(broker_host, broker_port), daemon=True)
    thr.start()

    start_time = time.time()

    def process_queue():
        """
        Drain queue and update data structures.
        """
        while True:
            try:
                topic, payload = msg_q.get_nowait()
            except Empty:
                break

            if topic.startswith("stl/signals/"):
                # Expect payload like: {"signal":"x","value":1.23,"timestamp_secs":2.0}
                name = payload.get("signal") or topic.split("/")[-1]
                ts = payload.get("timestamp_secs")
                val = payload.get("value")
                if ts is None or val is None:
                    continue
                # ensure numeric
                try:
                    tsf = float(ts)
                    vf = float(val)
                except Exception:
                    continue
                signals[name]["t"].append(tsf)
                signals[name]["v"].append(vf)

            elif topic == "stl/robustness":
                # Expect payload like: {"outputs":[{"signal":"output","value":8.0,"timestamp_secs":...}, ...], "timestamp_secs":...}
                outputs = payload.get("outputs", [])
                for entry in outputs:
                    name = entry.get("signal", "output")
                    val = entry.get("value")
                    ts = entry.get("timestamp_secs")
                    if val is None or ts is None:
                        # skip null robustness
                        continue
                    try:
                        tsf = float(ts)
                        vf = float(val)
                    except Exception:
                        continue
                    robustness[name]["t"].append(tsf)
                    robustness[name]["v"].append(vf)

    def animate(frame):
        nonlocal robustness_zero_fill
        process_queue()

        # Update signal lines
        # add new lines if new signal names appear
        for name, data in list(signals.items()):
            if name not in signal_lines:
                line, = ax_signals.plot([], [], label=name, marker="o", markersize=3, linewidth=1)
                signal_lines[name] = line
                ax_signals.legend(loc="upper left")
            xs = data["t"]
            ys = data["v"]
            if len(xs) == 0:
                signal_lines[name].set_data([], [])
            else:
                signal_lines[name].set_data(xs, ys)

        # Update robustness lines
        for name, data in list(robustness.items()):
            if name not in robustness_lines:
                line, = ax_rob.plot([], [], label=name, marker="x", markersize=4, linewidth=1, linestyle="-")
                robustness_lines[name] = line
                ax_rob.legend(loc="upper left")
            xs = data["t"]
            ys = data["v"]
            if len(xs) == 0:
                robustness_lines[name].set_data([], [])
            else:
                robustness_lines[name].set_data(xs, ys)

        # Autoscale axes and optionally show only the last window_seconds seconds
        all_times = []
        for d in signals.values():
            all_times.extend(d["t"])
        for d in robustness.values():
            all_times.extend(d["t"])

        if all_times:
            tmin = min(all_times)
            tmax = max(all_times)
            if window_seconds:
                tmin = max(tmax - window_seconds, tmin)
            # set x limits with a bit of padding
            pad = max(0.1, (tmax - tmin) * 0.05) if tmax > tmin else 1.0
            ax_signals.set_xlim(tmin - pad, tmax + pad)
            ax_rob.set_xlim(tmin - pad, tmax + pad)


        # y autoscale per axis for signals (keep default behavior)
        ax_signals.relim()
        ax_signals.autoscale_view(scalex=False, scaley=True)

        # compute robustness y-limits manually to avoid drifting caused by
        # repeatedly adding/removing artists (axhspan/polys) which can affect
        # autoscaling. This uses only the numeric robustness data.
        all_r_vals = []
        for d in robustness.values():
            all_r_vals.extend(d["v"])

        if all_r_vals:
            rmin = min(all_r_vals)
            rmax = max(all_r_vals)
            # ensure zero is visible so the shaded area can appear
            rmin = min(rmin, 0.0)
            # padding: small fraction of span, or a default if flat
            pad = max(0.1, (rmax - rmin) * 0.05) if rmax > rmin else 1.0
            ax_rob.set_ylim(rmin - pad, rmax + pad)
        else:
            # reasonable default when no robustness data yet
            ax_rob.set_ylim(-1.0, 1.0)

        # draw a single translucent red band for the area at or below 0.0
        try:
            # remove previous fill if present
            if robustness_zero_fill is not None:
                try:
                    robustness_zero_fill.remove()
                except Exception:
                    pass
                robustness_zero_fill = None

            ylim = ax_rob.get_ylim()
            # draw the band from the bottom of the axis up to 0.0 (if 0 is in range)
            if ylim[0] <= 0.0:
                # ensure band doesn't extend above the axis top if top < 0
                top = min(0.0, ylim[1])
                robustness_zero_fill = ax_rob.axhspan(ylim[0], top, color="red", alpha=0.25, zorder=0)
        except Exception:
            pass

        return list(signal_lines.values()) + list(robustness_lines.values())

    # Disable frame data caching to avoid an unbounded cache warning when frames=None
    ani = animation.FuncAnimation(fig, animate, interval=refresh_ms, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MQTT real-time plotter for STL demo")
    parser.add_argument("--host", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--window", type=float, default=30.0, help="time window in seconds to display (default 30s)")
    parser.add_argument("--refresh", type=int, default=10, help="plot refresh interval in ms")
    args = parser.parse_args()

    run_plot(broker_host=args.host, broker_port=args.port, window_seconds=args.window, refresh_ms=args.refresh)