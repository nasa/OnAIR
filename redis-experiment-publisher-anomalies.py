#!/usr/bin/env python3
import argparse, json, random, time
import redis

# Create two Redis clients (different ports)
r1 = redis.Redis(host="localhost", port=6379, password="", decode_responses=True)
r2 = redis.Redis(host="localhost", port=6380, password="", decode_responses=True)

# Available channels where messages can be published
CHANNELS = ["state_0", "state_1", "state_2"]

def pick_client(channel: str):
    # Select which Redis client to use depending on the channel
    return r1 if channel == "state_0" else r2

def publish(channel: str, payload: dict):
    # Publish a payload (as JSON) to the selected channel
    client = pick_client(channel)
    client.publish(channel, json.dumps(payload))

def normal_sample(t: int):
    # Generate a normal sample with predictable values
    return {"time": t, "x": t + 0.1, "y": t + 0.2}

def anomaly_sample(t: int):
    # Generate a sample with anomalies of different types
    choice = random.choice([
        "spike_values",     # very large outliers
        "missing_field",    # missing the 'y' field
        "wrong_type",       # x stored as a string instead of number
        "negative_time",    # invalid negative time
        "frozen_values",    # values stuck at a constant point
        "swapped_fields"    # x and y swapped
    ])
    if choice == "spike_values":
        return {"time": t, "x": 1e9, "y": -1e9}
    if choice == "missing_field":
        return {"time": t, "x": t + 0.1}
    if choice == "wrong_type":
        return {"time": t, "x": f"{t+0.1}", "y": t + 0.2}
    if choice == "negative_time":
        return {"time": -t, "x": t + 0.1, "y": t + 0.2}
    if choice == "frozen_values":
        base = random.randint(0, 5)
        return {"time": t, "x": base + 0.1, "y": base + 0.2}
    if choice == "swapped_fields":
        return {"time": t, "x": t + 0.2, "y": t + 0.1}

def main():
    # Parse command-line arguments
    ap = argparse.ArgumentParser(description="Publish dataset with anomalies for testing.")
    ap.add_argument("--normals", type=int, default=100, help="Number of normal samples")
    ap.add_argument("--anomalies", type=int, default=50, help="Number of anomalous samples (unique)")
    ap.add_argument("--repeat", type=int, default=3, help="Repetitions per anomaly sample")
    ap.add_argument("--delay", type=float, default=0.1, help="Seconds between publications")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle final order of events")
    args = ap.parse_args()

    events = []
    t = 0

    # Generate normal samples
    for _ in range(args.normals):
        channel = random.choice(CHANNELS)
        events.append((channel, normal_sample(t)))
        t += 1

    # Generate anomalous samples and repeat each anomaly
    for _ in range(args.anomalies):
        channel = random.choice(CHANNELS)
        a = anomaly_sample(t)
        for _r in range(args.repeat):
            events.append((channel, a))
        t += 1

    # Shuffle all events if option is enabled
    if args.shuffle:
        random.shuffle(events)

    # Publish all events to the selected channels
    total = len(events)
    for i, (channel, payload) in enumerate(events, 1):
        publish(channel, payload)
        print(f"[{i}/{total}] -> {channel}: {payload}")
        time.sleep(args.delay)

    # Print summary of the publishing process
    print(f"Done: {total} messages published "
          f"({args.normals} normal, {args.anomalies} anomalous x{args.repeat}).")

if __name__ == "__main__":
    main()
