import redis
import time
import random

# Initialize the Redis connection for server #1
redis_host = "localhost"
redis_port = 6379
# When your Redis server requires a password, fill it in here
redis_password = ""
# Connect to Redis
r1 = redis.Redis(
    host=redis_host, port=redis_port, password=redis_password, decode_responses=True
)

# Initialize the Redis connection for server #2
redis_host = "localhost"
redis_port = 6380
# When your Redis server requires a password, fill it in here
redis_password = ""
# Connect to Redis
r2 = redis.Redis(
    host=redis_host, port=redis_port, password=redis_password, decode_responses=True
)

# List of all channels
all_channels = ["state_0", "state_1", "state_2"]


# Publish messages on each channel in random order
def publish_messages():
    loop_count = 0
    inner_loop_count = 0
    max_loops = 9
    while loop_count < max_loops:
        # Shuffle all channels
        random.shuffle(all_channels)
        for channel in all_channels:
            # Choose the appropriate Redis connection based on the channel
            r = r1 if channel == "state_0" else r2

            r.publish(
                channel,
                f'{{"time":{inner_loop_count}, '
                f'"x":{inner_loop_count+0.1}, '
                f'"y":{inner_loop_count+0.2}}}',
            )

            print(
                f"Published data to {channel}, "
                f"[{inner_loop_count}, "
                f"{inner_loop_count+0.1}, "
                f"{inner_loop_count+0.2}]"
            )

            inner_loop_count += 1
            time.sleep(2)

        loop_count += 1
        print(f"Completed {loop_count} loops")


if __name__ == "__main__":
    publish_messages()
