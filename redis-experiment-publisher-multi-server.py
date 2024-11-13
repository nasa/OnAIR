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

# Initialize the Redis connection for server #1
redis_host = "localhost"
redis_port = 6380
# When your Redis server requires a password, fill it in here
redis_password = ""
# Connect to Redis
r2 = redis.Redis(
    host=redis_host, port=redis_port, password=redis_password, decode_responses=True
)

# List of channel names
server1_channels = ["state_0"]
server2_channels = ["state_1", "state_2"]


# Publish messages on each channel in random order
def publish_messages():
    loop_count = 0
    inner_loop_count = 0
    max_loops = 9
    while loop_count < max_loops:
        random.shuffle(server1_channels)
        for channel in server1_channels:
            r1.publish(
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

        random.shuffle(server2_channels)
        for channel in server2_channels:
            r2.publish(
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
