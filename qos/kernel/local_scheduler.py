from multiprocessing import Process
import qos.database as database
import redis


class local_scheduler:
    def __init__(self, qpuId: int) -> None:

        with redis.Redis() as db:
            # Subscribe to a specific key
            keywatch = database.qpuIdGen(qpuId)

            channel = f"key_update:{keywatch}"
            pubsub = db.pubsub()
            pubsub.subscribe(channel)

            # Process notifications
            for message in pubsub.listen():
                if message["type"] == "message":
                    key = (
                        message["channel"].decode().split(":")[1]
                    )  # Extract the key from the channel name
                    print(f"Key '{key}' has been modified.")

                    # TODO: Perform your desired action here based on the key modification
