import redis

# Connect to Redis
r = redis.Redis()

# Use the SCAN command to retrieve keys matching the pattern
pattern = "job*"
cursor = "0"
keys_deleted = 0

while cursor != 0:
    cursor, keys = r.scan(cursor=cursor, match=pattern)
    keys_deleted += len(keys)
    if keys:
        r.delete(*keys)

r.delete("jobCounter")

print(f"Deleted {keys_deleted} keys matching the pattern '{pattern}'.")
