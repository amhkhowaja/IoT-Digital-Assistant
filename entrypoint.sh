#!/bin/bash

# Start the MongoDB server
mongod --fork --logpath /var/log/mongodb.log

# Restore the database
mongorestore --db IOTA IOTA/

# Keep the MongoDB server running
# mongod --shutdown
# exec mongod
tail -f /dev/null
