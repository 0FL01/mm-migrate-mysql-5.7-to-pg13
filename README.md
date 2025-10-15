How its works?

Install go 1.24

Install migration agent

`go install github.com/mattermost/migration-assist/cmd/migration-assist@latest`

setup environment in migrate.py for your setup

place mattermost.sql from mysql 5.7 to work dir

prepare docker-compose.yml (script use docker enviroment)

run `python3 migrate.py` and wait

you get dump mattermost.sql prepare to import in postgres13
